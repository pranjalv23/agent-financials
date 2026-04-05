import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date as _date, datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agents.agent import _agent_instances, get_agent, run_query, create_stream, save_memory, _fix_flash_card_format, get_session_lock
from agent_sdk.database.memory import _get_client as _get_mem0_client
from database.mongo import MongoDB
from a2a_service.server import create_a2a_app
from charts.data import fetch_chart_data, VALID_PERIODS

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)

_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logging.root.addHandler(_handler)
logger = logging.getLogger("agent_financials.api")

def get_remote_address_or_user(request: Request) -> str:
    user_id = request.headers.get("X-User-Id")
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)

limiter = Limiter(key_func=get_remote_address_or_user)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect MCP servers on startup for both modes
    for mode in ("standard", "financial_analyst"):
        agent = get_agent(mode)
        await agent._ensure_initialized()
        logger.info("MCP servers connected, agent ready (mode=%s)", mode)
    await MongoDB.ensure_indexes()
    yield
    # Disconnect MCP on shutdown for all initialized agents
    for agent in list(_agent_instances.values()):
        await agent._disconnect_mcp()
    await MongoDB.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Financial Agent API",
    description="Ask investing questions, analyze stocks, and get AI-powered financial market insights.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def verify_internal_key(request: Request, call_next):
    if request.url.path not in ["/health", "/docs", "/openapi.json", "/a2a/.well-known/agent.json"]:
        expected = os.getenv("INTERNAL_API_KEY")
        if expected and request.headers.get("X-Internal-API-Key") != expected:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Unauthorized internal access"})
    return await call_next(request)

# Mount the A2A server as a sub-application
a2a_app = create_a2a_app()
app.mount("/a2a", a2a_app.build())


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    session_id: str | None = None
    response_format: str | None = Field(default=None, pattern="^(summary|flash_cards|detailed)$")
    model_id: str | None = None
    mode: str = Field(default="financial_analyst", pattern="^(standard|financial_analyst)$")
    watchlist_id: str | None = None
    as_of_date: str | None = None

    model_config = {"json_schema_extra": {"examples": [{"query": "Analyze RELIANCE.NS quarterly income statement.", "session_id": None, "response_format": "detailed", "model_id": None, "mode": "financial_analyst"}]}}

    @field_validator("as_of_date")
    @classmethod
    def validate_as_of_date(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            parsed = _date.fromisoformat(v)
        except ValueError:
            raise ValueError("as_of_date must be a valid ISO date string (YYYY-MM-DD)")
        if parsed > _date.today():
            raise ValueError("as_of_date cannot be in the future")
        return v

class WatchlistCreate(BaseModel):
    name: str
    tickers: list[str]

class WatchlistUpdate(BaseModel):
    name: str | None = None
    tickers: list[str] | None = None


class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str
    structured: dict | None = None


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


# ── Agent endpoints ──

@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(body: AskRequest, request: Request):
    user_id = request.headers.get("X-User-Id") or None
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    logger.info("POST /ask — session='%s' (%s), user='%s', query='%s'",
                session_id, "new" if is_new else "existing", user_id or "anonymous", body.query[:100])

    result = await run_query(body.query, session_id=session_id,
                             response_format=body.response_format, model_id=body.model_id,
                             mode=body.mode, user_id=user_id,
                             watchlist_id=body.watchlist_id, as_of_date=body.as_of_date)
    response = result["response"]
    steps = result["steps"]

    await MongoDB.save_conversation(
        session_id=session_id,
        query=body.query,
        response=response,
        steps=steps,
        user_id=user_id,
    )

    logger.info("POST /ask complete — session='%s', response length: %d chars, tool_calls: %d",
                session_id, len(response),
                sum(1 for s in steps if s.get("action") == "tool_call"))

    return AskResponse(
        session_id=session_id,
        query=body.query,
        response=response,
        structured=result.get("synthesis_report"),
    )


@app.post("/ask/stream")
@limiter.limit("30/minute")
async def ask_stream(body: AskRequest, request: Request):
    user_id = request.headers.get("X-User-Id") or None
    """Stream the agent's response as Server-Sent Events (SSE).

    Each event is a JSON object with a `text` field containing a chunk.
    The stream ends with a `[DONE]` sentinel.
    """
    session_id = body.session_id or MongoDB.generate_session_id()
    logger.info("POST /ask/stream — session='%s', user='%s', query='%s'",
                session_id, user_id or "anonymous", body.query[:100])

    stream = await create_stream(body.query, session_id=session_id,
                           response_format=body.response_format, model_id=body.model_id,
                           mode=body.mode, user_id=user_id,
                           watchlist_id=body.watchlist_id, as_of_date=body.as_of_date)

    _STREAM_TIMEOUT = float(os.getenv("STREAM_TIMEOUT_SECONDS", "300"))

    _PROGRESS_PREFIX = "__PROGRESS__:"

    async def event_stream():
        full_response = []
        try:
            try:
                import asyncio
                async with get_session_lock(session_id):
                    async with asyncio.timeout(_STREAM_TIMEOUT):
                        async for chunk in stream:
                            if chunk.startswith(_PROGRESS_PREFIX):
                                # Phase progress marker — route to a separate SSE event
                                # type so the UI can display a progress indicator without
                                # polluting the conversation text.
                                label = chunk[len(_PROGRESS_PREFIX):]
                                yield f"event: progress\ndata: {json.dumps({'phase': label})}\n\n"
                            else:
                                full_response.append(chunk)
                                yield f"data: {json.dumps({'text': chunk})}\n\n"
            except TimeoutError:
                logger.error("Stream timed out after %.0fs for session='%s'", _STREAM_TIMEOUT, session_id)
                fallback = f"\n\n[Response timed out after {_STREAM_TIMEOUT:.0f} seconds. Please try a simpler query.]"
                yield f"data: {json.dumps({'text': fallback})}\n\n"
                full_response.append(fallback)
            except Exception as e:
                logger.error("Stream failed: %s", e, exc_info=True)
                fallback = "\n\n[An error occurred while generating the response.]"
                yield f"data: {json.dumps({'text': fallback})}\n\n"
                full_response.append(fallback)

            # Save to MongoDB with steps tracked during streaming
            response_text = "".join(full_response)

            if not response_text.strip():
                fallback = "Sorry, the model returned an empty response. Please try again or switch to a different model."
                yield f"data: {json.dumps({'text': fallback})}\n\n"
                response_text = fallback

            if body.response_format == "flash_cards":
                response_text = _fix_flash_card_format(response_text)

            try:
                save_memory(user_id=user_id or session_id, query=body.query, response=response_text)

                await MongoDB.save_conversation(
                    session_id=session_id,
                    query=body.query,
                    response=response_text,
                    steps=stream.steps,
                    user_id=user_id,
                )
            except Exception as e:
                logger.error("Failed to save memory/conversation: %s", e)

            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history/user/me", response_model=HistoryResponse)
async def get_history_by_user(http_request: Request):
    user_id = http_request.headers.get("X-User-Id") or None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    """Get all conversation history for the authenticated user (across all sessions)."""
    logger.info("GET /history/user/me — user='%s'", user_id)
    history = await MongoDB.get_history_by_user(user_id)
    logger.info("Returning %d history entries for user='%s'", len(history), user_id)
    return HistoryResponse(session_id=user_id, history=history)


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    logger.info("GET /history — session='%s'", session_id)
    history = await MongoDB.get_history(session_id)
    logger.info("Returning %d history entries for session='%s'", len(history), session_id)
    return HistoryResponse(session_id=session_id, history=history)


@app.get("/charts/{ticker}")
async def get_chart_data(ticker: str, period: str = "1y"):
    """Return OHLCV history and technical indicators for a ticker — no LLM involved.
    period: 1mo | 3mo | 6mo | 1y | 2y | 5y (default: 1y)
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g., RELIANCE.NS"""
    if period not in VALID_PERIODS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period '{period}'. Valid options: {sorted(VALID_PERIODS)}",
        )
    logger.info("GET /charts/%s?period=%s", ticker, period)
    try:
        data = fetch_chart_data(ticker, period)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    return data

# ── Watchlist Endpoints ──

@app.post("/watchlists")
async def create_watchlist(body: WatchlistCreate, request: Request):
    user_id = request.headers.get("X-User-Id", "anonymous")
    inserted_id = await MongoDB.create_watchlist(user_id, body.name, body.tickers)
    return {"id": inserted_id}

@app.get("/watchlists")
async def list_watchlists(request: Request):
    user_id = request.headers.get("X-User-Id", "anonymous")
    watchlists = await MongoDB.get_watchlists(user_id)
    return {"watchlists": watchlists}

@app.get("/watchlists/{watchlist_id}")
async def get_watchlist(watchlist_id: str, request: Request):
    user_id = request.headers.get("X-User-Id", "anonymous")
    watchlist = await MongoDB.get_watchlist(user_id, watchlist_id)
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    return watchlist

@app.put("/watchlists/{watchlist_id}")
async def update_watchlist(watchlist_id: str, body: WatchlistUpdate, request: Request):
    user_id = request.headers.get("X-User-Id", "anonymous")
    success = await MongoDB.update_watchlist(user_id, watchlist_id, body.name, body.tickers)
    if not success:
        raise HTTPException(status_code=404, detail="Watchlist not found or unauthorized")
    return {"success": True}

@app.delete("/watchlists/{watchlist_id}")
async def delete_watchlist(watchlist_id: str, request: Request):
    user_id = request.headers.get("X-User-Id", "anonymous")
    success = await MongoDB.delete_watchlist(user_id, watchlist_id)
    if not success:
        raise HTTPException(status_code=404, detail="Watchlist not found or unauthorized")
    return {"success": True}


@app.get("/health")
async def health():
    checks: dict = {"service": "agent-financials", "status": "ok"}

    # Mem0 connectivity check
    mem0_api_key = os.getenv("MEM0_API_KEY")
    if not mem0_api_key:
        checks["mem0"] = "unconfigured"
    else:
        try:
            client = _get_mem0_client()
            # Lightweight ping — search with an empty query just to verify the connection
            client.search(query="health check", version="v2", filters={"user_id": "__healthcheck__"}, limit=1)
            checks["mem0"] = "ok"
        except Exception as e:
            logger.warning("Mem0 health check failed: %s", e)
            checks["mem0"] = "degraded"
            checks["status"] = "degraded"

    return checks


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    ssl_certfile = os.getenv("SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("SSL_KEYFILE") or None
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
