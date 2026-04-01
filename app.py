import json
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agents.agent import _agent_instances, get_agent, run_query, stream_query, create_stream, save_memory
from database.mongo import MongoDB
from a2a_service.server import create_a2a_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_financials.api")
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect MCP servers on startup for both modes
    for mode in ("standard", "financial_analyst"):
        agent = get_agent(mode)
        await agent._ensure_initialized()
        logger.info("MCP servers connected, agent ready (mode=%s)", mode)
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

# Mount the A2A server as a sub-application
a2a_app = create_a2a_app()
app.mount("/a2a", a2a_app.build())


class AskRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None
    mode: str = "financial_analyst"  # "standard" or "financial_analyst"

    model_config = {"json_schema_extra": {"examples": [{"query": "Analyze RELIANCE.NS quarterly income statement.", "session_id": None, "response_format": "detailed", "model_id": None, "mode": "financial_analyst"}]}}


class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


# ── Agent endpoints ──

@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(request: AskRequest, http_request: Request):
    user_id = http_request.headers.get("X-User-Id") or None
    is_new = request.session_id is None
    session_id = request.session_id or MongoDB.generate_session_id()

    logger.info("POST /ask — session='%s' (%s), user='%s', query='%s'",
                session_id, "new" if is_new else "existing", user_id or "anonymous", request.query[:100])

    result = await run_query(request.query, session_id=session_id,
                             response_format=request.response_format, model_id=request.model_id,
                             mode=request.mode, user_id=user_id)
    response = result["response"]
    steps = result["steps"]

    await MongoDB.save_conversation(
        session_id=session_id,
        query=request.query,
        response=response,
        steps=steps,
        user_id=user_id,
    )

    logger.info("POST /ask complete — session='%s', response length: %d chars, tool_calls: %d",
                session_id, len(response),
                sum(1 for s in steps if s.get("action") == "tool_call"))

    return AskResponse(
        session_id=session_id,
        query=request.query,
        response=response,
    )


@app.post("/ask/stream")
@limiter.limit("30/minute")
async def ask_stream(request: AskRequest, http_request: Request):
    user_id = http_request.headers.get("X-User-Id") or None
    """Stream the agent's response as Server-Sent Events (SSE).

    Each event is a JSON object with a `text` field containing a chunk.
    The stream ends with a `[DONE]` sentinel.
    """
    session_id = request.session_id or MongoDB.generate_session_id()
    logger.info("POST /ask/stream — session='%s', user='%s', query='%s'",
                session_id, user_id or "anonymous", request.query[:100])

    stream = create_stream(request.query, session_id=session_id,
                           response_format=request.response_format, model_id=request.model_id,
                           mode=request.mode, user_id=user_id)

    async def event_stream():
        full_response = []
        try:
            try:
                async for chunk in stream:
                    full_response.append(chunk)
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
            except Exception as e:
                logger.error("Stream failed: %s", e)
                fallback = "\n\n[An error occurred while generating the response.]"
                yield f"data: {json.dumps({'text': fallback})}\n\n"
                full_response.append(fallback)

            # Save to MongoDB with steps tracked during streaming
            response_text = "".join(full_response)

            if not response_text.strip():
                fallback = "Sorry, the model returned an empty response. Please try again or switch to a different model."
                yield f"data: {json.dumps({'text': fallback})}\n\n"
                response_text = fallback

            try:
                save_memory(user_id=user_id or session_id, query=request.query, response=response_text)

                await MongoDB.save_conversation(
                    session_id=session_id,
                    query=request.query,
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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-financials"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    ssl_certfile = os.getenv("SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("SSL_KEYFILE") or None
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
