from agent_sdk.secrets.akv import load_akv_secrets
load_akv_secrets()

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import date as _date, datetime, timezone
from cachetools import TTLCache

import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi.util import get_remote_address

from agent_sdk.logging import configure_logging
from agent_sdk.utils.env import validate_required_env_vars
from agent_sdk.utils.validation import SAFE_SESSION_RE
from agent_sdk.metrics import metrics_response
from agent_sdk.server.app_factory import create_agent_app
from agent_sdk.server.models import HistoryResponse, SessionsHistoryRequest
from agent_sdk.server.sse import create_sse_stream
from agent_sdk.server.session import verify_session_ownership
from agents.agent import _agent_instances, get_agent, run_query, create_stream, save_memory, _news_client
from database.mongo import MongoDB
from database.profile import ONBOARDING_QUESTIONS, VALID_RISK_TOLERANCES, VALID_GOALS, VALID_KNOWLEDGE_LEVELS
from a2a_service.server import create_a2a_app
from charts.data import fetch_chart_data, VALID_PERIODS

configure_logging("agent_financials")
logger = logging.getLogger("agent_financials.api")


def _get_remote_address_or_user(request: Request) -> str:
    user_id = request.headers.get("X-User-Id")
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from agent_sdk.observability import init_sentry
    init_sentry("agent-financials")
    validate_required_env_vars(
        ["MONGO_URI", "AZURE_AI_FOUNDRY_ENDPOINT", "AZURE_AI_FOUNDRY_API_KEY",
         "MEM0_API_KEY", "PINECONE_API_KEY"],
        "agent-financials",
    )
    if not os.getenv("INTERNAL_API_KEY"):
        logger.warning("INTERNAL_API_KEY is not set — internal API is unprotected. Set this in production.")
    # Connect MCP servers on startup for both modes
    for mode in ("standard", "financial_analyst"):
        agent = get_agent(mode)
        try:
            await agent._ensure_initialized()
            if getattr(agent, '_degraded', False):
                logger.warning("Agent started in DEGRADED mode — MCP tools unavailable (mode=%s)", mode)
            else:
                logger.info("MCP servers connected, agent ready (mode=%s)", mode)
        except Exception as e:
            logger.error("Agent initialization failed (continuing without MCP, mode=%s): %s", mode, e)
    await MongoDB.ensure_indexes()
    yield
    # Disconnect MCP on shutdown for all initialized agents
    for agent in list(_agent_instances.values()):
        await agent._disconnect_mcp()
    if _news_client:
        await _news_client.aclose()
    await MongoDB.close()
    logger.info("Shutdown complete")


app, limiter = create_agent_app("Financial Agent API", lifespan, key_func=_get_remote_address_or_user)

a2a_app = create_a2a_app()
app.mount("/a2a", a2a_app.build())


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    session_id: str | None = None
    response_format: str | None = Field(
        default=None,
        pattern="^(summary|flash_cards|detailed|beginner|intermediate|expert)$",
    )
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


class InvestorProfileCreate(BaseModel):
    age: int = Field(ge=10, le=100)
    monthly_investable_inr: int = Field(ge=0)
    time_horizon_years: int = Field(ge=1, le=50)
    goals: str
    risk_tolerance: str
    knowledge_level: str

    @field_validator("goals")
    @classmethod
    def validate_goals(cls, v: str) -> str:
        if v not in VALID_GOALS:
            raise ValueError(f"Must be one of: {sorted(VALID_GOALS)}")
        return v

    @field_validator("risk_tolerance")
    @classmethod
    def validate_risk_tolerance(cls, v: str) -> str:
        if v not in VALID_RISK_TOLERANCES:
            raise ValueError(f"Must be one of: {sorted(VALID_RISK_TOLERANCES)}")
        return v

    @field_validator("knowledge_level")
    @classmethod
    def validate_knowledge_level(cls, v: str) -> str:
        if v not in VALID_KNOWLEDGE_LEVELS:
            raise ValueError(f"Must be one of: {sorted(VALID_KNOWLEDGE_LEVELS)}")
        return v


class WatchlistTickerItem(BaseModel):
    symbol: str
    entry_price: float | None = None
    added_at: str | None = None  # ISO date string

class WatchlistCreate(BaseModel):
    name: str
    tickers: list[str | WatchlistTickerItem]

class WatchlistUpdate(BaseModel):
    name: str | None = None
    tickers: list[str | WatchlistTickerItem] | None = None


class HoldingCreate(BaseModel):
    ticker: str
    quantity: float = Field(gt=0)
    avg_buy_price: float = Field(gt=0)


class HoldingUpdate(BaseModel):
    quantity: float | None = Field(default=None, gt=0)
    avg_buy_price: float | None = Field(default=None, gt=0)


class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str
    structured: dict | None = None


# ── Agent endpoints ──

@app.post("/ask", response_model=AskResponse)
@limiter.limit("30/minute")
async def ask(body: AskRequest, request: Request):
    user_id = request.headers.get("X-User-Id") or None
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    if not is_new:
        await verify_session_ownership(session_id, user_id, MongoDB)

    logger.info("POST /ask — session='%s' (%s), user='%s', query='%s'",
                session_id, "new" if is_new else "existing", user_id or "anonymous", body.query[:100])

    result = await run_query(body.query, session_id=session_id,
                             response_format=body.response_format, model_id=body.model_id,
                             mode=body.mode, user_id=user_id,
                             watchlist_id=body.watchlist_id, as_of_date=body.as_of_date)
    response = result["response"]
    steps = result["steps"]

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
    is_new = body.session_id is None
    session_id = body.session_id or MongoDB.generate_session_id()

    if not is_new:
        await verify_session_ownership(session_id, user_id, MongoDB)

    logger.info("POST /ask/stream — session='%s', user='%s', query='%s'",
                session_id, user_id or "anonymous", body.query[:100])

    stream = await create_stream(body.query, session_id=session_id,
                                 response_format=body.response_format, model_id=body.model_id,
                                 mode=body.mode, user_id=user_id,
                                 watchlist_id=body.watchlist_id, as_of_date=body.as_of_date)

    async def _on_complete(response_text: str, steps: list, plan: str | None) -> None:
        save_memory(user_id=user_id or session_id, query=body.query, response=response_text)
        await MongoDB.save_conversation(
            session_id=session_id, query=body.query, response=response_text,
            steps=steps, user_id=user_id, plan=plan,
        )

    return StreamingResponse(
        create_sse_stream(stream, session_id=session_id, query=body.query, on_complete=_on_complete),
        media_type="text/event-stream",
    )


@app.get("/history/user/me", response_model=HistoryResponse)
@limiter.limit("60/minute")
async def get_history_by_user(request: Request):
    user_id = request.headers.get("X-User-Id") or None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    logger.info("GET /history/user/me — user='%s'", user_id)
    history = await MongoDB.get_history_by_user(user_id)
    logger.info("Returning %d history entries for user='%s'", len(history), user_id)
    return HistoryResponse(session_id=user_id, history=history)


@app.get("/history/{session_id}", response_model=HistoryResponse)
@limiter.limit("60/minute")
async def get_history(request: Request, session_id: str):
    user_id = request.headers.get("X-User-Id") or None
    logger.info("GET /history — session='%s'", session_id)
    history = await MongoDB.get_history(session_id, user_id=user_id)
    logger.info("Returning %d history entries for session='%s'", len(history), session_id)
    return HistoryResponse(session_id=session_id, history=history)


@app.post("/history/sessions")
@limiter.limit("30/minute")
async def get_history_by_sessions(request: Request, body: SessionsHistoryRequest):
    user_id = request.headers.get("X-User-Id") or None
    safe_ids = [s for s in body.session_ids[:20] if isinstance(s, str) and SAFE_SESSION_RE.match(s)]
    logger.info("POST /history/sessions — %d session(s)", len(safe_ids))
    history = await MongoDB.get_history_by_sessions(safe_ids, user_id=user_id)
    return {"history": history}


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
        data = await asyncio.to_thread(fetch_chart_data, ticker, period)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    return data


# ── Quotes Endpoint ──

_SYMBOL_MAP: dict[str, str] = {
    "NIFTY 50":  "^NSEI",
    "SENSEX":    "^BSESN",
    "RELIANCE":  "RELIANCE.NS",
    "TCS":       "TCS.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "INFOSYS":   "INFY.NS",
    "WIPRO":     "WIPRO.NS",
    "BAJAJ FIN": "BAJFINANCE.NS",
}

# keyed on frozenset of requested display names; 60-second TTL
_quotes_cache: TTLCache = TTLCache(maxsize=32, ttl=60)


def _format_value(price: float, symbol: str) -> str:
    """Format price with ₹ prefix for INR symbols, plain number for indices."""
    if symbol.startswith("^"):
        return f"{price:,.0f}"
    return f"₹{price:,.2f}"


def _fetch_quotes_sync(display_names: list[str]) -> list[dict]:
    results = []
    for name in display_names:
        yf_symbol = _SYMBOL_MAP.get(name)
        if not yf_symbol:
            logger.warning("No yfinance mapping for symbol '%s'", name)
            continue
        try:
            fi = yf.Ticker(yf_symbol).fast_info
            price = float(fi.last_price) if fi.last_price else None
            prev = float(fi.previous_close) if fi.previous_close else None
            if price is None:
                continue
            change_pct = ((price - prev) / prev * 100) if prev else 0.0
            results.append({
                "symbol": name,
                "value": _format_value(price, yf_symbol),
                "change": f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
                "positive": change_pct >= 0,
            })
        except Exception as e:
            logger.warning("Quote fetch failed for '%s' (%s): %s", name, yf_symbol, e)
    return results


@app.get("/quotes")
async def get_quotes(symbols: str = ""):
    """Return current price + daily change for a comma-separated list of display-name symbols.
    Results are cached for 60 seconds. No auth required."""
    requested = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else list(_SYMBOL_MAP)
    cache_key = frozenset(requested)
    if cache_key in _quotes_cache:
        return {"quotes": _quotes_cache[cache_key], "cached": True}
    quotes = await asyncio.to_thread(_fetch_quotes_sync, requested)
    _quotes_cache[cache_key] = quotes
    return {"quotes": quotes, "cached": False}


# ── Profile Endpoints ──

def _require_user_id(request: Request) -> str:
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user_id

@app.get("/profile/onboard/start")
async def onboard_start():
    """Return the ordered onboarding questionnaire."""
    return {"questions": ONBOARDING_QUESTIONS, "total": len(ONBOARDING_QUESTIONS)}

@app.get("/profile")
async def get_profile(request: Request):
    user_id = _require_user_id(request)
    profile = await MongoDB.get_profile(user_id)
    return {"profile": profile, "onboarding_complete": profile is not None}

@app.put("/profile")
async def upsert_profile(body: InvestorProfileCreate, request: Request):
    user_id = _require_user_id(request)
    await MongoDB.upsert_profile(user_id, body.model_dump())
    logger.info("Upserted investor profile for user='%s'", user_id)
    return {"success": True}


# ── Watchlist Endpoints ──

@app.post("/watchlists")
async def create_watchlist(body: WatchlistCreate, request: Request):
    user_id = _require_user_id(request)
    raw_tickers = [t if isinstance(t, str) else t.model_dump() for t in body.tickers]
    inserted_id = await MongoDB.create_watchlist(user_id, body.name, raw_tickers)
    return {"id": inserted_id}

@app.get("/watchlists")
async def list_watchlists(request: Request):
    user_id = _require_user_id(request)
    watchlists = await MongoDB.get_watchlists(user_id)
    return {"watchlists": watchlists}

@app.get("/watchlists/{watchlist_id}/performance")
async def get_watchlist_performance(watchlist_id: str, request: Request):
    """Fetch live prices for all watchlist tickers and compute P&L vs entry_price."""
    user_id = _require_user_id(request)
    watchlist = await MongoDB.get_watchlist(user_id, watchlist_id)
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    items = MongoDB._normalize_tickers(watchlist.get("tickers", []))
    results = []
    for item in items:
        symbol = item["symbol"]
        entry_price = item.get("entry_price")
        current_price = day_change_pct = pnl_pct = None
        try:
            fi = yf.Ticker(symbol).fast_info
            current_price = round(float(fi.last_price), 2) if fi.last_price else None
            if current_price and fi.previous_close:
                day_change_pct = round(((current_price - fi.previous_close) / fi.previous_close) * 100, 2)
            if current_price and entry_price:
                pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2)
        except Exception as e:
            logger.warning("Price fetch failed for '%s': %s", symbol, e)
        results.append({
            "symbol": symbol,
            "entry_price": entry_price,
            "added_at": item.get("added_at"),
            "current_price": current_price,
            "day_change_pct": day_change_pct,
            "pnl_pct": pnl_pct,
        })

    return {
        "watchlist_id": watchlist_id,
        "name": watchlist.get("name"),
        "performance": results,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/watchlists/{watchlist_id}")
async def get_watchlist(watchlist_id: str, request: Request):
    user_id = _require_user_id(request)
    watchlist = await MongoDB.get_watchlist(user_id, watchlist_id)
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    return watchlist

@app.put("/watchlists/{watchlist_id}")
async def update_watchlist(watchlist_id: str, body: WatchlistUpdate, request: Request):
    user_id = _require_user_id(request)
    raw_tickers = (
        [t if isinstance(t, str) else t.model_dump() for t in body.tickers]
        if body.tickers is not None else None
    )
    success = await MongoDB.update_watchlist(user_id, watchlist_id, body.name, raw_tickers)
    if not success:
        raise HTTPException(status_code=404, detail="Watchlist not found or unauthorized")
    return {"success": True}

@app.delete("/watchlists/{watchlist_id}")
async def delete_watchlist(watchlist_id: str, request: Request):
    user_id = _require_user_id(request)
    success = await MongoDB.delete_watchlist(user_id, watchlist_id)
    if not success:
        raise HTTPException(status_code=404, detail="Watchlist not found or unauthorized")
    return {"success": True}


# ── Holdings Endpoints ──

@app.post("/holdings")
async def create_holding(body: HoldingCreate, request: Request):
    user_id = _require_user_id(request)
    inserted_id = await MongoDB.create_holding(user_id, body.ticker, body.quantity, body.avg_buy_price)
    return {"id": inserted_id}

@app.get("/holdings")
async def list_holdings(request: Request):
    user_id = _require_user_id(request)
    holdings = await MongoDB.get_holdings(user_id)
    return {"holdings": holdings}

@app.put("/holdings/{holding_id}")
async def update_holding(holding_id: str, body: HoldingUpdate, request: Request):
    user_id = _require_user_id(request)
    success = await MongoDB.update_holding(user_id, holding_id, body.quantity, body.avg_buy_price)
    if not success:
        raise HTTPException(status_code=404, detail="Holding not found or unauthorized")
    return {"success": True}

@app.delete("/holdings/{holding_id}")
async def delete_holding(holding_id: str, request: Request):
    user_id = _require_user_id(request)
    success = await MongoDB.delete_holding(user_id, holding_id)
    if not success:
        raise HTTPException(status_code=404, detail="Holding not found or unauthorized")
    return {"success": True}

@app.get("/holdings/performance")
async def get_portfolio_performance(request: Request):
    """Fetch live prices for all holdings and compute portfolio-level P&L."""
    user_id = _require_user_id(request)
    holdings = await MongoDB.get_holdings(user_id)
    results = []
    total_invested = 0.0
    total_current_value = 0.0
    total_day_change = 0.0

    for h in holdings:
        ticker = h["ticker"]
        quantity = h["quantity"]
        avg_buy_price = h["avg_buy_price"]
        invested = quantity * avg_buy_price
        total_invested += invested

        current_price = day_change_pct = total_pnl = total_pnl_pct = None
        current_value = None
        try:
            fi = yf.Ticker(ticker).fast_info
            current_price = round(float(fi.last_price), 2) if fi.last_price else None
            if current_price:
                current_value = round(quantity * current_price, 2)
                total_current_value += current_value
                total_pnl = round(current_value - invested, 2)
                total_pnl_pct = round(((current_price - avg_buy_price) / avg_buy_price) * 100, 2)
                if fi.previous_close:
                    day_change_pct = round(((current_price - fi.previous_close) / fi.previous_close) * 100, 2)
                    day_pnl = round(quantity * (current_price - fi.previous_close), 2)
                    total_day_change += day_pnl
        except Exception as e:
            logger.warning("Price fetch failed for '%s': %s", ticker, e)

        results.append({
            "id": h["id"],
            "ticker": ticker,
            "quantity": quantity,
            "avg_buy_price": avg_buy_price,
            "current_price": current_price,
            "total_invested": round(invested, 2),
            "current_value": current_value,
            "day_change_pct": day_change_pct,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
        })

    total_pnl = round(total_current_value - total_invested, 2) if total_current_value else None
    total_pnl_pct = round((total_pnl / total_invested) * 100, 2) if total_pnl is not None and total_invested > 0 else None
    day_change_pct = round((total_day_change / total_current_value) * 100, 2) if total_current_value > 0 else None

    return {
        "holdings": results,
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current_value, 2) if total_current_value else None,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "day_change_pct": day_change_pct,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/download/{file_id}")
async def download_report(file_id: str):
    """Download a generated investment report by its file_id."""
    result = await MongoDB.retrieve_file(file_id)
    if not result:
        raise HTTPException(status_code=404, detail="Report not found.")
    data, meta = result
    filename = meta.get("filename", "report")
    media_type = "application/pdf" if filename.endswith(".pdf") else "text/markdown"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/metrics")
async def metrics():
    content, content_type = metrics_response()
    return Response(content=content, media_type=content_type)


@app.get("/health")
async def health():
    checks: dict = {"service": "agent-financials", "status": "ok"}

    # Mem0 connectivity check - passive check only to save quota
    mem0_api_key = os.getenv("MEM0_API_KEY")
    if not mem0_api_key:
        checks["mem0"] = "unconfigured"
    else:
        # We no longer perform a live search here (client.search) as it burns quota
        # every 15 seconds during Docker health checks.
        checks["mem0"] = "configured"

    return checks


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    ssl_certfile = os.getenv("SSL_CERTFILE") or None
    ssl_keyfile = os.getenv("SSL_KEYFILE") or None
    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
