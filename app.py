import json
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.agent import _agent_instances, get_agent, run_query, stream_query, create_stream, save_memory
from database.auth import AuthDB
from database.mongo import MongoDB
from a2a_service.server import create_a2a_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_financials.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure unique index on users collection
    await AuthDB.ensure_index()
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


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user_id: str
    email: str


# ── Auth endpoints ──

@app.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    existing = await AuthDB.get_user_by_email(request.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user = await AuthDB.create_user(request.email, request.password)
    logger.info("Registered user email='%s'", user["email"])
    return AuthResponse(user_id=user["user_id"], email=user["email"])


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    user = await AuthDB.get_user_by_email(request.email)
    if not user or not AuthDB.verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    logger.info("Login user='%s'", user["user_id"])
    return AuthResponse(user_id=user["user_id"], email=user["email"])


# ── Agent endpoints ──

@app.post("/ask", response_model=AskResponse)
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
        async for chunk in stream:
            full_response.append(chunk)
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        # Save to MongoDB with steps tracked during streaming
        response_text = "".join(full_response)

        if not response_text.strip():
            fallback = "Sorry, the model returned an empty response. Please try again or switch to a different model."
            yield f"data: {json.dumps({'text': fallback})}\n\n"
            response_text = fallback

        save_memory(user_id=user_id or session_id, query=request.query, response=response_text)

        await MongoDB.save_conversation(
            session_id=session_id,
            query=request.query,
            response=response_text,
            steps=stream.steps,
            user_id=user_id,
        )

        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
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
    uvicorn.run(app, host="0.0.0.0", port=port)
