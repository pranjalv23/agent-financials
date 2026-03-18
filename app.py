import json
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.agent import get_agent, run_query, stream_query
from database.mongo import MongoDB
from a2a.server import create_a2a_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_financials.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect MCP servers on startup (triggers lazy init)
    agent = get_agent()
    await agent._ensure_initialized()
    logger.info("MCP servers connected, agent ready")
    yield
    # Disconnect MCP on shutdown
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

    model_config = {"json_schema_extra": {"examples": [{"query": "Analyze RELIANCE.NS quarterly income statement.", "session_id": None}]}}


class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    is_new = request.session_id is None
    session_id = request.session_id or MongoDB.generate_session_id()

    logger.info("POST /ask — session='%s' (%s), query='%s'",
                session_id, "new" if is_new else "existing", request.query[:100])

    result = await run_query(request.query, session_id=session_id)
    response = result["response"]
    steps = result["steps"]

    await MongoDB.save_conversation(
        session_id=session_id,
        query=request.query,
        response=response,
        steps=steps,
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
async def ask_stream(request: AskRequest):
    """Stream the agent's response as Server-Sent Events (SSE).

    Each event is a JSON object with a `text` field containing a chunk.
    The stream ends with a `[DONE]` sentinel.
    """
    session_id = request.session_id or MongoDB.generate_session_id()
    logger.info("POST /ask/stream — session='%s', query='%s'", session_id, request.query[:100])

    async def event_stream():
        full_response = []
        async for chunk in stream_query(request.query, session_id=session_id):
            full_response.append(chunk)
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        # Save to MongoDB after streaming completes
        response_text = "".join(full_response)
        await MongoDB.save_conversation(
            session_id=session_id,
            query=request.query,
            response=response_text,
            steps=[],
        )

        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
