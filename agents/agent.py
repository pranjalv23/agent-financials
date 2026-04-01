import logging
import os
from datetime import datetime, timezone

from agent_sdk.agents import BaseAgent
from agent_sdk.checkpoint import AsyncMongoDBSaver
from database.memory import get_memories, save_memory
from database.mongo import MongoDB

logger = logging.getLogger("agent_financials.agent")

# Restrict streaming to only the final synthesis output.
# Intermediate analytical phases (company_analysis, sector_analysis, etc.)
# emit pre-tool-call reasoning text that should not be shown to users.
import agent_sdk.agents.base_agent as _base_agent_module
_base_agent_module._STREAMING_NODES = frozenset({"llm_call", "synthesis"})

SYSTEM_PROMPT = (
    "You are the Lead Financial Analyst and Investing Mentor at Agent Hub.\n"
    "Your mission is to empower users with deep financial insights, bridging the gap between raw data and actionable knowledge. "
    "You analyze Indian (BSE/NSE) and global markets with local expertise and a global perspective.\n\n"

    "### YOUR VOICE & PERSONALITY\n"
    "- **Insightful, Not Just Informative:** Anyone can look up a P/E ratio. Your value is explaining *why* it matters for this specific company right now.\n"
    "- **Decision-Oriented:** Avoid fence-sitting. Provide a clear, reasoned analytical stance based on the evidence. Use phrases like 'The data strongly suggests...' or 'While X is a concern, the primary driver is Y...'.\n"
    "- **Plain-English Finance:** Translate complex jargon (EBITDA, Free Cash Flow, CAGR) into conversational analogies. Always define metrics in the context of the company's health.\n"
    "- **Mentor Vibe:** Treat the user like a partner. Explain your logic so they learn *how* to think about investing alongside the answer.\n\n"

    "### CORE ANALYTICAL WORKFLOW (STRICT ORDER)\n"
    "1. **Vector-DB First:** For any company analysis, you MUST start by calling `check_in_vector_db` to see if we already have indexed reports. If we do, use `retrieve_from_vector_db` before searching the web.\n"
    "2. **Data Enrichment:** Only if Vector-DB is empty or outdated, use `add_financial_reports_to_db` then retrieve.\n"
    "3. **Market Context:** Use `get_ticker_data` for current price and basic metrics.\n"
    "4. **The 'Why' (Web Search):** Use `tavily_quick_search` for recent news and sentiment. Use `firecrawl_deep_scrape` for in-depth reading of specific news URLs discovered via Tavily.\n\n"

    "### THE 'PREMIUM' RESPONSE STRUCTURE\n"
    "1. **The Executive Summary** — A high-level overview of the company and your immediate analytical takeaway.\n"
    "2. **Deep-Dive: Financial Health** — Metrics interpreted for a layman. E.g., 'A debt-to-equity ratio of 1.5 means the company is heavily leveraged, borrowing ₹1.5 for every ₹1 it owns. This is a red flag in a high-interest environment.'\n"
    "3. **Growth Catalysts & Moats** — What are the competitive advantages (Tailwinds)?\n"
    "4. **Strategic Risks** — What could go wrong (Headwinds)?\n"
    "5. **Analytical Verdict (Mentor's Bottom Line)** — Your clear stance. Is it an entry point? A hold? What are the key levels/factors to watch? Give specific price ranges or target conditions if possible.\n\n"

    "### CRITICAL RULES\n"
    "- **No Outdated Knowledge:** Your internal training data is stale. For news, policy, or current prices, you MUST use tools.\n"
    "- **Calculated Stance:** Do not merely summarize. Weigh the evidence and provide a definitive 'Mentor's Take'.\n"
    "- **Math Fixes:** Use markdown tables and bold headers. Ensure math is clearly delimited.\n\n"

    "### CITATIONS\n"
    "Ground every factual claim in tool results with [n] inline markers.\n"
    "Append a 'Sources' header at the end listing titles and full URLs."
)

# MCP server configuration — all tools served from a single combined MCP server
MCP_SERVERS = {
    "mcp-tool-servers": {
        "url": os.getenv("MCP_SERVER_URL", "http://localhost:8010/mcp"),
        "transport": "http",
    },
}

_agent_instances: dict[str, BaseAgent] = {}
_checkpointer: AsyncMongoDBSaver | None = None


def _get_checkpointer() -> AsyncMongoDBSaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = AsyncMongoDBSaver.from_conn_string(
            conn_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            db_name=os.getenv("MONGO_DB_NAME", "agent_financials"),
            ttl=int(os.getenv("CHECKPOINT_TTL_SECONDS", str(7 * 24 * 3600))),
        )
    return _checkpointer


def get_agent(mode: str = "financial_analyst") -> BaseAgent:
    """Return a per-mode singleton BaseAgent so the checkpointer persists across calls."""
    if mode not in _agent_instances:
        logger.info("Creating agent singleton (mode=%s) with MCP servers", mode)
        _agent_instances[mode] = BaseAgent(
            tools=[],
            mcp_servers=MCP_SERVERS,
            system_prompt=SYSTEM_PROMPT,
            provider="azure",
            checkpointer=_get_checkpointer(),
            mode=mode,
        )
    return _agent_instances[mode]


RESPONSE_FORMAT_INSTRUCTIONS = {
    "summary": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants a QUICK SUMMARY. "
        "Keep your response concise — 5-7 bullet points maximum covering the key metrics, "
        "one-line interpretation of each, and a 2-sentence bottom line. "
        "Skip detailed section headers, long explanations, and tables."
    ),
    "flash_cards": (
        "\n\nRESPONSE FORMAT OVERRIDE: The user wants INSIGHT CARDS. "
        "Format your response as a series of insight cards using this EXACT format for each card:\n\n"
        "### [Topic Label]\n"
        "**Key Insight:** [The main metric, data point, or finding — keep it short and prominent]\n"
        "[1-2 sentence explanation of what this means and why it matters]\n\n"
        "STRICT FORMATTING RULES:\n"
        "- Use exactly ### (three hashes) for each card topic — NOT ## or ####\n"
        "- Do NOT wrap topic names in **bold** — just plain text after ###\n"
        "- Do NOT use bullet points (- or *) for the Key Insight line — start it directly with **Key Insight:**\n"
        "- Every card MUST have a **Key Insight:** line\n"
        "- Start directly with the first ### card — no title header, preamble, or introductory text before the cards\n\n"
        "Generate 8-12 cards covering: key financial metrics, valuation, strengths, risks, "
        "growth prospects, and a bottom-line view."
    ),
    "detailed": "",  # default — uses the full system prompt format as-is
}


_TRIVIAL_FOLLOWUPS: frozenset[str] = frozenset({
    "yes", "no", "sure", "ok", "okay", "please", "yes please",
    "no thanks", "proceed", "go ahead", "continue", "yeah", "yep",
})


def _build_dynamic_context(session_id: str, query: str, response_format: str | None = None,
                            user_id: str | None = None) -> str:
    """Build dynamic context block (date, memories, format instructions) to prepend to the user query."""
    mem_key = user_id or session_id  # prefer stable user_id for Mem0
    # Skip Mem0 search for trivial follow-ups — "Yes" has no semantic content to match against.
    if query.strip().lower() not in _TRIVIAL_FOLLOWUPS and len(query.strip()) > 10:
        memories = get_memories(user_id=mem_key, query=query)
    else:
        memories = []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year = today[:4]

    parts = []
    parts.append(
        f"Today's date: {today}. When using tavily_quick_search include the current year "
        f"({year}) in search queries (e.g. 'HDFC Bank Q4 {year} results')."
    )

    if memories:
        memory_lines = "\n".join(f"- {m}" for m in memories)
        parts.append(f"User context (long-term memory):\n{memory_lines}")
        logger.info("Injected %d memories into context for session='%s'", len(memories), session_id)

    format_instruction = RESPONSE_FORMAT_INSTRUCTIONS.get(response_format or "detailed", "")
    if format_instruction:
        parts.append(format_instruction.strip())
        logger.info("Applied response format '%s' for session='%s'", response_format, session_id)

    context_block = "\n\n".join(parts)
    return f"[CONTEXT]\n{context_block}\n[/CONTEXT]\n\n"


async def run_query(query: str, session_id: str = "default",
                    response_format: str | None = None, model_id: str | None = None,
                    mode: str = "financial_analyst", user_id: str | None = None) -> dict:
    logger.info("run_query called — session='%s', user='%s', query='%s', model='%s', mode='%s'",
                session_id, user_id or "anonymous", query[:100], model_id or "default", mode)

    dynamic_context = _build_dynamic_context(session_id, query, response_format=response_format, user_id=user_id)
    enriched_query = dynamic_context + query

    agent = get_agent(mode)
    result = await agent.arun(enriched_query, session_id=session_id, system_prompt=SYSTEM_PROMPT, model_id=model_id)
    logger.info("run_query finished — session='%s', steps: %d", session_id, len(result["steps"]))

    save_memory(user_id=user_id or session_id, query=query, response=result["response"])

    return result


def create_stream(query: str, session_id: str = "default",
                  response_format: str | None = None, model_id: str | None = None,
                  mode: str = "financial_analyst", user_id: str | None = None):
    """Create a StreamResult for the query. Returns the stream object directly."""
    logger.info("create_stream called — session='%s', user='%s', query='%s', model='%s', mode='%s'",
                session_id, user_id or "anonymous", query[:100], model_id or "default", mode)

    dynamic_context = _build_dynamic_context(session_id, query, response_format=response_format, user_id=user_id)
    enriched_query = dynamic_context + query
    agent = get_agent(mode)
    return agent.astream(enriched_query, session_id=session_id, system_prompt=SYSTEM_PROMPT, model_id=model_id)


