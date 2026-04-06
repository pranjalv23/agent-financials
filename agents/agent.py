import asyncio
import logging
import os
import re
import string
import time
from datetime import datetime, timezone

from agent_sdk.agents import BaseAgent
from agent_sdk.checkpoint import AsyncMongoDBSaver
from agent_sdk.database.memory import get_memories, save_memory
from database.mongo import MongoDB

logger = logging.getLogger("agent_financials.agent")

# Restrict streaming to only the final synthesis output.
# Intermediate analytical phases (company_analysis, sector_analysis, etc.)
# emit pre-tool-call reasoning text that should not be shown to users.

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
    "0. **Ticker Resolution — MANDATORY first step for any Indian company:** "
    "If the user mentions any Indian company by name or abbreviation (e.g. 'SBI', 'UBI', "
    "'HDFC Bank', 'Reliance', 'State Bank') — even if it resembles a ticker symbol — call "
    "`resolve_indian_ticker` BEFORE any other tool and use the NSE ticker it returns "
    "(e.g. 'SBIN.NS') for ALL subsequent tool calls in this session. "
    "Skip this step ONLY when the user provides an explicit exchange-suffixed ticker "
    "(e.g. 'RELIANCE.NS', 'HDFCBANK.BO').\n"
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

    "### TOOL FAILURE PROTOCOL\n"
    "- If a tool returns an error or empty data, **explicitly tell the user which data source failed** — never silently omit it.\n"
    "- Suggest an alternative: e.g., 'I couldn't fetch the financial reports directly. Let me search for the latest results online.'\n"
    "- Try a fallback: use `tavily_quick_search` as a fallback for any failed data-fetching tool.\n"
    "- Clearly state which conclusions are weakened by missing data: 'Without the cash flow statement, my FCF estimate is unverified.'\n\n"

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

class LockCache:
    def __init__(self, ttl: int = 3600):
        self._locks = {}
        self._timestamps = {}
        self._ttl = ttl
    
    def get_lock(self, session_id: str) -> asyncio.Lock:
        now = time.time()
        expired = [sid for sid, ts in self._timestamps.items() if now - ts > self._ttl]
        for sid in expired:
            if sid in self._locks and not self._locks[sid].locked():
                del self._locks[sid]
                del self._timestamps[sid]
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        self._timestamps[session_id] = now
        return self._locks[session_id]

_session_locks = LockCache()

def get_session_lock(session_id: str) -> asyncio.Lock:
    return _session_locks.get_lock(session_id)


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
            streaming_nodes=frozenset({"llm_call", "synthesis"}) if mode == "financial_analyst" else None,
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


from agent_sdk.agents.formatters import _fix_flash_card_format


_TRIVIAL_FOLLOWUP_PATTERN = re.compile(
    r'^(yes|no|sure|ok|okay|please|proceed|go\s*ahead|continue|yeah|yep|thanks|thank\s*you|got\s*it|tell\s*me\s*more|no\s*thanks)$',
    re.IGNORECASE
)

def _is_trivial_followup(query: str) -> bool:
    normalized = query.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    return bool(_TRIVIAL_FOLLOWUP_PATTERN.match(normalized))


async def _build_dynamic_context(session_id: str, query: str, response_format: str | None = None,
                                user_id: str | None = None, as_of_date: str | None = None,
                                watchlist_id: str | None = None) -> str:
    """Build dynamic context block (date, memories, format instructions) to prepend to the user query."""
    mem_key = user_id or session_id  # prefer stable user_id for Mem0
    # Skip Mem0 search for trivial follow-ups
    mem_error: str | None = None
    if not _is_trivial_followup(query) and len(query.strip()) > 10:
        memories, mem_error = get_memories(user_id=mem_key, query=query)
    else:
        memories = []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year = today[:4]

    parts = []
    
    # Point-in-time context injection (PR 13)
    if as_of_date:
        parts.append(
            f"IMPORTANT: The user is requesting a historical analysis as of {as_of_date}. "
            f"While yfinance provides current data, please interpret all reported metrics "
            f"within the context of that specific date. Acknowledge that fundamental data "
            f"may be the most recent available rather than a true point-in-time snapshot."
        )
    else:
        parts.append(
            f"Today's date: {today}. When using tavily_quick_search include the current year "
            f"({year}) in search queries (e.g. 'HDFC Bank Q4 {year} results')."
        )

    # Watchlist context injection (PR 13)
    if watchlist_id and user_id:
        watchlist = await MongoDB.get_watchlist(user_id, watchlist_id)
        if watchlist:
            tickers = ", ".join(watchlist.get("tickers", []))
            parts.append(f"User's active watchlist ('{watchlist.get('name')}'): {tickers}")
            logger.info("Injected watchlist '%s' into context", watchlist_id)

    if memories:
        memory_lines = "\n".join(f"- {m}" for m in memories)
        parts.append(f"User context (long-term memory):\n{memory_lines}")
        logger.info("Injected %d memories into context for session='%s'", len(memories), session_id)

    if mem_error:
        parts.append(f"Note: {mem_error}")
        logger.warning("Mem0 degradation for session='%s': %s", session_id, mem_error)

    format_instruction = RESPONSE_FORMAT_INSTRUCTIONS.get(response_format or "detailed", "")
    if format_instruction:
        parts.append(format_instruction.strip())
        logger.info("Applied response format '%s' for session='%s'", response_format, session_id)

    context_block = "\n\n".join(parts)
    return f"[CONTEXT]\n{context_block}\n[/CONTEXT]\n\n"


async def run_query(query: str, session_id: str = "default",
                    response_format: str | None = None, model_id: str | None = None,
                    mode: str = "financial_analyst", user_id: str | None = None,
                    as_of_date: str | None = None, watchlist_id: str | None = None) -> dict:
    logger.info("run_query called — session='%s', user='%s', query='%s', model='%s', mode='%s', as_of=%s, watchlist=%s",
                session_id, user_id or "anonymous", query[:100], model_id or "default", mode, as_of_date, watchlist_id)

    dynamic_context = await _build_dynamic_context(
        session_id, query, response_format=response_format, user_id=user_id,
        as_of_date=as_of_date, watchlist_id=watchlist_id
    )
    enriched_query = dynamic_context + query

    agent = get_agent(mode)
    async with get_session_lock(session_id):
        result = await agent.arun(enriched_query, session_id=session_id, system_prompt=SYSTEM_PROMPT, model_id=model_id)
    logger.info("run_query finished — session='%s', steps: %d", session_id, len(result["steps"]))

    # Sanitize potential JSON-wrapped outputs like {"full_report": "..."}
    try:
        if isinstance(result.get("response"), str):
            maybe_json = json.loads(result["response"])  # type: ignore[arg-type]
            if isinstance(maybe_json, dict) and "full_report" in maybe_json:
                result["response"] = maybe_json.get("full_report") or result["response"]
                # Preserve the structured synthesis if not already present
                if "synthesis_report" not in result:
                    result["synthesis_report"] = maybe_json
    except Exception:
        # Not JSON — leave as-is
        pass

    if response_format == "flash_cards":
        result["response"] = _fix_flash_card_format(result["response"])

    save_memory(user_id=user_id or session_id, query=query, response=result["response"])

    return result


async def create_stream(query: str, session_id: str = "default",
                   response_format: str | None = None, model_id: str | None = None,
                   mode: str = "financial_analyst", user_id: str | None = None,
                   as_of_date: str | None = None, watchlist_id: str | None = None):
    """Create a StreamResult for the query. Returns the stream object directly."""
    logger.info("create_stream called — session='%s', user='%s', query='%s', model='%s', mode='%s', as_of=%s, watchlist=%s",
                session_id, user_id or "anonymous", query[:100], model_id or "default", mode, as_of_date, watchlist_id)

    dynamic_context = await _build_dynamic_context(
        session_id, query, response_format=response_format, user_id=user_id,
        as_of_date=as_of_date, watchlist_id=watchlist_id
    )
    enriched_query = dynamic_context + query
    agent = get_agent(mode)
    return agent.astream(enriched_query, session_id=session_id, system_prompt=SYSTEM_PROMPT, model_id=model_id)


