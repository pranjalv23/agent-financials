import asyncio
import json
import logging
import os
import re
import string
import time
from datetime import datetime, timezone

import httpx
from agent_sdk.agents import BaseAgent
from agent_sdk.checkpoint import AsyncMongoDBSaver
from agent_sdk.database.memory import get_memories, save_memory
from database.mongo import MongoDB
from database.profile import derive_output_mode, profile_context_summary

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
    "4. **The 'Why' (Web Search):** Use `tavily_quick_search` for recent news and sentiment. Use `firecrawl_deep_scrape` for in-depth reading of specific news URLs discovered via Tavily.\n"
    "5. **Calculators:** Use `calculate_sip_returns`, `calculate_goal_sip`, and `calculate_inflation_impact` when the user asks about SIP projections, investment goals, or inflation impact.\n\n"

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

# ── Pattern detectors ──────────────────────────────────────────

_HIGH_RISK_PATTERN = re.compile(
    r'\b(penny\s*stock|f\s*[&and]+\s*o|futures?\s*and\s*options?|options?\s*trading|'
    r'double\s*my\s*money|triple\s*returns?|intraday|derivatives?|microcap|nano[\s-]cap)\b',
    re.IGNORECASE,
)
_NEWS_QUERY_PATTERN = re.compile(
    r'\b(news|market\s+today|what\s+happened|rbi|budget|policy|rate\s+cut|'
    r'inflation|gdp|fii|dii|sensex\s+today|nifty\s+today|crash|rally|'
    r'quarterly\s+results?|earnings?)\b',
    re.IGNORECASE,
)

_NEWS_AGENT_URL = os.getenv("NEWS_AGENT_URL", "http://localhost:9004")
_INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")

# ── Jargon glossary — injected into system prompt for beginner users ──

_JARGON_GLOSSARY_INJECTION = (
    "\n\n### JARGON RULE (active — user is a beginner)\n"
    "The FIRST time any of these terms appears in your response, "
    "immediately append a parenthetical plain-English definition:\n"
    "- P/E ratio → (how many years of profits you're paying for the stock)\n"
    "- CAGR → (Compound Annual Growth Rate — average yearly growth over multiple years)\n"
    "- NAV → (Net Asset Value — the price of one unit of a mutual fund)\n"
    "- SIP → (Systematic Investment Plan — a fixed amount invested every month automatically)\n"
    "- ELSS → (Equity Linked Savings Scheme — a tax-saving mutual fund, locked for 3 years)\n"
    "- F&O → (Futures & Options — advanced, high-risk derivatives; NOT suitable for beginners)\n"
    "- XIRR → (your actual annualised return accounting for the timing of investments)\n"
    "- Sensex / Nifty → (India's main stock market indices — the overall market's scoreboard)\n"
    "- FII / DII → (Foreign / Domestic Institutional Investors — large money managers)\n"
    "- Debt fund → (a mutual fund investing in bonds — safer and more stable than equity funds)\n"
)

RESPONSE_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "beginner": (
        "\n\nRESPONSE FORMAT — BEGINNER MODE:\n"
        "1. ONE clear verdict or recommendation — no fence-sitting.\n"
        "2. Exactly 3 bullet points: plain English, no dense analysis.\n"
        "3. ONE risk warning in simple language.\n"
        "4. Use analogies (e.g. 'think of an index fund like buying a tiny slice of all 50 top Indian companies').\n"
        "5. No tables with more than 3 columns. Maximum 400 words.\n"
        "6. Apply the JARGON RULE defined in your system prompt."
    ),
    "intermediate": (
        "\n\nRESPONSE FORMAT — INTERMEDIATE MODE:\n"
        "1. A concise analytical verdict (2-3 sentences).\n"
        "2. Key metrics in a compact table with brief interpretations.\n"
        "3. Main catalyst and main risk — 1 paragraph each.\n"
        "4. Clear actionable takeaway. Maximum 600 words."
    ),
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
    "expert": "",    # alias for detailed
}


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
            streaming_nodes=None,  # Defaults to DEFAULT_STREAMING_NODES (all phases)
        )
    return _agent_instances[mode]


_TRIVIAL_FOLLOWUP_PATTERN = re.compile(
    r'^(yes|no|sure|ok|okay|please|proceed|go\s*ahead|continue|yeah|yep|thanks|thank\s*you|got\s*it|tell\s*me\s*more|no\s*thanks)$',
    re.IGNORECASE
)

def _is_trivial_followup(query: str) -> bool:
    normalized = query.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    return bool(_TRIVIAL_FOLLOWUP_PATTERN.match(normalized))


def _is_high_risk_query(query: str) -> bool:
    return bool(_HIGH_RISK_PATTERN.search(query))


def _is_news_query(query: str) -> bool:
    return bool(_NEWS_QUERY_PATTERN.search(query))


async def _fetch_news_context(query: str, session_id: str) -> str | None:
    """Call agent-news /ask and return a context string. Returns None on any failure."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_NEWS_AGENT_URL}/ask",
                json={
                    "query": query,
                    "session_id": f"fin-ctx-{session_id}",
                    "response_format": "summary",
                },
                headers={"X-Internal-API-Key": _INTERNAL_API_KEY},
            )
            resp.raise_for_status()
            news = resp.json().get("response", "").strip()
            if news:
                logger.info("Fetched news context (%d chars) for session='%s'", len(news), session_id)
                return f"LIVE NEWS CONTEXT (from agent-news):\n{news}"
    except httpx.TimeoutException:
        logger.warning("News agent timed out for session='%s'", session_id)
    except Exception as e:
        logger.warning("News agent fetch failed for session='%s': %s", session_id, e)
    return None


def _build_system_prompt(profile: dict | None) -> str:
    """Compose the final system prompt, adapting to investor knowledge level."""
    kl = (profile or {}).get("knowledge_level", "expert")
    prompt = SYSTEM_PROMPT

    prompt += (
        "\n\n### LEARNING PATH & CONTINUITY\n"
        "Review the [CONTEXT] memories for topics the user has previously explored. "
        "When a prior topic connects to the current question, reference it explicitly "
        "(e.g. 'You asked about SIPs last week — here's how that connects to ELSS tax saving.'). "
        "At the end of concept-explanations, suggest one natural next topic in one sentence.\n"
    )

    prompt += (
        "\n\n### MANDATORY DISCLAIMER\n"
        "Append to EVERY response:\n"
        "> This is educational content only and does not constitute SEBI-registered investment advice. "
        "Consult a qualified financial advisor before making investment decisions.\n"
        "For high-risk queries (F&O, penny stocks, small-caps for beginners), "
        "add a proportional risk warning before the disclaimer.\n"
    )

    if kl == "beginner":
        prompt += _JARGON_GLOSSARY_INJECTION
        prompt += (
            "\n\n### WHERE-TO-START WIZARD\n"
            "When the user asks 'where do I start' or 'how do I begin investing', "
            "walk them through this decision tree conversationally:\n"
            "1. Emergency fund (3-6 months expenses)?\n"
            "   → NO → Liquid fund or high-yield savings account first.\n"
            "   → YES → Continue.\n"
            "2. Stable monthly income?\n"
            "   → NO → Start with a Liquid Fund SIP.\n"
            "   → YES → Continue.\n"
            "3. Time horizon?\n"
            "   → <3 years → Debt mutual fund or recurring deposit.\n"
            "   → 3-5 years → Balanced/hybrid mutual fund SIP.\n"
            "   → >5 years → Nifty 50 index fund SIP (or ELSS for tax saving).\n"
            "Always close with: 'Start small — even ₹500/month via SIP is a great first step.'\n"
        )

    return prompt


async def _build_dynamic_context(
    session_id: str,
    query: str,
    response_format: str | None = None,
    user_id: str | None = None,
    as_of_date: str | None = None,
    watchlist_id: str | None = None,
    profile: dict | None = None,
) -> tuple[str, str]:
    """Build [CONTEXT] block and resolve effective response format.
    Returns (context_block_string, effective_format)."""
    mem_key = user_id or session_id
    mem_error: str | None = None
    if not _is_trivial_followup(query) and len(query.strip()) > 10:
        memories, mem_error = get_memories(user_id=mem_key, query=query)
    else:
        memories = []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year = today[:4]

    # Resolve effective output mode (explicit > profile-derived > default)
    effective_format = response_format
    if effective_format is None and profile:
        effective_format = derive_output_mode(profile)
    effective_format = effective_format or "detailed"

    parts: list[str] = []

    if as_of_date:
        parts.append(
            f"IMPORTANT: Historical analysis as of {as_of_date}. "
            f"Interpret metrics within that date's context."
        )
    else:
        parts.append(
            f"Today's date: {today}. Include current year ({year}) in tavily_quick_search queries "
            f"(e.g. 'HDFC Bank Q4 {year} results')."
        )

    if profile:
        parts.append(profile_context_summary(profile))

    if watchlist_id and user_id:
        watchlist = await MongoDB.get_watchlist(user_id, watchlist_id)
        if watchlist:
            raw = watchlist.get("tickers", [])
            symbols = [t["symbol"] if isinstance(t, dict) else t for t in raw]
            parts.append(f"User's active watchlist ('{watchlist.get('name')}'): {', '.join(symbols)}")
            logger.info("Injected watchlist '%s' into context", watchlist_id)

    if memories:
        parts.append("User context (long-term memory):\n" + "\n".join(f"- {m}" for m in memories))
        logger.info("Injected %d memories for session='%s'", len(memories), session_id)

    if mem_error:
        parts.append(f"Note: {mem_error}")
        logger.warning("Mem0 degradation for session='%s': %s", session_id, mem_error)

    if _is_news_query(query) and not _is_trivial_followup(query):
        news_ctx = await _fetch_news_context(query, session_id)
        if news_ctx:
            parts.append(news_ctx)

    if _is_high_risk_query(query) and profile and profile.get("knowledge_level") == "beginner":
        parts.append(
            "SAFETY ALERT: User's query involves high-risk instruments (F&O / penny stocks). "
            "Their profile shows BEGINNER level. You MUST: "
            "(1) warn clearly this is unsuitable for beginners, "
            "(2) explain why the risk is disproportionate to their profile, "
            "(3) suggest a safer alternative (index fund SIP / large-cap MF). "
            "Do NOT analyze the high-risk instrument as if it were appropriate."
        )

    format_instruction = RESPONSE_FORMAT_INSTRUCTIONS.get(effective_format, "")
    if format_instruction:
        parts.append(format_instruction.strip())
        logger.info("Applied response format '%s' for session='%s'", effective_format, session_id)

    context_block = "[CONTEXT]\n" + "\n\n".join(parts) + "\n[/CONTEXT]\n\n"
    return context_block, effective_format


async def run_query(query: str, session_id: str = "default",
                    response_format: str | None = None, model_id: str | None = None,
                    mode: str = "financial_analyst", user_id: str | None = None,
                    as_of_date: str | None = None, watchlist_id: str | None = None) -> dict:
    logger.info("run_query called — session='%s', user='%s', query='%s', model='%s', mode='%s', as_of=%s, watchlist=%s",
                session_id, user_id or "anonymous", query[:100], model_id or "default", mode, as_of_date, watchlist_id)

    profile: dict | None = await MongoDB.get_profile(user_id) if user_id else None

    dynamic_context, _ = await _build_dynamic_context(
        session_id, query, response_format=response_format, user_id=user_id,
        as_of_date=as_of_date, watchlist_id=watchlist_id, profile=profile,
    )
    enriched_query = dynamic_context + query

    agent = get_agent(mode)
    async with get_session_lock(session_id):
        result = await agent.arun(
            enriched_query, session_id=session_id,
            system_prompt=_build_system_prompt(profile),
            model_id=model_id, as_of_date=as_of_date,
        )
    logger.info("run_query finished — session='%s', steps: %d", session_id, len(result["steps"]))

    save_memory(user_id=user_id or session_id, query=query, response=result["response"])

    return result


async def create_stream(query: str, session_id: str = "default",
                        response_format: str | None = None, model_id: str | None = None,
                        mode: str = "financial_analyst", user_id: str | None = None,
                        as_of_date: str | None = None, watchlist_id: str | None = None):
    """Create a StreamResult for the query. Returns the stream object directly."""
    logger.info("create_stream called — session='%s', user='%s', query='%s', model='%s', mode='%s', as_of=%s, watchlist=%s",
                session_id, user_id or "anonymous", query[:100], model_id or "default", mode, as_of_date, watchlist_id)

    profile: dict | None = await MongoDB.get_profile(user_id) if user_id else None

    dynamic_context, _ = await _build_dynamic_context(
        session_id, query, response_format=response_format, user_id=user_id,
        as_of_date=as_of_date, watchlist_id=watchlist_id, profile=profile,
    )
    enriched_query = dynamic_context + query
    agent = get_agent(mode)
    return agent.astream(
        enriched_query, session_id=session_id,
        system_prompt=_build_system_prompt(profile),
        model_id=model_id, as_of_date=as_of_date,
    )
