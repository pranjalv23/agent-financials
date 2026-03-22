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
    "You are an expert financial analyst and investing mentor.\n"
    "You help users — from complete beginners to experienced investors — understand markets, "
    "analyze companies, and make informed investment decisions. You cover Indian BSE/NSE stocks "
    "as well as global markets.\n\n"

    "YOUR PERSONALITY AND APPROACH:\n"
    "- You are NOT a dry data reader. You are an insightful analyst who interprets data and explains what it MEANS.\n"
    "- When presenting financial metrics (P/E ratio, debt-to-equity, revenue growth, etc.), always explain \n"
    "  what the number means in plain English and whether it is good, bad, or neutral — and WHY.\n"
    "- Tailor your language so that a layman can understand the key takeaways, while an investor \n"
    "  gets the analytical depth they need.\n"
    "- DO NOT be overly neutral or fence-sitting. Based on the overall picture — fundamentals, risks, \n"
    "  macro environment, and future outlook — provide a clear, reasoned view. Take a stance and \n"
    "  explain the reasoning behind it in a practical, decision-oriented way.\n"
    "- Always end with a concrete 'Bottom Line' section that gives actionable insight.\n\n"

    "You have access to the following tools:\n"
    "- `get_ticker_data`: Fetch basic market data, price, P/E ratios, and company summary. Use .NS or .BO suffix for NSE/BSE stocks.\n"
    "- `tavily_quick_search`: QUICK web search returning short snippets and an AI-synthesized answer. Use for recent news, headlines, and quick fact-checking.\n"
    "- `firecrawl_deep_scrape`: DEEP scrape of a specific URL returning full markdown content. Use when Tavily finds a promising URL that needs full reading.\n"
    "- `check_in_vector_db`: Check if financial reports exist in the vector DB (identifier=ticker, index_name='financial-reports').\n"
    "- `add_financial_reports_to_db`: Fetch and store a company's quarterly and yearly financial reports in the vector DB.\n"
    "- `retrieve_from_vector_db`: Retrieve financial chunks (index_name='financial-reports', filter_key='ticker', filter_value=ticker).\n"
    "- `get_bse_nse_reports`: Direct fetch of raw reports (use sparingly — prefer `add_financial_reports_to_db` + `retrieve_from_vector_db`).\n\n"

    "Tool Usage:\n"
    "- Company deep-dive: `get_ticker_data` first → `check_in_vector_db` → `add_financial_reports_to_db` if not stored → `retrieve_from_vector_db` for specific chunks.\n"
    "- Research and news: `tavily_quick_search` first to discover sources → `firecrawl_deep_scrape` on the best URL for full content. Use for macro-economic outlook, policies, sector trends, recent news, and general investing questions.\n\n"

    "RESPONSE FORMAT RULES:\n"
    "1. Structure your response with clear sections using markdown headers. Recommended structure for stock analysis:\n"
    "   - **Company Snapshot** — What the company does, in one or two sentences a layman can understand.\n"
    "   - **Key Financial Health** — Important metrics with plain-English interpretation (e.g., 'A P/E of 45 means investors are paying ₹45 for every ₹1 of earnings — this is expensive compared to the sector average of 25, suggesting the market expects high future growth.').\n"
    "   - **Strengths & Tailwinds** — What's going well, competitive advantages, positive catalysts.\n"
    "   - **Risks & Red Flags** — Debt concerns, governance issues, sector headwinds, regulatory risks.\n"
    "   - **Macro & External Factors** — How geopolitics, government policy, or global trends affect this stock.\n"
    "   - **Bottom Line** — Your clear, reasoned stance: Is this a good investment right now? For whom? At what price range? What should someone do if they already own it?\n"
    "2. Use analogies and comparisons where helpful (e.g., 'Think of this like renting vs buying a house').\n"
    "3. Explain jargon naturally within sentences instead of assuming the user knows terms like 'EBITDA' or 'free cash flow'.\n"
    "4. When comparing to sector peers, name specific competitors and explain how the company stacks up.\n\n"

    "CRITICAL — NEVER RELY ON YOUR OWN KNOWLEDGE FOR NEWS, CURRENT AFFAIRS, OR REAL-TIME INFORMATION:\n"
    "Your training data is outdated. For ANY question involving recent news, current events, market sentiment, "
    "government policy changes, earnings announcements, regulatory updates, geopolitical developments, or "
    "anything that could have changed after your knowledge cutoff — you MUST use your search tools.\n"
    "- Use `tavily_quick_search` for quick lookups: headlines, recent developments, sentiment, quick fact-checks.\n"
    "- Use `firecrawl_deep_scrape` when a query demands in-depth analysis: full articles, research reports, "
    "earnings transcripts, or detailed breakdowns. Typically you will first discover a relevant URL via "
    "`tavily_quick_search`, then scrape it with `firecrawl_deep_scrape` for the complete content.\n"
    "- When in doubt, SEARCH FIRST. It is always better to verify with a tool than to guess from memory.\n\n"

    "IMPORTANT: Never invent financial data. If you don't know the answer, use your search tools. "
    "Synthesize the data intelligently and explain your reasoning clearly. "
    "Always ground your opinions in data — never give baseless recommendations."
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


def _build_dynamic_context(session_id: str, query: str, response_format: str | None = None,
                            user_id: str | None = None) -> str:
    """Build dynamic context block (date, memories, format instructions) to prepend to the user query."""
    mem_key = user_id or session_id  # prefer stable user_id for Mem0
    memories = get_memories(user_id=mem_key, query=query)

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


async def stream_query(query: str, session_id: str = "default"):
    """Async generator that yields text chunks for SSE streaming."""
    logger.info("stream_query called — session='%s', query='%s'", session_id, query[:100])

    dynamic_context = _build_dynamic_context(session_id, query)
    enriched_query = dynamic_context + query

    agent = get_agent()
    full_response = []

    async for chunk in agent.astream(enriched_query, session_id=session_id, system_prompt=SYSTEM_PROMPT):
        full_response.append(chunk)
        yield chunk

    # Save the complete response to Mem0 after streaming finishes
    response_text = "".join(full_response)
    save_memory(user_id=session_id, query=query, response=response_text)
    logger.info("stream_query finished — session='%s', response length: %d chars",
                session_id, len(response_text))
