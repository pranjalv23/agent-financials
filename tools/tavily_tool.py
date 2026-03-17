import asyncio
import logging
import os
from typing import List, Dict, Any

from tavily import TavilyClient
from langchain_core.tools import tool

logger = logging.getLogger("agent_financials.tavily_tool")

_tavily_client = None

def _get_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily_client


def _tavily_search_sync(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    logger.info("Tavily search — query='%s', max_results=%d", query, max_results)
    try:
        client = _get_client()
        response = client.search(
            query=query, 
            search_depth="advanced", 
            max_results=max_results,
            include_answer=True
        )
        
        results = [{"title": r["title"], "url": r["url"], "content": r["content"]} for r in response.get("results", [])]
        if response.get("answer"):
            results.insert(0, {"title": "AI Answer Synthesis", "url": "N/A", "content": response["answer"]})
            
        return results
    except Exception as e:
        logger.error("Tavily search error: %s", e)
        return [{"error": f"Search failed: {str(e)}"}]


@tool("tavily_quick_search")
async def tavily_quick_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Perform a quick and generalized web search across the internet to answer questions.
    Ideal for 'How to invest', 'Where to start', broad concepts, or current news.
    Returns synthesized answers and snippets from top sources.
    """
    return await asyncio.to_thread(_tavily_search_sync, query, max_results)
