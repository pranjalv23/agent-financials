import asyncio
import logging
import os
from typing import Dict, Any

from firecrawl import FirecrawlApp
from langchain_core.tools import tool

logger = logging.getLogger("agent_financials.firecrawl_tool")

_firecrawl_app = None

def _get_app() -> FirecrawlApp:
    global _firecrawl_app
    if _firecrawl_app is None:
        _firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    return _firecrawl_app


def _firecrawl_scrape_sync(url: str) -> Dict[str, Any]:
    logger.info("Firecrawl scraping — url='%s'", url)
    try:
        app = _get_app()
        # Using scrape_url for deep single-page extraction
        # Alternatively, crawl_url could be exposed if recursive scraping is needed
        scrape_result = app.scrape(url, formats=['markdown'])
        
        return {
            "url": url,
            "markdown": scrape_result.get("markdown", ""),
            "metadata": scrape_result.get("metadata", {})
        }
    except Exception as e:
        logger.error("Firecrawl scrape error: %s", e)
        return {"error": f"Scrape failed for {url}. Error: {str(e)}"}


@tool("firecrawl_deep_scrape")
async def firecrawl_deep_scrape(url: str) -> Dict[str, Any]:
    """
    Perform a deep, comprehensive scrape of a single specific URL to extract its full markdown content.
    Ideal when the agent needs to read a long-form article, detailed documentation, or a specific blog post in its entirety.
    Unlike Tavily (which searches), this expects a direct URL to read deeply.
    """
    return await asyncio.to_thread(_firecrawl_scrape_sync, url)
