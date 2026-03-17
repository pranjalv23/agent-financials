import asyncio
import logging
from typing import Dict, Any

import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger("agent_financials.yfinance_tool")


def _get_ticker_data_sync(ticker: str) -> Dict[str, Any]:
    logger.info("Fetching market data for ticker='%s'", ticker)
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Extract the most relevant fields
        relevant_keys = [
            "shortName", "symbol", "currentPrice", "marketCap", "sector", "industry", 
            "trailingPE", "forwardPE", "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "longBusinessSummary"
        ]
        
        data = {k: info.get(k) for k in relevant_keys}
        return data
    except Exception as e:
        logger.error("Error fetching data for ticker='%s': %s", ticker, e)
        return {"error": f"Failed to fetch data for {ticker}. Error: {str(e)}"}


@tool("get_ticker_data")
async def get_ticker_data(ticker: str) -> Dict[str, Any]:
    """
    Get basic market data and company information for a given ticker symbol.
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g., 'RELIANCE.NS'.
    Returns current price, market cap, P/E ratios, and a business summary.
    """
    return await asyncio.to_thread(_get_ticker_data_sync, ticker)
