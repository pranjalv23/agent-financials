import asyncio
import logging
from typing import List, Dict, Any

from langchain_core.tools import tool

from database.vector_db import VectorDB
from tools.bse_nse_reports import _get_financial_reports_sync

logger = logging.getLogger("agent_financials.db_tools")


@tool("check_reports_in_db")
async def check_reports_in_db(ticker: str) -> Dict[str, Any]:
    """
    Check if financial reports for a given ticker already exist in the vector database.
    Use this to verify if the data is already available before calling add_reports_to_db.
    """
    return await asyncio.to_thread(_check_reports_in_db_sync, ticker)

def _check_reports_in_db_sync(ticker: str) -> Dict[str, Any]:
    logger.info("Checking if reports exist in Vector DB for ticker='%s'", ticker)
    try:
        db = VectorDB()
        if db.reports_exist(ticker):
            return {"exists": True, "message": f"Reports for {ticker} exist in the database."}
        else:
            return {"exists": False, "message": f"No reports found for {ticker} in the database."}
    except Exception as e:
        logger.error("Error checking reports in vector DB: %s", e)
        return {"exists": False, "error": f"Database error: {str(e)}"}

def _add_reports_to_db_sync(ticker: str) -> Dict[str, Any]:
    logger.info("Adding reports to Vector DB for ticker='%s'", ticker)
    try:
        db = VectorDB()
        if db.reports_exist(ticker):
            logger.info("Reports already exist for %s, skipping.", ticker)
            return {"status": "success", "message": f"Reports for {ticker} already exist in the database."}

        reports = _get_financial_reports_sync(ticker)
        
        # Check if the fetch failed
        if reports and "error" in reports[0]:
            return {"status": "error", "message": reports[0]["error"]}
            
        if not reports:
            return {"status": "error", "message": f"No financial reports found for {ticker}."}

        db.upsert_reports(ticker, reports)
        return {"status": "success", "message": f"Successfully loaded {len(reports)} reports into the Vector DB for {ticker}."}
        
    except Exception as e:
        logger.error("Error adding reports to vector DB: %s", e)
        return {"status": "error", "message": f"Database error: {str(e)}"}


@tool("add_reports_to_db")
async def add_reports_to_db(ticker: str) -> Dict[str, Any]:
    """
    Fetch quarterly and yearly reports for a given ticker and add them to the vector database for later semantic retrieval.
    Always call this BEFORE trying to analyze dense financial data if it's not already in the database.
    """
    return await asyncio.to_thread(_add_reports_to_db_sync, ticker)


def _retrieve_reports_sync(query: str, ticker: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
    logger.info("Retrieving reports from vector DB — query='%s', ticker='%s'", query, ticker)
    try:
        db = VectorDB()
        results = db.retrieve(query=query, ticker=ticker, top_k=top_k)
        return results
    except Exception as e:
        logger.error("Error retrieving from vector DB: %s", e)
        return [{"error": f"Retrieval failed: {str(e)}"}]


@tool("retrieve_reports_from_db")
async def retrieve_reports_from_db(query: str, ticker: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant financial report chunks from the vector database using semantic search.
    Useful for answering specific questions about a company's financials (e.g., 'What was Reliance's net income in Q2?').
    Args:
        query: Search query (e.g., 'net income 2023')
        ticker: Optional. The specific stock ticker to filter by (e.g., 'RELIANCE.NS'). Highly recommended.
        top_k: Number of relevant chunks to return (default: 5)
    """
    return await asyncio.to_thread(_retrieve_reports_sync, query, ticker, top_k)
