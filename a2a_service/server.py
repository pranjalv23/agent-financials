import logging

from agent_sdk.a2a.factory import create_a2a_app as _create

from .agent_card import FINANCIAL_AGENT_CARD
from .executor import FinancialAgentExecutor

logger = logging.getLogger("agent_financials.a2a_server")


def create_a2a_app():
    """Build the A2A Starlette application for the financial agent."""
    app = _create(FINANCIAL_AGENT_CARD, FinancialAgentExecutor, "agent_financials")
    logger.info("A2A application created for Financial Agent")
    return app
