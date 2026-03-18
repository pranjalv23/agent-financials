import logging

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from .agent_card import FINANCIAL_AGENT_CARD
from .executor import FinancialAgentExecutor

logger = logging.getLogger("agent_financials.a2a_server")


def create_a2a_app() -> A2AStarletteApplication:
    """Build the A2A Starlette application for the financial agent."""
    task_store = InMemoryTaskStore()
    executor = FinancialAgentExecutor()
    request_handler = DefaultRequestHandler(
        agent_card=FINANCIAL_AGENT_CARD,
        task_store=task_store,
        executor=executor,
    )
    a2a_app = A2AStarletteApplication(
        agent_card=FINANCIAL_AGENT_CARD,
        http_handler=request_handler,
    )
    logger.info("A2A application created for Financial Agent")
    return a2a_app
