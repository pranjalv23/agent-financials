import logging
import traceback

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskState,
    TextPart,
    UnsupportedOperationError,
)

from agents.agent import run_query

logger = logging.getLogger("agent_financials.a2a_executor")


class FinancialAgentExecutor(AgentExecutor):
    """A2A executor that bridges incoming A2A tasks to the financial agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("A2A execute — task_id='%s'", context.task_id)

        query = context.get_user_input()
        if not query:
            logger.error("No text content found in the request")
            await event_queue.enqueue_event(
                task_id=context.task_id,
                state=TaskState.failed,
                parts=[TextPart(text="No text content found in the request.")],
            )
            return

        session_id = context.context_id or context.task_id

        try:
            result = await run_query(query, session_id=session_id)
            response_text = result["response"]

            await event_queue.enqueue_event(
                task_id=context.task_id,
                state=TaskState.completed,
                parts=[TextPart(text=response_text)],
            )
        except Exception as e:
            logger.error("A2A execution failed: %s\n%s", e, traceback.format_exc())
            await event_queue.enqueue_event(
                task_id=context.task_id,
                state=TaskState.failed,
                parts=[TextPart(text=f"Agent execution failed: {e}")],
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancel is not supported.")
