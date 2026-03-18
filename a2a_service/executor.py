import logging
import traceback

from a2a.server.agent_execution import AgentExecutor
from a2a.server.events import EventQueue
from a2a.types import (
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)

from agents.agent import run_query

logger = logging.getLogger("agent_financials.a2a_executor")


class FinancialAgentExecutor(AgentExecutor):
    """A2A executor that bridges incoming A2A tasks to the financial agent."""

    async def execute(self, task: Task, event_queue: EventQueue) -> None:
        logger.info("A2A execute — task_id='%s'", task.id)

        # Extract text from the A2A message
        query = self._extract_text(task)
        if not query:
            raise InvalidParamsError(message="No text content found in the task message.")

        session_id = task.sessionId or task.id

        try:
            result = await run_query(query, session_id=session_id)
            response_text = result["response"]

            event_queue.enqueue_event(
                task_id=task.id,
                state=TaskState.completed,
                parts=[TextPart(text=response_text)],
            )
        except Exception as e:
            logger.error("A2A execution failed: %s\n%s", e, traceback.format_exc())
            event_queue.enqueue_event(
                task_id=task.id,
                state=TaskState.failed,
                parts=[TextPart(text=f"Agent execution failed: {e}")],
            )

    async def cancel(self, task: Task, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancel is not supported.")

    @staticmethod
    def _extract_text(task: Task) -> str:
        """Extract text content from the task's message parts."""
        if not task.history:
            return ""
        last_message = task.history[-1]
        texts = []
        for part in last_message.parts:
            if isinstance(part, TextPart):
                texts.append(part.text)
            elif hasattr(part, "root") and isinstance(part.root, TextPart):
                texts.append(part.root.text)
        return " ".join(texts)
