"""Generate downloadable investment thesis PDF/markdown reports."""
import logging
import os
import uuid
from datetime import datetime, timezone

from langchain_core.tools import tool
from agent_sdk.utils.pdf import MarkdownPDFRenderer, slugify

logger = logging.getLogger("agent_financials.tools.investment_report")

_BASE_URL = (os.getenv("BACKEND_URL") or os.getenv("PUBLIC_URL") or "").rstrip("/")
_pdf_renderer = MarkdownPDFRenderer()


@tool
async def generate_investment_report(title: str, content: str, ticker: str = "", format: str = "pdf") -> str:
    """Generate a downloadable investment analysis report (PDF or markdown).

    Args:
        title: Report title, e.g. "RELIANCE.NS Investment Thesis — Q4 2026".
        content: Full markdown content of the report. Use ## for sections,
                 > for mentor takeaways, #### for metric groups.
        ticker: The main ticker symbol (optional, used in filename).
        format: "pdf" or "markdown". Defaults to "pdf".
    """
    from database.mongo import MongoDB

    file_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = slugify(ticker or title)

    if format == "pdf":
        filename = f"{timestamp}_{slug}_report.pdf"
    else:
        filename = f"{timestamp}_{slug}_report.md"

    try:
        if format == "pdf":
            file_bytes = _pdf_renderer.render(content, title)
        else:
            file_bytes = f"# {title}\n\n{content}".encode("utf-8")

        await MongoDB.store_file(
            file_id=file_id,
            filename=filename,
            data=file_bytes,
            file_type="investment_report",
        )

        logger.info("Generated investment report: file_id='%s', format='%s', size=%d bytes",
                    file_id, format, len(file_bytes))

        return (
            f"Investment report generated!\n\n"
            f"**Title:** {title}\n"
            f"**Format:** {format.upper()}\n"
            f"**Download:** [Download Report: {title}]({_BASE_URL}/download/{file_id})"
        )

    except Exception as e:
        logger.error("Failed to generate investment report: %s", e)
        return (
            f"Error generating report ({format}): {e}. "
            "If format was 'pdf', retry with format='markdown'."
        )
