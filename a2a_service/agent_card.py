import os

from a2a.types import AgentCard, AgentSkill, AgentCapabilities


FINANCIAL_AGENT_CARD = AgentCard(
    name="Financial Analysis Agent",
    description=(
        "Expert financial analyst and investing mentor. Analyzes stocks, "
        "fetches real-time market data, interprets financial statements, "
        "and provides actionable investment insights for Indian BSE/NSE and global markets."
    ),
    url=os.getenv("AGENT_PUBLIC_URL", "http://localhost:9001"),
    version="1.0.0",
    metadata={"mode": "financial_analyst"},
    skills=[
        AgentSkill(
            id="stock-analysis",
            name="Stock Analysis",
            description="Analyze stock fundamentals, price data, P/E ratios, and company profiles.",
            tags=["finance", "stocks", "investing", "market"],
        ),
        AgentSkill(
            id="financial-reports",
            name="Financial Reports",
            description="Fetch and analyze quarterly/yearly income statements, balance sheets, and cash flow.",
            tags=["finance", "BSE", "NSE", "reports", "balance-sheet"],
        ),
        AgentSkill(
            id="financial-research",
            name="Financial Research",
            description="Web research on market trends, news, earnings, and macro-economic factors.",
            tags=["research", "news", "market-trends", "investing"],
        ),
    ],
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
)
