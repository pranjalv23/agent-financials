"""Investor Profile — domain logic. MongoDB I/O is in mongo.py."""
from __future__ import annotations

from typing import Literal

KnowledgeLevel = Literal["beginner", "intermediate", "expert"]

VALID_RISK_TOLERANCES = {"conservative", "moderate", "aggressive"}
VALID_GOALS = {
    "wealth_creation", "retirement", "tax_saving",
    "emergency_fund", "education", "home_purchase", "other",
}
VALID_KNOWLEDGE_LEVELS = {"beginner", "intermediate", "expert"}

ONBOARDING_QUESTIONS: list[dict] = [
    {
        "key": "age",
        "question": "How old are you? (A rough number is fine — it helps me tailor advice to your life stage.)",
        "hint": "e.g. 22, 35, 45",
        "type": "integer",
    },
    {
        "key": "monthly_investable_inr",
        "question": "How much can you comfortably invest each month in INR? Even ₹500/month is a great start!",
        "hint": "e.g. 2000, 10000, 50000",
        "type": "integer",
    },
    {
        "key": "time_horizon_years",
        "question": "How many years are you willing to stay invested before needing this money?",
        "hint": "e.g. 3, 7, 15",
        "type": "integer",
    },
    {
        "key": "goals",
        "question": "What's your primary financial goal right now?",
        "options": sorted(VALID_GOALS),
        "type": "choice",
    },
    {
        "key": "risk_tolerance",
        "question": "If your investment dropped 20% in a month, what would you do?",
        "options": {
            "conservative": "Sell immediately — I can't afford losses",
            "moderate": "Hold and wait for recovery",
            "aggressive": "Buy more — great buying opportunity!",
        },
        "type": "choice",
    },
    {
        "key": "knowledge_level",
        "question": "How familiar are you with investing?",
        "options": {
            "beginner": "Just starting out — heard of SIPs but haven't invested yet",
            "intermediate": "I invest in mutual funds or stocks, understand basic concepts",
            "expert": "I actively track markets, understand F&O, read financial statements",
        },
        "type": "choice",
    },
]


def derive_output_mode(profile: dict) -> KnowledgeLevel:
    """Infer default response verbosity from knowledge_level."""
    kl = profile.get("knowledge_level", "beginner")
    return kl if kl in VALID_KNOWLEDGE_LEVELS else "beginner"  # type: ignore[return-value]


def profile_context_summary(profile: dict) -> str:
    """One-line profile summary injected into the agent [CONTEXT] block."""
    return (
        f"INVESTOR PROFILE — Age: {profile.get('age', 'unknown')} | "
        f"Monthly budget: ₹{profile.get('monthly_investable_inr', 'unknown')} | "
        f"Time horizon: {profile.get('time_horizon_years', 'unknown')} years | "
        f"Goal: {profile.get('goals', 'unknown')} | "
        f"Risk tolerance: {profile.get('risk_tolerance', 'unknown')} | "
        f"Knowledge level: {profile.get('knowledge_level', 'beginner')}"
    )
