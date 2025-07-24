import os
from typing import Literal
from eve.agent.session.models import LLMConfig

DEFAULT_SESSION_LLM_CONFIG_DEV = {
    "premium": LLMConfig(
        model="gemini/gemini-2.5-flash-preview-05-20",
    ),
    "free": LLMConfig(
        model="gemini/gemini-2.5-flash-preview-05-20",
    ),
}
DEFAULT_SESSION_LLM_CONFIG_STAGE = {
    "premium": LLMConfig(
        model="claude-sonnet-4-20250514",
    ),
    "free": LLMConfig(
        model="gemini/gemini-2.5-flash-preview-05-20",
    ),
}

DEFAULT_SESSION_LLM_CONFIG_PROD = {
    "premium": LLMConfig(
        model="claude-sonnet-4-20250514",
    ),
    "free": LLMConfig(
        model="gemini/gemini-2.5-flash-preview-05-20",
    ),
}


def get_default_session_llm_config(tier: Literal["premium", "free"] = "free"):
    if os.getenv("LANGFUSE_TRACING_ENVIRONMENT") == "jmill-dev":
        return DEFAULT_SESSION_LLM_CONFIG_DEV[tier]
    if os.getenv("DB") == "prod":
        return DEFAULT_SESSION_LLM_CONFIG_PROD[tier]
    else:
        return DEFAULT_SESSION_LLM_CONFIG_STAGE[tier]


DEFAULT_SESSION_SELECTION_LIMIT = 25
