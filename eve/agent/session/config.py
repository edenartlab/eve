import os
from eve.agent.session.models import LLMConfig

DEFAULT_SESSION_LLM_CONFIG_STAGE = LLMConfig(
    model="gpt-4o-mini",
)

DEFAULT_SESSION_LLM_CONFIG_PROD = LLMConfig(
    model="claude-sonnet-4-20250514",
)


def get_default_session_llm_config():
    if os.getenv("DB") == "prod":
        return DEFAULT_SESSION_LLM_CONFIG_PROD
    else:
        return DEFAULT_SESSION_LLM_CONFIG_STAGE
