import copy
from typing import Dict, List, Literal, Optional

from eve.agent.llm.constants import ModelTier, get_default_model_defaults
from eve.agent.session.models import LLMConfig

DEFAULT_SESSION_SELECTION_LIMIT = 25

MODEL_TIERS: Dict[str, List[str]] = {
    # chain[0] is the tier's model, chain[1:3] its fallbacks — keep a
    # cross-provider fallback in slot 2 (Anthropic outages take out slot 1 too).
    "high": [
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "gpt-5.4-nano",
        "gpt-4o-mini",
        "gemini-3-flash-preview",
    ],
}


def _clone_config(config: LLMConfig) -> LLMConfig:
    return copy.deepcopy(config)


def get_default_session_llm_config(
    tier: Literal["premium", "free"] = "free",
) -> LLMConfig:
    """Return tier defaults derived from eve.agent.llm.constants."""
    model_tier = ModelTier.PREMIUM if tier == "premium" else ModelTier.FREE
    defaults = get_default_model_defaults(model_tier)
    fallback_models = [model for model, _ in defaults.fallbacks]
    return LLMConfig(model=defaults.model, fallback_models=fallback_models)


async def build_llm_config_from_agent_settings(
    agent,
    tier: str = "premium",
    thinking_override: Optional[bool] = None,
    context_messages: Optional[List] = None,
) -> LLMConfig:
    """Build LLMConfig (model + fallbacks) from an agent's llm_settings.

    NOTE: the extended-thinking / reasoning-effort system was removed. It was
    dead cost: `thinking` was never actually sent to the Anthropic API, and the
    "auto" policy spent an extra gpt-4o-mini call per turn to compute an effort
    level that then went unused. `thinking_override` and `context_messages` are
    accepted for call-site compatibility but no longer affect the config.
    """
    llm_settings = agent.llm_settings

    base_config = get_default_session_llm_config(
        "premium" if tier != "free" else "free"
    )

    if not llm_settings:
        return base_config

    model_profile = llm_settings.model_profile or "medium"

    effective_profile = model_profile
    if tier == "free" and getattr(agent, "owner_pays", "off") == "off":
        effective_profile = "low"

    overrides = MODEL_TIERS.get(effective_profile)
    if overrides:
        model_chain = overrides.copy()
    else:
        model_chain = [base_config.model]
        if base_config.fallback_models:
            model_chain.extend(base_config.fallback_models)
        if "gpt-4o-mini" not in model_chain:
            model_chain.append("gpt-4o-mini")

    model = model_chain[0]
    fallback_models = model_chain[1:3]

    return LLMConfig(
        model=model,
        fallback_models=fallback_models,
    )
