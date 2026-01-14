import copy
import os
from typing import Dict, List, Literal, Optional

from loguru import logger

from eve.agent.llm.constants import ModelTier, get_default_model_defaults
from eve.agent.session.models import LLMConfig, LLMThinkingSettings

DEFAULT_SESSION_SELECTION_LIMIT = 25

MODEL_TIERS: Dict[str, List[str]] = {
    "high": [
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "gpt-5-nano",
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
    """Build LLMConfig from agent's llm_settings with optional overrides."""
    llm_settings = agent.llm_settings

    base_config = get_default_session_llm_config(
        "premium" if tier != "free" else "free"
    )

    if not llm_settings:
        return base_config

    model_profile = llm_settings.model_profile or "medium"
    thinking_policy = llm_settings.thinking_policy or "auto"
    thinking_effort_cap = llm_settings.thinking_effort_cap or "medium"
    thinking_effort_instructions = llm_settings.thinking_effort_instructions or None

    if thinking_override is not None:
        thinking_policy = "always" if thinking_override else "off"

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

    thinking_settings = None
    if effective_profile == "high" and thinking_policy != "off":
        thinking_settings = LLMThinkingSettings(
            policy=thinking_policy,
            effort_cap=thinking_effort_cap,
            effort_instructions=thinking_effort_instructions
            or "Use low for simple tasks, high for complex reasoning-intensive tasks.",
        )

    reasoning_effort: Optional[str] = None
    if thinking_settings and thinking_policy != "off":
        if thinking_policy == "always":
            reasoning_effort = thinking_effort_cap
        elif thinking_policy == "auto" and context_messages:
            effort_level = await route_thinking_effort(
                context_messages,
                thinking_effort_instructions,
            )
            if thinking_effort_cap == "low":
                effort_level = "low"
            elif thinking_effort_cap == "medium" and effort_level == "high":
                effort_level = "medium"
            if effort_level in ["medium", "high"]:
                reasoning_effort = effort_level

    return LLMConfig(
        model=model,
        fallback_models=fallback_models,
        thinking=thinking_settings,
        reasoning_effort=reasoning_effort,
    )


async def route_thinking_effort(
    context_messages: List, instructions: Optional[str]
) -> str:
    """Route thinking effort based on context using a small LLM"""

    fake_mode = os.getenv("FF_SESSION_FAKE_LLM", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if fake_mode:
        return "medium"

    for msg in reversed(context_messages or []):
        content = getattr(msg, "content", None)
        if content and ("===test" in content or "===tool" in content):
            return "medium"

    routing_messages = (
        context_messages[-5:]
        if context_messages and len(context_messages) > 5
        else context_messages or []
    )
    routing_context = []
    for msg in routing_messages:
        role = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "") or ""
        routing_context.append(f"{role}: {content[:200]}")

    routing_prompt = f"""Analyze this conversation context and determine how much thinking effort is needed.

Instructions: {instructions or "Use low for simple tasks, high for complex reasoning-intensive tasks."}

Conversation context:
{chr(10).join(routing_context)}

Respond with exactly one word: low, medium, or high.
Response:"""

    try:
        from litellm import acompletion

        router_response = await acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": routing_prompt}],
            max_tokens=10,
            temperature=0,
            timeout=10,
        )
        effort = router_response.choices[0].message.content.strip().lower()
        if effort in {"low", "medium", "high"}:
            return effort
    except Exception as exc:
        logger.error(f"Error routing thinking effort: {exc}")
    return "medium"
