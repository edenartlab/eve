import os
from typing import Literal
from eve.agent.session.models import LLMConfig
from loguru import logger

DEFAULT_SESSION_LLM_CONFIG_DEV = {
    "premium": LLMConfig(
        # model="claude-haiku-4-5",
        model="claude-sonnet-4-5",
        fallback_models=[
            "gpt-5-nano"
        ],
    ),
    "free": LLMConfig(
        model="claude-sonnet-4-5",
        fallback_models=[
            "gpt-5-nano"
        ],
    ),
}
DEFAULT_SESSION_LLM_CONFIG_STAGE = {
    "premium": LLMConfig(
        model="claude-sonnet-4-5",
        fallback_models=[
            "gpt-5-nano"
        ],
    ),
    "free": LLMConfig(
        model="claude-sonnet-4-5",
        fallback_models=[
            "gpt-5-nano"
        ],
    ),
}

DEFAULT_SESSION_LLM_CONFIG_PROD = {
    "premium": LLMConfig(
        model="claude-sonnet-4-5",
    ),
    "free": LLMConfig(
        model="claude-sonnet-4-5",
    ),
}

DEFAULT_SESSION_SELECTION_LIMIT = 100

# Master model configuration: tier -> [primary, fallback1, fallback2]
MODEL_TIERS = {
    "high": ["claude-sonnet-4-5", "claude-haiku-4-5", "openai/gpt-5-nano", "openai/gpt-4o-mini"],
    "medium": ["claude-sonnet-4-5","claude-haiku-4-5", "openai/gpt-5-nano", "openai/gpt-4o-mini"],
    "low": ["claude-sonnet-4-5", "claude-haiku-4-5", "openai/gpt-5-nano", "openai/gpt-4o-mini"],
}


def get_default_session_llm_config(tier: Literal["premium", "free"] = "free"):
    if os.getenv("LANGFUSE_TRACING_ENVIRONMENT") == "jmill-dev":
        return DEFAULT_SESSION_LLM_CONFIG_DEV[tier]
    if os.getenv("DB") == "PROD":
        return DEFAULT_SESSION_LLM_CONFIG_PROD[tier]
    else:
        return DEFAULT_SESSION_LLM_CONFIG_STAGE[tier]

async def build_llm_config_from_agent_settings(
    agent,
    tier: str = "premium",
    thinking_override: bool = None,
    context_messages: list = None,
) -> LLMConfig:
    """Build LLMConfig from agent's llm_settings with optional thinking override and context for routing"""
    llm_settings = agent.llm_settings

    if not llm_settings:
        return get_default_session_llm_config(tier)

    model_profile = llm_settings.model_profile or "medium"
    thinking_policy = llm_settings.thinking_policy or "auto"
    thinking_effort_cap = llm_settings.thinking_effort_cap or "medium"
    thinking_effort_instructions = llm_settings.thinking_effort_instructions or None

    # Apply thinking override if provided
    if thinking_override is not None:
        thinking_policy = "always" if thinking_override else "off"

    # Get model and fallbacks based on profile and tier
    effective_profile = model_profile
    if (
        tier == "free" and agent.owner_pays == "off"
    ):  # tbd: distinguish between full and deployments
        effective_profile = "low"

    # Get models array
    models = MODEL_TIERS.get(effective_profile, MODEL_TIERS["medium"])
    model, fallback_models = models[0], models[1:3]

    # Create thinking settings
    from eve.agent.session.models import LLMThinkingSettings

    thinking_settings = None
    if effective_profile == "high" and thinking_policy != "off":
        thinking_settings = LLMThinkingSettings(
            policy=thinking_policy,
            effort_cap=thinking_effort_cap,
            effort_instructions=thinking_effort_instructions
            or "Use low for simple tasks, high for complex reasoning-intensive tasks.",
        )

    # Resolve reasoning effort
    reasoning_effort = None
    if thinking_settings and thinking_policy != "off":
        if thinking_policy == "always":
            reasoning_effort = thinking_effort_cap
        elif thinking_policy == "auto" and context_messages:
            # Route thinking effort based on context
            effort_level = await route_thinking_effort(
                context_messages, thinking_effort_instructions
            )

            # Cap to maximum allowed effort
            if thinking_effort_cap == "low":
                effort_level = "low"
            elif thinking_effort_cap == "medium" and effort_level == "high":
                effort_level = "medium"

            # Only apply reasoning_effort for medium/high complexity
            if effort_level in ["medium", "high"]:
                reasoning_effort = effort_level

    config = LLMConfig(
        model=model,
        fallback_models=fallback_models,
        thinking=thinking_settings,
        reasoning_effort=reasoning_effort,
    )

    # Single log showing final LLM configuration
    override_info = (
        f" (override: {thinking_override})" if thinking_override is not None else ""
    )
    tier_info = (
        f" (tier: {tier})"
        if tier == "free" and model_profile != "low"
        else f" (tier: {tier})"
    )

    return config


async def route_thinking_effort(context_messages: list, instructions: str) -> str:
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

    # Extract last 5 messages for routing context
    routing_messages = (
        context_messages[-5:] if len(context_messages) > 5 else context_messages
    )

    # Create a simple text context for routing
    routing_context = []
    for msg in routing_messages:
        role = getattr(msg, "role", "unknown")
        content = (
            getattr(msg, "content", "")[:200] if getattr(msg, "content", "") else ""
        )  # Truncate for efficiency
        routing_context.append(f"{role}: {content}")

    routing_prompt = f"""Analyze this conversation context and determine how much thinking effort is needed.

Instructions: {instructions or "Use low for simple tasks, high for complex reasoning-intensive tasks."}

Conversation context:
{chr(10).join(routing_context)}

Based on the complexity, reasoning requirements, and context, respond with exactly one word:
- "low" for simple questions, basic information, or straightforward tasks
- "medium" for moderate complexity requiring some analysis
- "high" for complex reasoning, multi-step problems, or deep analysis

Response:"""

    try:
        # Use fast model for routing
        from litellm import acompletion

        router_response = await acompletion(
            model="gpt-4o-mini",  # Fast, cheap routing model
            messages=[{"role": "user", "content": routing_prompt}],
            max_tokens=10,
            temperature=0,
            timeout=10,
        )

        effort = router_response.choices[0].message.content.strip().lower()

        if effort in ["low", "medium", "high"]:
            return effort
        else:
            return "medium"  # Default fallback

    except Exception as e:
        logger.error(f"Error routing thinking effort: {e}")
        return "medium"  # Default fallback
