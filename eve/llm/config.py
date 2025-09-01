import os
import time
import litellm
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from eve.session.message import ToolCall


DEFAULT_THINKING_EFFORT_INSTRUCTIONS = "Use low for simple tasks, high for complex reasoning-intensive tasks."

class LLMThinkingPolicy(str, Enum):
    AUTO = "auto"
    OFF = "off"
    ALWAYS = "always"


class LLMThinkingEffortCap(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LLMThinkingSettings(BaseModel):
    """Thinking policy for an agent"""
    
    policy: Optional[LLMThinkingPolicy] = LLMThinkingPolicy.AUTO
    effort_cap: Optional[LLMThinkingEffortCap] = LLMThinkingEffortCap.MEDIUM
    effort_instructions: Optional[str] = Field(default=DEFAULT_THINKING_EFFORT_INSTRUCTIONS)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMConfig(BaseModel):
    """Config for agent to use LLM"""
    
    model: Optional[str] = "gpt-4o-mini"
    fallback_models: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    response_format: Optional[BaseModel] = None
    thinking: Optional[LLMThinkingSettings] = None
    reasoning_effort: Optional[str] = None  # Final resolved reasoning effort (low/medium/high)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMResponse(BaseModel):
    """Response from an LLM"""

    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    stop: Optional[str] = None
    tokens_spent: Optional[int] = None
    thought: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


SUPPORTED_MODELS = [
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-20250219",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "anthropic/claude-3-5-haiku-20241022",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-3-7-sonnet-20250219",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
]


CONTEXT_WINDOW_FALLBACK_DICT = {
    "gpt-4o-mini": "gpt-4o",
    "gpt-5-mini": "gpt-5",
    "gpt-5-nano": "gpt-5",
    "gemini-2.5-flash": "gemini-2.5-pro",
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
}


DEFAULT_SESSION_LLM_CONFIG_DEV = {
    "premium": LLMConfig(
        model="claude-sonnet-4-20250514",
        fallback_models=[
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
            "gpt-4o",
        ]
    ),
    "free": LLMConfig(
        model="claude-sonnet-4-20250514",
        fallback_models=[
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
            "gpt-4o",
        ]
    ),
}

DEFAULT_SESSION_LLM_CONFIG_STAGE = {
    "premium": LLMConfig(
        model="claude-sonnet-4-20250514",
        fallback_models=[
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
            "gpt-4o",
        ]
    ),
    "free": LLMConfig(
        model="claude-sonnet-4-20250514",
        fallback_models=[
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
            "gpt-4o",
        ]
    ),
}

DEFAULT_SESSION_LLM_CONFIG_PROD = {
    "premium": LLMConfig(
        model="claude-sonnet-4-20250514",
    ),
    "free": LLMConfig(
        model="claude-sonnet-4-20250514",
    ),
}


MODEL_TIERS = {
    "high": [
        "anthropic/claude-sonnet-4-20250514", "gemini/gemini-2.5-pro", "openai/gpt-5"
    ],
    "medium": [
        "anthropic/claude-sonnet-4-20250514", "gemini/gemini-2.5-flash", "anthropic/claude-3-5-haiku-20241022" 
    ],
    "low": [
        "anthropic/claude-3-5-haiku-20241022", "gemini/gemini-2.5-flash", "openai/gpt-5-nano"
    ]
}


def get_default_session_llm_config(
    tier: Literal["premium", "free"] = "free"
):
    if os.getenv("LANGFUSE_TRACING_ENVIRONMENT") == "jmill-dev":
        return DEFAULT_SESSION_LLM_CONFIG_DEV[tier]
    if os.getenv("DB") == "PROD":
        return DEFAULT_SESSION_LLM_CONFIG_PROD[tier]
    else:
        return DEFAULT_SESSION_LLM_CONFIG_STAGE[tier]


def get_models_for_profile(model_profile: str, tier: str = "premium") -> tuple[str, list[str]]:
    """Get primary model and fallbacks based on model_profile setting and user tier"""
    
    # Free tier users are automatically capped to low profile
    effective_profile = "low" if tier == "free" else model_profile
    models = MODEL_TIERS.get(effective_profile, MODEL_TIERS["medium"])

    return models[0], models[1:3]  # Return (primary, [fallback1, fallback2])



async def build_llm_config_from_agent_settings(
    agent_settings, 
    tier: str = "premium", 
    thinking_override: bool = None, 
    context_messages: list = None
) -> LLMConfig:
    """Build LLMConfig from agent's llm_settings with optional thinking override and context for routing"""

    if not agent_settings:
        return get_default_session_llm_config(tier)
    
    model_profile = agent_settings.model_profile or 'medium'
    thinking_policy = agent_settings.thinking_policy or 'auto'
    thinking_effort_cap = agent_settings.thinking_effort_cap or 'medium'
    thinking_effort_instructions = agent_settings.thinking_effort_instructions or None
    
    # Apply thinking override if provided
    if thinking_override is not None:
        thinking_policy = "always" if thinking_override else "off"
    
    # Get model and fallbacks based on profile and tier
    model, fallback_models = get_models_for_profile(model_profile, tier)

    # Create thinking settings
    from eve.agent.session.models import LLMThinkingSettings
    thinking_settings = None
    if model_profile == "high" and thinking_policy != "off":
        thinking_settings = LLMThinkingSettings(
            policy=thinking_policy,
            effort_cap=thinking_effort_cap,
            effort_instructions=thinking_effort_instructions or DEFAULT_THINKING_EFFORT_INSTRUCTIONS
        )
    
    # Resolve reasoning effort
    reasoning_effort = None
    if thinking_settings and thinking_policy != "off":
        if thinking_policy == "always":
            reasoning_effort = thinking_effort_cap
        elif thinking_policy == "auto" and context_messages:
            # Route thinking effort based on context
            effort_level = await route_thinking_effort(context_messages, thinking_effort_instructions)
            
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
        reasoning_effort=reasoning_effort
    )
    
    # Single log showing final LLM configuration
    override_info = f" (override: {thinking_override})" if thinking_override is not None else ""
    tier_info = f" (tier: {tier})" if tier == "free" and model_profile != "low" else f" (tier: {tier})"
    print(f"🔧 LLM Config: profile={model_profile}, model={config.model}, thinking={config.thinking.policy if config.thinking else 'off'}, reasoning_effort={config.reasoning_effort or 'none'}{tier_info}{override_info}")
    
    return config



async def route_thinking_effort(context_messages: list, instructions: str) -> str:
    """Route thinking effort based on context using a small LLM"""
    
    start_time = time.time()
    
    # Extract last 5 messages for routing context
    routing_messages = context_messages[-5:] if len(context_messages) > 5 else context_messages

    routing_context = []
    for msg in routing_messages:
        role = getattr(msg, 'role', 'unknown')
        content = msg.content if getattr(msg, 'content', '') else "" 
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
        router_response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": routing_prompt}],
            max_tokens=10,
            temperature=0,
            timeout=10,
        )
        
        effort = router_response.choices[0].message.content.strip().lower()
        total_time = time.time() - start_time
        
        if effort in ["low", "medium", "high"]:
            return effort
        else:
            return "medium"  # Default fallback
            
    except Exception as e:
        return "medium"  # Default fallback
