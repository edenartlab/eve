import os
from typing import Literal
from eve.agent.session.models import LLMConfig

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


def get_default_session_llm_config(tier: Literal["premium", "free"] = "free"):
    if os.getenv("LANGFUSE_TRACING_ENVIRONMENT") == "jmill-dev":
        return DEFAULT_SESSION_LLM_CONFIG_DEV[tier]
    if os.getenv("DB") == "PROD":
        return DEFAULT_SESSION_LLM_CONFIG_PROD[tier]
    else:
        return DEFAULT_SESSION_LLM_CONFIG_STAGE[tier]



DEFAULT_SESSION_SELECTION_LIMIT = 25


# Master model configuration: tier -> [primary, fallback1, fallback2]
# MODEL_TIERS = {
#     "high": ["openai/gpt-5", "anthropic/claude-sonnet-4", "vertex_ai/gemini-2.5-pro"],
#     "medium": ["openai/gpt-5-mini", "vertex_ai/gemini-2.5-flash", "anthropic/claude-3-5-haiku"],
#     "low": ["vertex_ai/gemini-2.5-flash", "openai/gpt-5-nano", "anthropic/claude-3-5-haiku"]
# }

MODEL_TIERS = {
    "high": ["anthropic/claude-sonnet-4-20250514", "gemini/gemini-2.5-pro", "openai/gpt-5"],
    "medium": ["anthropic/claude-sonnet-4-20250514", "gemini/gemini-2.5-flash", "anthropic/claude-3-5-haiku-20241022" ],
    "low": ["anthropic/claude-3-5-haiku-20241022", "gemini/gemini-2.5-flash", "openai/gpt-5-nano"]
}


async def build_llm_config_from_agent_settings(
    agent, 
    tier: str = "premium", 
    thinking_override: bool = None, 
    context_messages: list = None
) -> LLMConfig:
    """Build LLMConfig from agent's llm_settings with optional thinking override and context for routing"""
    llm_settings = agent.llm_settings

    if not llm_settings:
        return get_default_session_llm_config(tier)
    
    model_profile = llm_settings.model_profile or 'medium'
    thinking_policy = llm_settings.thinking_policy or 'auto'
    thinking_effort_cap = llm_settings.thinking_effort_cap or 'medium'
    thinking_effort_instructions = llm_settings.thinking_effort_instructions or None
    
    # Apply thinking override if provided
    if thinking_override is not None:
        thinking_policy = "always" if thinking_override else "off"
    
    # Get model and fallbacks based on profile and tier
    effective_profile = model_profile    
    if tier == "free" and agent.owner_pays == "off":  # tbd: distinguish between full and deployments
        effective_profile = "low"
    
    # Get models array
    models = MODEL_TIERS.get(effective_profile, MODEL_TIERS["medium"])
    model, fallback_models = models[0], models[1:3] 
    print(f"🔧 [CONFIG] get_models_for_profile - Effective profile: {effective_profile}, model_profile: {model_profile}, tier: {tier}")

    # Create thinking settings
    from eve.agent.session.models import LLMThinkingSettings
    thinking_settings = None
    if effective_profile == "high" and thinking_policy != "off":
        thinking_settings = LLMThinkingSettings(
            policy=thinking_policy,
            effort_cap=thinking_effort_cap,
            effort_instructions=thinking_effort_instructions or "Use low for simple tasks, high for complex reasoning-intensive tasks."
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
    import time
    
    start_time = time.time()
    print(f"🤖 [ROUTER] Starting thinking effort routing...")
    
    # Extract last 5 messages for routing context
    routing_messages = context_messages[-5:] if len(context_messages) > 5 else context_messages
    print(f"🤖 [ROUTER] Using {len(routing_messages)} messages for routing context")
    
    # Create a simple text context for routing
    routing_context = []
    for msg in routing_messages:
        role = getattr(msg, 'role', 'unknown')
        content = getattr(msg, 'content', '')[:200] if getattr(msg, 'content', '') else ""  # Truncate for efficiency
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

    print(f"🤖 [ROUTER] Routing prompt:\n{routing_prompt}")
    print(f"🤖 [ROUTER] Calling gpt-4o-mini for routing...")

    try:
        # Use fast model for routing
        from litellm import acompletion
        router_start = time.time()
        router_response = await acompletion(
            model="gpt-4o-mini",  # Fast, cheap routing model
            messages=[{"role": "user", "content": routing_prompt}],
            max_tokens=10,
            temperature=0,
            timeout=10,
        )
        router_end = time.time()
        
        effort = router_response.choices[0].message.content.strip().lower()
        total_time = time.time() - start_time
        router_time = router_end - router_start
        
        print(f"🤖 [ROUTER] Raw response: '{router_response.choices[0].message.content}'")
        print(f"🤖 [ROUTER] Parsed effort: '{effort}'")
        print(f"🤖 [ROUTER] ⏱️  Router LLM call: {router_time:.3f}s, Total routing: {total_time:.3f}s")
        
        if effort in ["low", "medium", "high"]:
            print(f"🤖 [ROUTER] ✅ Valid effort level returned: {effort}")
            return effort
        else:
            print(f"🤖 [ROUTER] ⚠️  Invalid effort '{effort}', defaulting to medium")
            return "medium"  # Default fallback
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"🤖 [ROUTER] ❌ Routing error after {total_time:.3f}s: {e}")
        print(f"🤖 [ROUTER] Defaulting to medium effort")
        return "medium"  # Default fallback
