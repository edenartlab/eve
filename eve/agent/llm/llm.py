from typing import Optional, AsyncGenerator, Callable, Any, List

from eve.agent.llm.constants import ModelProvider
from eve.agent.llm.providers import LLMProvider
from eve.agent.session.models import LLMContext, LLMResponse, ToolCall
from eve.agent.llm.util import (
    validate_input,
    should_force_fake_response,
)

ProviderFactory = Callable[[LLMContext], Optional[LLMProvider]]
_provider_factory_override: Optional[ProviderFactory] = None


def set_provider_factory(factory: Optional[ProviderFactory]) -> None:
    """Allow callers (and tests) to override provider resolution."""
    global _provider_factory_override
    _provider_factory_override = factory


def get_provider(
    context: LLMContext, instrumentation=None
) -> Optional[LLMProvider]:
    """Resolve the appropriate provider for a context, if available."""
    if _provider_factory_override:
        provider = _provider_factory_override(context)
        if provider is not None:
            return provider

    if should_force_fake_response(context):
        from eve.agent.llm.providers.fake import FakeProvider
        return FakeProvider(instrumentation=instrumentation)

    model_name = (context.config.model or "").strip()
    if not model_name:
        return None

    provider_type = _detect_provider(model_name)
    fallbacks = context.config.fallback_models or []

    if provider_type == ModelProvider.ANTHROPIC:
        from eve.agent.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=model_name,
            fallbacks=fallbacks,
            instrumentation=instrumentation,
        )

    if provider_type == ModelProvider.OPENAI:
        from eve.agent.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=model_name,
            fallbacks=fallbacks,
            instrumentation=instrumentation,
        )

    if provider_type == ModelProvider.GEMINI:
        from eve.agent.llm.providers.google import GoogleProvider

        return GoogleProvider(
            model=model_name,
            fallbacks=fallbacks,
            instrumentation=instrumentation,
        )

    return None


def _detect_provider(model_name: str) -> ModelProvider:
    normalized = model_name.lower()
    if normalized.startswith("openai/"):
        return ModelProvider.OPENAI
    if normalized.startswith("anthropic/"):
        return ModelProvider.ANTHROPIC
    if normalized.startswith("google/") or normalized.startswith("vertex/"):
        return ModelProvider.GEMINI
    if normalized.startswith("claude") or "anthropic" in normalized:
        return ModelProvider.ANTHROPIC
    if normalized.startswith("gemini"):
        return ModelProvider.GEMINI
    return ModelProvider.OPENAI


async def async_run_tool_call(
    llm_context: LLMContext,
    tool_call: ToolCall,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    public: bool = True,
    is_client_platform: bool = False,
):
    tool = llm_context.tools[tool_call.tool]
    task = await tool.async_start_task(
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        args=tool_call.args,
        mock=False,
        public=public,
        is_client_platform=is_client_platform,
    )

    result = await tool.async_wait(task)

    # Add task.cost and task.id to the result object
    if isinstance(result, dict):
        result["cost"] = getattr(task, "cost", None)
        result["task"] = getattr(task, "id", None)

    return result


async def async_prompt(
    context: LLMContext,
    provider: LLMProvider,
) -> LLMResponse:
    validate_input(context)
    return await provider.prompt(context)


async def async_prompt_stream(
    context: LLMContext,
    provider: LLMProvider,
) -> AsyncGenerator[Any, None]:
    validate_input(context)

    async for chunk in provider.prompt_stream(context):
        yield chunk
