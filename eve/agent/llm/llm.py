from typing import Optional, AsyncGenerator, Callable, Any
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


def get_provider(context: LLMContext) -> Optional[LLMProvider]:
    """Resolve the appropriate provider for a context, if available."""
    if _provider_factory_override:
        provider = _provider_factory_override(context)
        if provider is not None:
            return provider

    if should_force_fake_response(context):
        from eve.agent.llm.providers.fake import FakeProvider
        return FakeProvider()

    return None


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
