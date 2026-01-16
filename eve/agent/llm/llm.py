import logging
from typing import Any, AsyncGenerator, Callable, List, Optional, Tuple

from eve.agent.llm.constants import MODEL_PROVIDER_OVERRIDES, ModelProvider
from eve.agent.llm.providers import LLMProvider
from eve.agent.llm.util import should_force_fake_response, validate_input
from eve.agent.session.models import LLMContext, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[LLMContext], Optional[LLMProvider]]
_provider_factory_override: Optional[ProviderFactory] = None


def set_provider_factory(factory: Optional[ProviderFactory]) -> None:
    """Allow callers (and tests) to override provider resolution."""
    global _provider_factory_override
    _provider_factory_override = factory


class FallbackChainProvider(LLMProvider):
    """Proxy provider that sequentially tries multiple providers until one succeeds."""

    def __init__(
        self,
        providers: List[LLMProvider],
        *,
        instrumentation=None,
    ) -> None:
        super().__init__(instrumentation=instrumentation)
        self._providers = providers

    async def prompt(self, context: LLMContext) -> LLMResponse:
        last_error: Optional[Exception] = None
        for provider in self._providers:
            try:
                return await provider.prompt(context)
            except Exception as exc:
                logger.warning(
                    f"Provider {provider.__class__.__name__} failed, trying next: {exc}"
                )
                last_error = exc
        if last_error:
            raise last_error
        raise RuntimeError("No providers available for prompt request")

    async def prompt_stream(self, context: LLMContext) -> AsyncGenerator[Any, None]:
        last_error: Optional[Exception] = None
        for provider in self._providers:
            try:
                async for chunk in provider.prompt_stream(context):
                    yield chunk
                return
            except Exception as exc:
                logger.warning(
                    f"Provider {provider.__class__.__name__} failed (stream), trying next: {exc}"
                )
                last_error = exc
        if last_error:
            raise last_error
        raise RuntimeError("No providers available for streaming request")


def get_provider(context: LLMContext, instrumentation=None) -> Optional[LLMProvider]:
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

    all_models = [model_name] + list(context.config.fallback_models or [])
    provider_chain = _build_provider_chain(all_models)
    if not provider_chain:
        return None

    providers: List[LLMProvider] = []
    for provider_type, provider_models in provider_chain:
        instance = _instantiate_provider(
            provider_type,
            provider_models,
            instrumentation=instrumentation,
        )
        if instance is not None:
            providers.append(instance)

    if not providers:
        return None
    if len(providers) == 1:
        return providers[0]
    return FallbackChainProvider(providers, instrumentation=instrumentation)


def _detect_provider(model_name: str) -> ModelProvider:
    normalized = model_name.lower()
    if "/" in normalized:
        _, base = normalized.split("/", 1)
    else:
        base = normalized

    if base in MODEL_PROVIDER_OVERRIDES:
        return MODEL_PROVIDER_OVERRIDES[base]

    if base.startswith("openai"):
        return ModelProvider.OPENAI
    if base.startswith("anthropic") or base.startswith("claude"):
        return ModelProvider.ANTHROPIC
    if base.startswith("gemini") or base.startswith("vertex"):
        return ModelProvider.GEMINI
    if base.startswith("gpt"):
        return ModelProvider.OPENAI

    return ModelProvider.OPENAI


def _build_provider_chain(
    models: List[str],
) -> List[Tuple[ModelProvider, List[str]]]:
    """Group consecutive models by provider while preserving order."""
    chain: List[Tuple[ModelProvider, List[str]]] = []
    for raw_name in models:
        name = (raw_name or "").strip()
        if not name:
            continue
        provider = _detect_provider(name)
        if chain and chain[-1][0] == provider:
            chain[-1][1].append(name)
            continue
        chain.append((provider, [name]))
    return chain


def _instantiate_provider(
    provider_type: ModelProvider,
    models: List[str],
    *,
    instrumentation=None,
) -> Optional[LLMProvider]:
    if not models:
        return None
    model, *fallbacks = models

    if provider_type == ModelProvider.ANTHROPIC:
        from eve.agent.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=model,
            fallbacks=fallbacks,
            instrumentation=instrumentation,
        )

    if provider_type == ModelProvider.OPENAI:
        from eve.agent.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=model,
            fallbacks=fallbacks,
            instrumentation=instrumentation,
        )

    if provider_type == ModelProvider.GEMINI:
        from eve.agent.llm.providers.google import GoogleProvider

        return GoogleProvider(
            model=model,
            fallbacks=fallbacks,
            instrumentation=instrumentation,
        )

    if provider_type == ModelProvider.FAKE:
        from eve.agent.llm.providers.fake import FakeProvider

        return FakeProvider(instrumentation=instrumentation)

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
    tool = llm_context.tools.get(tool_call.tool)
    if not tool and tool_call.tool.startswith("tool_"):
        stripped_tool = tool_call.tool[len("tool_") :]
        tool = llm_context.tools.get(stripped_tool)
        if tool:
            logger.warning(
                f"Mapped tool call name from {tool_call.tool} to {stripped_tool}"
            )
            tool_call.tool = stripped_tool
    if not tool:
        raise KeyError(tool_call.tool)
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
    provider: Optional[LLMProvider] = None,
) -> LLMResponse:
    """Execute a prompt against the resolved provider.

    When a provider isn't supplied, resolve one automatically using the context's
    model metadata. This keeps backwards compatibility with legacy call sites
    while still allowing callers to inject a specific provider.
    """
    validate_input(context)
    resolved_provider = provider or get_provider(context)
    if resolved_provider is None:
        raise RuntimeError(
            f"No LLM provider available for model {context.config.model}"
        )
    return await resolved_provider.prompt(context)


async def async_prompt_stream(
    context: LLMContext,
    provider: Optional[LLMProvider] = None,
) -> AsyncGenerator[Any, None]:
    validate_input(context)

    resolved_provider = provider or get_provider(context)
    if resolved_provider is None:
        raise RuntimeError(
            f"No LLM provider available for model {context.config.model}"
        )

    async for chunk in resolved_provider.prompt_stream(context):
        yield chunk
