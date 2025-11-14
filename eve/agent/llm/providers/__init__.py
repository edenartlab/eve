from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import AsyncGenerator, Any, Optional
import json

from eve.agent.session.models import LLMContext, LLMResponse

try:
    from eve.agent.session_new.instrumentation import PromptSessionInstrumentation
except Exception:  # pragma: no cover - optional import for legacy contexts
    PromptSessionInstrumentation = None  # type: ignore


class LLMProvider(ABC):
    """Abstract provider contract implemented by backend-specific SDK clients."""

    def __init__(
        self,
        *,
        instrumentation: Optional["PromptSessionInstrumentation"] = None,
    ):
        self.instrumentation = instrumentation

    @abstractmethod
    async def prompt(self, context: LLMContext) -> LLMResponse:
        """Execute a non-streaming request."""

    async def prompt_stream(self, context: LLMContext) -> AsyncGenerator[Any, None]:
        """Default streaming implementation yields the full response once."""
        response = await self.prompt(context)
        yield self._response_to_chunk(response)

    @staticmethod
    def _response_to_chunk(response: LLMResponse) -> Any:
        """Convert an LLMResponse into an OpenAI-style stream chunk."""
        tool_calls = None
        if response.tool_calls:
            tool_calls = []
            for idx, tool_call in enumerate(response.tool_calls):
                tool_calls.append(
                    SimpleNamespace(
                        index=idx,
                        id=tool_call.id,
                        function=SimpleNamespace(
                            name=tool_call.tool,
                            arguments=json.dumps(tool_call.args or {}),
                        ),
                    )
                )

        delta = SimpleNamespace(
            content=response.content,
            tool_calls=tool_calls,
        )

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=response.stop)],
            usage=SimpleNamespace(total_tokens=response.tokens_spent or 0),
        )
        return chunk
