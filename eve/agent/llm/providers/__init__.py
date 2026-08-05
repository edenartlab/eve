import json
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, Optional

from eve.agent.session.models import LLMContext, LLMResponse

try:
    from eve.agent.session.instrumentation import PromptSessionInstrumentation
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

        # Carry the cost through, not just the token count. Billing reads
        # `chunk.usage.cost_usd` (runtime._persist_assistant_message) to compute
        # the cost-plus top-up; this synthesized usage object omitted it, so
        # every streamed turn on a provider that routes through the default
        # prompt_stream (OpenAI, Google) reported no cost and the top-up never
        # fired — the whole run, up to max_turns of tool-loop calls, collected
        # only the 2-manna floor. `stream` is client-controlled, so it was
        # selectable. Anthropic's native stream already attaches cost.
        usage = SimpleNamespace(
            total_tokens=response.tokens_spent or 0,
            prompt_tokens=getattr(response, "prompt_tokens", None),
            completion_tokens=getattr(response, "completion_tokens", None),
            cost_usd=getattr(getattr(response, "usage", None), "cost_usd", None),
        )

        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=response.stop)],
            usage=usage,
            llm_call_id=getattr(response, "llm_call_id", None),
        )
        return chunk

    @staticmethod
    def _serialize_llm_response(response: LLMResponse) -> Dict[str, Any]:
        """Return a JSON-serializable representation of an LLM response."""
        payload: Dict[str, Any] = {
            "content": response.content,
            "stop": response.stop,
            "tokens_spent": response.tokens_spent,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "thought": response.thought,
        }
        if response.tool_calls:
            payload["tool_calls"] = [tc.model_dump() for tc in response.tool_calls]
        if response.usage:
            payload["usage"] = response.usage.model_dump(exclude_none=True)
        return {k: v for k, v in payload.items() if v not in (None, [], {})}
