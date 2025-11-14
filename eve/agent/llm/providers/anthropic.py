from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from eve.agent.llm.formatting import construct_anthropic_tools, construct_observability_metadata
from eve.agent.llm.pricing import calculate_cost_usd
from eve.agent.llm.providers import LLMProvider
from eve.agent.session.models import LLMContext, LLMResponse, ToolCall, ChatMessage


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(
        self,
        model: str,
        fallbacks: Optional[List[str]] = None,
        instrumentation=None,
    ):
        super().__init__(instrumentation=instrumentation)
        self.models = [model] + [m for m in (fallbacks or []) if m]
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for AnthropicProvider")
        self.client = AsyncAnthropic(api_key=api_key)

    async def prompt(self, context: LLMContext) -> LLMResponse:
        include_thoughts = bool(context.config.reasoning_effort)
        system_prompt, conversation = self._prepare_messages(
            context.messages, include_thoughts=include_thoughts
        )
        tools = construct_anthropic_tools(context)
        observability = (
            construct_observability_metadata(context) if context.enable_tracing else {}
        )

        last_error: Optional[Exception] = None
        for model_name in self.models:
            stage = (
                self.instrumentation.track_stage(
                    "llm.anthropic", metadata={"model": model_name}
                )
                if self.instrumentation
                else nullcontext()
            )
            try:
                with stage:
                    response = await self.client.messages.create(
                        model=model_name,
                        system=system_prompt,
                        messages=conversation,
                        max_tokens=context.config.max_tokens or 1024,
                        tools=tools,
                        metadata=observability or None,
                    )
                    llm_response = self._to_llm_response(response)
                    self._record_usage(model_name, response, llm_response)
                    return llm_response
            except Exception as exc:
                last_error = exc
                if self.instrumentation:
                    self.instrumentation.log_event(
                        f"Anthropic model {model_name} failed",
                        level="warning",
                        payload={"error": str(exc)},
                    )

        if last_error:
            raise last_error
        raise RuntimeError("Failed to obtain Anthropic response")

    async def prompt_stream(self, context: LLMContext):
        response = await self.prompt(context)
        yield self._response_to_chunk(response)

    def _prepare_messages(
        self, messages: List[ChatMessage], include_thoughts: bool = False
    ) -> (Optional[str], List[Dict[str, Any]]):
        system_prompt_parts: List[str] = []
        conversation: List[Dict[str, Any]] = []

        for chat_message in messages:
            schemas = chat_message.anthropic_schema(include_thoughts=include_thoughts)
            for schema in schemas:
                role = schema.get("role")
                if role == "system":
                    system_prompt_parts.append(schema.get("content") or "")
                else:
                    conversation.append(schema)

        system_prompt = "\n\n".join(part for part in system_prompt_parts if part) or None
        return system_prompt, conversation

    def _to_llm_response(self, response) -> LLMResponse:
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        thoughts: List[Dict[str, Any]] = []

        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        tool=getattr(block, "name", ""),
                        args=getattr(block, "input", {}) or {},
                        status="pending",
                    )
                )
            elif block_type in {"thinking", "redacted_thinking"}:
                payload = {"type": block_type}
                if hasattr(block, "thinking"):
                    payload["thinking"] = block.thinking
                if hasattr(block, "signature") and block.signature:
                    payload["signature"] = block.signature
                if hasattr(block, "data"):
                    payload["data"] = block.data
                thoughts.append(payload)

        usage = response.usage
        total_tokens = (
            (usage.input_tokens or 0) + (usage.output_tokens or 0) if usage else None
        )

        return LLMResponse(
            content="\n".join(text_parts).strip(),
            tool_calls=tool_calls or None,
            stop=getattr(response, "stop_reason", None),
            tokens_spent=total_tokens,
            thought=thoughts or None,
        )

    def _record_usage(self, model: str, response, llm_response: LLMResponse) -> None:
        if not self.instrumentation:
            return

        usage = response.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        _, _, total_cost = calculate_cost_usd(
            model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

        self.instrumentation.record_counter("llm.prompt_tokens", prompt_tokens or 0)
        self.instrumentation.record_counter(
            "llm.completion_tokens", completion_tokens or 0
        )
        self.instrumentation.record_counter("llm.cost_usd", total_cost or 0)

        self.instrumentation.create_langfuse_generation(
            name=f"{self.provider_name}.generation",
            model=model,
            input_payload=None,
            output_payload=llm_response.content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
