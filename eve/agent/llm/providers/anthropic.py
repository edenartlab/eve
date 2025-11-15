from __future__ import annotations

import os
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from eve.agent.llm.formatting import (
    construct_anthropic_tools,
)
from eve.agent.llm.providers import LLMProvider
from eve.agent.llm.util import (
    calculate_cost_usd,
    serialize_context_messages,
    build_langfuse_prompt,
)
from eve.agent.session.models import (
    LLMContext,
    LLMResponse,
    ToolCall,
    ChatMessage,
    LLMUsage,
)


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
        base_input_payload = self._build_input_payload(
            context=context,
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
            include_thoughts=include_thoughts,
        )

        last_error: Optional[Exception] = None
        for attempt_index, model_name in enumerate(self.models):
            stage = (
                self.instrumentation.track_stage(
                    "llm.anthropic", metadata={"model": model_name}
                )
                if self.instrumentation
                else nullcontext()
            )
            try:
                with stage:
                    langfuse_input = dict(base_input_payload)
                    langfuse_input["model"] = model_name
                    langfuse_input["attempt"] = attempt_index + 1
                    request_kwargs = {
                        "model": model_name,
                        "system": system_prompt,
                        "messages": conversation,
                        "max_tokens": context.config.max_tokens or 1024,
                    }
                    if tools:
                        request_kwargs["tools"] = tools

                    start_time = datetime.now(timezone.utc)
                    response = await self.client.messages.create(**request_kwargs)
                    end_time = datetime.now(timezone.utc)
                    llm_response = self._to_llm_response(response)
                    self._record_usage(
                        model_name,
                        response,
                        llm_response,
                        input_payload=langfuse_input,
                        output_payload=self._serialize_llm_response(llm_response),
                        metadata={
                            "provider": self.provider_name,
                            "attempt": attempt_index + 1,
                        },
                        start_time=start_time,
                        end_time=end_time,
                    )
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

        system_prompt = (
            "\n\n".join(part for part in system_prompt_parts if part) or None
        )
        return system_prompt, conversation

    def _build_input_payload(
        self,
        *,
        context: LLMContext,
        system_prompt: Optional[str],
        conversation: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        include_thoughts: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "system": system_prompt,
            "messages": conversation,
            "tools": tools or [],
            "max_tokens": context.config.max_tokens or 1024,
            "reasoning_effort": context.config.reasoning_effort,
            "include_thoughts": include_thoughts,
            "fallback_models": list(context.config.fallback_models or []),
        }
        payload["context_messages"] = serialize_context_messages(context)
        return {k: v for k, v in payload.items() if v not in (None, [], {})}

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
        prompt_tokens = usage.input_tokens if usage else None
        completion_tokens = usage.output_tokens if usage else None
        cache_creation_tokens = usage.cache_creation_input_tokens if usage else None
        cache_read_tokens = usage.cache_read_input_tokens if usage else None
        total_tokens = (
            (prompt_tokens or 0) + (completion_tokens or 0) if usage else None
        )
        usage_payload = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cache_read_tokens,
            cached_completion_tokens=cache_creation_tokens,
            total_tokens=total_tokens,
        )

        finish_reason = getattr(response, "stop_reason", None)
        if finish_reason == "end_turn":
            finish_reason = "stop"

        return LLMResponse(
            content="\n".join(text_parts).strip(),
            tool_calls=tool_calls or None,
            stop=finish_reason,
            tokens_spent=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage=usage_payload,
            thought=thoughts or None,
        )

    def _record_usage(
        self,
        model: str,
        response,
        llm_response: LLMResponse,
        *,
        input_payload: Optional[Dict[str, Any]] = None,
        output_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        prompt: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.instrumentation:
            return

        usage = response.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        cache_creation_tokens = usage.cache_creation_input_tokens if usage else 0
        cache_read_tokens = usage.cache_read_input_tokens if usage else 0
        prompt_cost, completion_cost, total_cost = calculate_cost_usd(
            model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        if llm_response.usage:
            llm_response.usage.cost_usd = total_cost

        self.instrumentation.record_counter("llm.prompt_tokens", prompt_tokens or 0)
        self.instrumentation.record_counter(
            "llm.completion_tokens", completion_tokens or 0
        )
        self.instrumentation.record_counter("llm.cost_usd", total_cost or 0)

        usage_payload, usage_details, cost_details = (
            self.instrumentation.build_langfuse_usage_payload(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cache_read_tokens,
                cached_completion_tokens=cache_creation_tokens,
                cache_creation_input_tokens=cache_creation_tokens,
                cache_read_input_tokens=cache_read_tokens,
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=total_cost,
            )
        )

        self.instrumentation.create_langfuse_generation(
            name=f"{self.provider_name}.generation",
            model=model,
            input_payload=input_payload,
            output_payload=output_payload or self._serialize_llm_response(llm_response),
            usage=usage_payload,
            usage_details=usage_details,
            cost_details=cost_details,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
            prompt=prompt,
        )
