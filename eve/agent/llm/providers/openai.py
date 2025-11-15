from __future__ import annotations

import os
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel

from eve.agent.llm.formatting import (
    construct_observability_metadata,
    construct_tools,
    prepare_messages,
)
from eve.agent.llm.util import (
    calculate_cost_usd,
    serialize_context_messages,
)
from eve.agent.llm.providers import LLMProvider
from eve.agent.session.models import LLMContext, LLMResponse, ToolCall, LLMUsage


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(
        self,
        model: str,
        fallbacks: Optional[List[str]] = None,
        instrumentation=None,
    ):
        super().__init__(instrumentation=instrumentation)
        self.models = [model] + [m for m in (fallbacks or []) if m]
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        organization = os.getenv("OPENAI_ORG_ID")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIProvider")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    async def prompt(self, context: LLMContext) -> LLMResponse:
        messages = prepare_messages(
            context.messages,
            context.config.model,
            include_thoughts=bool(context.config.reasoning_effort),
        )
        tools = construct_tools(context)
        tool_choice = context.tool_choice if tools else None
        observability = (
            construct_observability_metadata(context) if context.enable_tracing else {}
        )
        response_format_payload = self._build_response_format_payload(
            context.config.response_format
        )

        base_input_payload = self._build_input_payload(
            context=context,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            metadata=observability,
            response_format=response_format_payload,
        )

        last_error: Optional[Exception] = None
        for attempt_index, model_name in enumerate(self.models):
            canonical_name = self._normalize_model_name(model_name)
            stage = (
                self.instrumentation.track_stage(
                    "llm.openai", metadata={"model": canonical_name}
                )
                if self.instrumentation
                else nullcontext()
            )
            try:
                with stage:
                    langfuse_input = dict(base_input_payload)
                    langfuse_input["model"] = canonical_name
                    langfuse_input["attempt"] = attempt_index + 1
                    start_time = datetime.now(timezone.utc)
                    request_kwargs = dict(
                        model=canonical_name,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        response_format=response_format_payload,
                        metadata=None,
                    )
                    if context.config.max_tokens is not None:
                        request_kwargs["max_tokens"] = context.config.max_tokens
                    response = await self.client.chat.completions.create(**request_kwargs)
                    end_time = datetime.now(timezone.utc)
                    llm_response = self._to_llm_response(response)
                    self._record_usage(
                        canonical_name,
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
                        f"OpenAI model {canonical_name} failed",
                        level="warning",
                        payload={"error": str(exc)},
                    )

        if last_error:
            raise last_error
        raise RuntimeError("Failed to obtain OpenAI response")

    async def prompt_stream(self, context: LLMContext):
        # For now, rely on base implementation (single-chunk streaming)
        response = await self.prompt(context)
        yield self._response_to_chunk(response)

    def _to_llm_response(self, completion: ChatCompletion) -> LLMResponse:
        message: ChatCompletionMessage = completion.choices[0].message

        content = self._extract_message_content(message)
        tool_calls = self._extract_tool_calls(message)
        thought = getattr(message, "reasoning_content", None)

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None
        usage_payload = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", None
            )
            if usage
            else None,
            cached_completion_tokens=None,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls or None,
            stop=completion.choices[0].finish_reason,
            tokens_spent=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage=usage_payload,
            thought=thought,
        )

    def _extract_message_content(self, message: ChatCompletionMessage) -> str:
        if message.content is None:
            return ""
        if isinstance(message.content, str):
            return message.content
        # When content is a list of dicts
        parts = []
        for part in message.content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    def _extract_tool_calls(
        self, message: ChatCompletionMessage
    ) -> Optional[List[ToolCall]]:
        if not getattr(message, "tool_calls", None):
            return None
        tool_calls = []
        for tool_call in message.tool_calls:
            tool_calls.append(ToolCall.from_openai(tool_call))
        return tool_calls

    def _record_usage(
        self,
        model: str,
        completion: ChatCompletion,
        response: LLMResponse,
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

        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        prompt_cost, completion_cost, total_cost = calculate_cost_usd(
            model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        response.usage.cost_usd = total_cost

        self.instrumentation.record_counter("llm.prompt_tokens", prompt_tokens or 0)
        self.instrumentation.record_counter(
            "llm.completion_tokens", completion_tokens or 0
        )
        self.instrumentation.record_counter("llm.cost_usd", total_cost or 0)

        cache_prompt = (
            getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
            if usage
            else None
        )
        cache_completion = (
            getattr(
                getattr(usage, "completion_tokens_details", None), "cached_tokens", None
            )
            if usage
            else None
        )
        usage_payload, usage_details, cost_details = (
            self.instrumentation.build_langfuse_usage_payload(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cache_prompt,
                cached_completion_tokens=cache_completion,
                cache_read_input_tokens=cache_prompt,
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=total_cost,
            )
        )

        self.instrumentation.create_langfuse_generation(
            name=f"{self.provider_name}.generation",
            model=model,
            input_payload=input_payload,
            output_payload=output_payload or self._serialize_llm_response(response),
            usage=usage_payload,
            usage_details=usage_details,
            cost_details=cost_details,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
            prompt=prompt,
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if "/" in model_name:
            provider, raw = model_name.split("/", 1)
            if provider == "openai":
                return raw
        return model_name

    def _build_response_format_payload(self, response_format):
        if not response_format:
            return None

        if isinstance(response_format, dict):
            return response_format

        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = response_format.model_json_schema()
            return {
                "type": "json_schema",
                "json_schema": {"name": response_format.__name__, "schema": schema},
            }

        if isinstance(response_format, BaseModel):
            schema = response_format.model_json_schema()
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__class__.__name__,
                    "schema": schema,
                },
            }

        return None

    def _build_input_payload(
        self,
        *,
        context: LLMContext,
        messages: List[dict],
        tools: Optional[List[dict]],
        tool_choice: Optional[str],
        metadata: Dict[str, Any],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": messages,
            "tools": tools or [],
            "tool_choice": tool_choice,
            "max_tokens": context.config.max_tokens,
            "reasoning_effort": context.config.reasoning_effort,
            "fallback_models": list(context.config.fallback_models or []),
        }
        if metadata:
            payload["metadata"] = metadata
        if response_format:
            payload["response_format"] = response_format
        payload["context_messages"] = serialize_context_messages(context)
        return {k: v for k, v in payload.items() if v not in (None, [], {})}
