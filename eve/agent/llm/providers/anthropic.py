from __future__ import annotations

import os
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from eve.agent.llm.formatting import (
    construct_anthropic_tools,
)
from eve.agent.llm.providers import LLMProvider
from eve.agent.llm.util import (
    calculate_cost_usd,
    serialize_context_messages,
)
from eve.agent.session.models import (
    ChatMessage,
    LLMContext,
    LLMResponse,
    LLMUsage,
    ToolCall,
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

    # Models that support structured outputs (as of Nov 2025)
    STRUCTURED_OUTPUT_MODELS = {
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1",
        "claude-opus-4-1-20250115",
    }

    # Fallback model for structured outputs when using unsupported models
    STRUCTURED_OUTPUT_FALLBACK = "claude-sonnet-4-5-20250929"

    async def prompt(self, context: LLMContext) -> LLMResponse:
        include_thoughts = bool(context.config.reasoning_effort)
        system_prompt, conversation = self._prepare_messages(
            context.messages, include_thoughts=include_thoughts
        )
        tools = construct_anthropic_tools(context)

        # Build structured output payload if response_format is specified
        output_format_payload = self._build_output_format_payload(
            context.config.response_format
        )

        base_input_payload = self._build_input_payload(
            context=context,
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
            include_thoughts=include_thoughts,
        )

        last_error: Optional[Exception] = None
        for attempt_index, model_name in enumerate(self.models):
            # Auto-switch to supported model if structured outputs requested
            effective_model = model_name
            if output_format_payload and not self._supports_structured_output(
                model_name
            ):
                effective_model = self.STRUCTURED_OUTPUT_FALLBACK

            stage = (
                self.instrumentation.track_stage(
                    "llm.anthropic", metadata={"model": effective_model}
                )
                if self.instrumentation
                else nullcontext()
            )
            try:
                with stage:
                    langfuse_input = dict(base_input_payload)
                    langfuse_input["model"] = effective_model
                    langfuse_input["attempt"] = attempt_index + 1
                    request_kwargs = {
                        "model": effective_model,
                        "system": system_prompt,
                        "messages": conversation,
                        "max_tokens": context.config.max_tokens or 1024,
                    }
                    if tools:
                        request_kwargs["tools"] = tools

                    start_time = datetime.now(timezone.utc)

                    # Use structured outputs if response_format specified
                    if output_format_payload:
                        response = await self._create_with_structured_output(
                            request_kwargs, output_format_payload
                        )
                    else:
                        response = await self.client.messages.create(**request_kwargs)

                    end_time = datetime.now(timezone.utc)
                    llm_response = self._to_llm_response(response)
                    self._record_usage(
                        effective_model,
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
                        f"Anthropic model {effective_model} failed",
                        level="warning",
                        payload={"error": str(exc)},
                    )

        if last_error:
            raise last_error
        raise RuntimeError("Failed to obtain Anthropic response")

    def _supports_structured_output(self, model_name: str) -> bool:
        """Check if a model supports structured outputs."""
        normalized = model_name.lower().strip()
        # Check exact match or prefix match for versioned models
        for supported in self.STRUCTURED_OUTPUT_MODELS:
            if normalized == supported or normalized.startswith(
                supported.split("-202")[0]
            ):
                # e.g., "claude-sonnet-4-5" matches "claude-sonnet-4-5-20250929"
                if "sonnet-4-5" in normalized or "opus-4-1" in normalized:
                    return True
        return False

    async def _create_with_structured_output(
        self, request_kwargs: Dict[str, Any], output_format: Dict[str, Any]
    ):
        """Create a message with structured output using the beta API."""
        return await self.client.beta.messages.create(
            **request_kwargs,
            betas=["structured-outputs-2025-11-13"],
            output_format=output_format,
        )

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

    def _build_output_format_payload(self, response_format) -> Optional[Dict[str, Any]]:
        """Build output_format payload for Anthropic structured outputs.

        Only returns a payload when response_format is specified.
        Supports Pydantic model classes, instances, or raw dicts.

        Anthropic format (different from OpenAI):
        {
            "type": "json_schema",
            "schema": { ... }  # JSON schema directly
        }
        """
        if not response_format:
            return None

        schema = None

        # Handle dict passed directly (assume it needs transformation)
        if isinstance(response_format, dict):
            # If it's already in output_format structure, extract and transform schema
            if "schema" in response_format:
                schema = response_format["schema"]
            else:
                schema = response_format

        # Handle Pydantic model class
        elif isinstance(response_format, type) and issubclass(
            response_format, BaseModel
        ):
            schema = response_format.model_json_schema()

        # Handle Pydantic model instance
        elif isinstance(response_format, BaseModel):
            schema = response_format.model_json_schema()

        if schema is None:
            return None

        # Transform schema for Anthropic compatibility
        transformed_schema = self._transform_schema_for_anthropic(schema)
        return {"type": "json_schema", "schema": transformed_schema}

    def _transform_schema_for_anthropic(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a JSON schema to meet Anthropic's structured output requirements.

        Anthropic requires:
        - additionalProperties: false on all object types
        - All properties should be in 'required' array
        """
        if not isinstance(schema, dict):
            return schema

        result = schema.copy()

        # Handle $defs / definitions (Pydantic uses $defs)
        for defs_key in ("$defs", "definitions"):
            if defs_key in result:
                result[defs_key] = {
                    k: self._transform_schema_for_anthropic(v)
                    for k, v in result[defs_key].items()
                }

        # If this is an object type, add additionalProperties: false
        if result.get("type") == "object":
            result["additionalProperties"] = False

            # Transform nested properties
            if "properties" in result:
                result["properties"] = {
                    k: self._transform_schema_for_anthropic(v)
                    for k, v in result["properties"].items()
                }

        # Handle array items
        if result.get("type") == "array" and "items" in result:
            result["items"] = self._transform_schema_for_anthropic(result["items"])

        # Handle anyOf, allOf, oneOf
        for key in ("anyOf", "allOf", "oneOf"):
            if key in result:
                result[key] = [
                    self._transform_schema_for_anthropic(item) for item in result[key]
                ]

        return result

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
