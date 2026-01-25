from __future__ import annotations

import json
import os
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
from anthropic import AsyncAnthropic
from bson import ObjectId
from loguru import logger
from pydantic import BaseModel

from eve import db
from eve.agent.llm.formatting import (
    construct_anthropic_tools,
)
from eve.agent.llm.providers import LLMProvider
from eve.agent.llm.util import (
    calculate_cost_usd,
    serialize_context_messages,
    truncate_base64_in_payload,
)
from eve.agent.session.models import (
    ChatMessage,
    LLMCall,
    LLMContext,
    LLMResponse,
    LLMUsage,
    ToolCall,
)
from eve.user import User

# Anthropic web search tool configuration
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}
TOOL_SEARCH_TOOL_BM25 = {
    "type": "tool_search_tool_bm25_20251119",
    "name": "tool_search_tool_bm25",
}


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

        # Check if we have a Pydantic class for structured outputs (for streaming)
        # or a dict schema (for non-streaming with timeout)
        response_format = context.config.response_format
        response_format_class = None
        output_format_payload = None

        if response_format is not None:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                # Pydantic class - can use streaming
                response_format_class = response_format
            else:
                # Dict or instance - need to build payload for non-streaming
                output_format_payload = self._build_output_format_payload(
                    response_format
                )

        # Extract context metadata for LLMCall
        llm_call_metadata = {}
        if context.metadata and context.metadata.trace_metadata:
            tm = context.metadata.trace_metadata
            llm_call_metadata = {
                "session": tm.session_id,
                "agent": tm.agent_id,
                "user": tm.user_id,
            }

        last_error: Optional[Exception] = None
        for attempt_index, model_name in enumerate(self.models):
            llm_call = (
                None  # Initialize to None for later reference in exception handler
            )
            # Auto-switch to supported model if structured outputs requested
            effective_model = model_name
            has_structured_output = response_format_class or output_format_payload
            if has_structured_output and not self._supports_structured_output(
                model_name
            ):
                effective_model = self.STRUCTURED_OUTPUT_FALLBACK

            tools_payload = self._build_tools_payload(tools, effective_model)
            base_input_payload = self._build_input_payload(
                context=context,
                system_prompt=system_prompt,
                conversation=conversation,
                tools=tools_payload,
                include_thoughts=include_thoughts,
            )

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
                        "max_tokens": context.config.max_tokens or 32000,
                    }

                    # Log the actual messages being sent to the API
                    session_info = llm_call_metadata.get("session", "unknown")
                    logger.info(
                        "[ANTHROPIC_PAYLOAD] ========== FULL LLM PAYLOAD =========="
                    )
                    logger.info(f"[ANTHROPIC_PAYLOAD] Session: {session_info}")
                    logger.info(f"[ANTHROPIC_PAYLOAD] Model: {effective_model}")
                    logger.info(
                        f"[ANTHROPIC_PAYLOAD] Total messages: {len(conversation)}"
                    )

                    # Log system prompt (first 500 chars)
                    if system_prompt:
                        logger.info(
                            f"[ANTHROPIC_PAYLOAD] System prompt length: {len(system_prompt)}"
                        )
                        logger.info(
                            f"[ANTHROPIC_PAYLOAD] System prompt (first 500): {system_prompt[:500]}..."
                        )

                    # Log FULL content of first 3 messages to debug session.context
                    logger.info(
                        "[ANTHROPIC_PAYLOAD] --- First 3 messages (FULL CONTENT) ---"
                    )
                    for i, msg in enumerate(conversation[:3]):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        # Handle content that might be a list (for multimodal)
                        if isinstance(content, list):
                            # For list content, show first text block fully
                            text_blocks = [
                                b
                                for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            ]
                            if text_blocks:
                                full_content = text_blocks[0].get("text", "")
                            else:
                                full_content = str(content)
                        else:
                            full_content = content or ""

                        has_system_tag = "<SystemMessage>" in str(full_content)
                        logger.info(
                            f"[ANTHROPIC_PAYLOAD] msg[{i}]: role={role}, len={len(full_content)}, has_system_tag={has_system_tag}"
                        )
                        # Log full content for debugging (up to 2000 chars)
                        if len(full_content) > 2000:
                            logger.info(
                                f"[ANTHROPIC_PAYLOAD] msg[{i}] CONTENT (first 2000):\n{full_content[:2000]}\n... (truncated)"
                            )
                        else:
                            logger.info(
                                f"[ANTHROPIC_PAYLOAD] msg[{i}] CONTENT:\n{full_content}"
                            )

                    if len(conversation) > 3:
                        logger.info(
                            f"[ANTHROPIC_PAYLOAD] ... and {len(conversation) - 3} more messages"
                        )
                    logger.info(
                        "[ANTHROPIC_PAYLOAD] =========================================="
                    )

                    # Build tools list with web search support
                    if tools_payload:
                        request_kwargs["tools"] = tools_payload

                    start_time = datetime.now(timezone.utc)

                    # Create LLMCall to store raw request payload
                    should_log_llm_call = db == "STAGE"
                    logger.info(
                        f"[ANTHROPIC_LLMCALL] Checking if should log: db={db}, "
                        f"initial_should_log={should_log_llm_call}, "
                        f"user_id={llm_call_metadata.get('user')}, "
                        f"session={llm_call_metadata.get('session')}"
                    )
                    if not should_log_llm_call and llm_call_metadata.get("user"):
                        try:
                            user = User.from_mongo(llm_call_metadata.get("user"))
                            should_log_llm_call = user.is_admin()
                            logger.info(
                                f"[ANTHROPIC_LLMCALL] User lookup: is_admin={should_log_llm_call}"
                            )
                        except ValueError as e:
                            logger.warning(
                                f"[ANTHROPIC_LLMCALL] User lookup failed: {e}"
                            )
                            pass  # User not found in current DB environment

                    logger.info(
                        f"[ANTHROPIC_LLMCALL] Final should_log_llm_call={should_log_llm_call}"
                    )
                    if should_log_llm_call:
                        try:
                            truncated_payload = truncate_base64_in_payload(
                                request_kwargs
                            )

                            # Parse session_id - it may be prefixed with DB name like "STAGE-{id}"
                            session_id_raw = llm_call_metadata.get("session")
                            session_oid = None
                            if session_id_raw:
                                # Extract ObjectId from "DB-{id}" format if present
                                if "-" in session_id_raw:
                                    session_oid = ObjectId(
                                        session_id_raw.split("-", 1)[1]
                                    )
                                else:
                                    session_oid = ObjectId(session_id_raw)

                            llm_call = LLMCall(
                                provider=self.provider_name,
                                model=effective_model,
                                request_payload=truncated_payload,
                                start_time=start_time,
                                status="pending",
                                session=session_oid,
                                agent=ObjectId(llm_call_metadata.get("agent"))
                                if llm_call_metadata.get("agent")
                                else None,
                                user=ObjectId(llm_call_metadata.get("user"))
                                if llm_call_metadata.get("user")
                                else None,
                            )
                            llm_call.save()
                            logger.info(
                                f"[ANTHROPIC_LLMCALL] Created LLMCall id={llm_call.id}"
                            )
                        except Exception as e:
                            logger.error(
                                f"[ANTHROPIC_LLMCALL] Failed to create LLMCall: {e}"
                            )
                            import traceback

                            logger.error(
                                f"[ANTHROPIC_LLMCALL] Traceback: {traceback.format_exc()}"
                            )

                    betas = []
                    if response_format_class or output_format_payload:
                        betas.append("structured-outputs-2025-11-13")
                    if self._deferred_tools_enabled() and tools_payload:
                        betas.append("advanced-tool-use-2025-11-20")
                    beta_kwargs = {"betas": betas} if betas else {}

                    if response_format_class:
                        # Pydantic class - use parse() for structured outputs
                        # SDK streaming doesn't support output_format parameter
                        response = await self.client.beta.messages.parse(
                            **request_kwargs,
                            **beta_kwargs,
                            output_format=response_format_class,
                            timeout=httpx.Timeout(600.0, connect=10.0),
                        )
                    elif output_format_payload:
                        # Dict schema - can't use streaming, use long timeout instead
                        response = await self.client.beta.messages.create(
                            **request_kwargs,
                            **beta_kwargs,
                            output_format=output_format_payload,
                            timeout=httpx.Timeout(600.0, connect=10.0),
                        )
                    else:
                        # No structured output - use streaming
                        if betas:
                            stream_manager = self.client.beta.messages.stream(
                                **request_kwargs,
                                **beta_kwargs,
                            )
                        else:
                            stream_manager = self.client.messages.stream(
                                **request_kwargs
                            )
                        async with stream_manager as stream:
                            response = await stream.get_final_message()

                    end_time = datetime.now(timezone.utc)
                    llm_response = self._to_llm_response(response)

                    # Update LLMCall with response data
                    if should_log_llm_call and llm_call:
                        duration_ms = int(
                            (end_time - start_time).total_seconds() * 1000
                        )
                        llm_call.update(
                            status="completed",
                            end_time=end_time,
                            duration_ms=duration_ms,
                            response_payload=self._serialize_llm_response(llm_response),
                            prompt_tokens=llm_response.prompt_tokens,
                            completion_tokens=llm_response.completion_tokens,
                            total_tokens=llm_response.tokens_spent,
                            cost_usd=llm_response.usage.cost_usd
                            if llm_response.usage
                            else None,
                        )

                        # Attach llm_call_id to response
                        llm_response.llm_call_id = llm_call.id

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
                # Update LLMCall with error if it was created
                if llm_call is not None:
                    llm_call.update(
                        status="failed",
                        error=str(exc),
                        end_time=datetime.now(timezone.utc),
                    )
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

    def _supports_web_search(self, model_name: str) -> bool:
        """Check if a model supports web search.

        Web search is available on Claude 3.7 Sonnet, 3.5 Sonnet, and 3.5 Haiku.
        """
        normalized = model_name.lower().strip()
        # Web search supported models
        web_search_patterns = [
            "claude-3-7-sonnet",
            "claude-3.7-sonnet",
            "claude-3-5-sonnet",
            "claude-3.5-sonnet",
            "claude-3-5-haiku",
            "claude-3.5-haiku",
            "claude-sonnet-4",  # Newer naming
            "claude-haiku-4",  # Newer naming
        ]
        for pattern in web_search_patterns:
            if pattern in normalized:
                return True
        return False

    def _deferred_tools_enabled(self) -> bool:
        return os.getenv("FF_DEFERRED_TOOLS", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _build_tools_payload(
        self, tools: Optional[List[Dict[str, Any]]], model_name: str
    ) -> Optional[List[Dict[str, Any]]]:
        all_tools: List[Dict[str, Any]] = []
        if tools:
            all_tools.extend(tools)
        if self._supports_web_search(model_name):
            all_tools.append(WEB_SEARCH_TOOL)
        if not all_tools:
            return None
        if not self._deferred_tools_enabled():
            return all_tools

        deferred_tools: List[Dict[str, Any]] = []
        has_tool_search = False
        for tool in all_tools:
            tool_copy = dict(tool)
            if (
                tool_copy.get("name") == TOOL_SEARCH_TOOL_BM25["name"]
                or tool_copy.get("type") == TOOL_SEARCH_TOOL_BM25["type"]
            ):
                tool_copy.pop("defer_loading", None)
                has_tool_search = True
            else:
                tool_copy["defer_loading"] = True
            deferred_tools.append(tool_copy)

        if not has_tool_search:
            deferred_tools.insert(0, dict(TOOL_SEARCH_TOOL_BM25))

        return deferred_tools

    async def prompt_stream(self, context: LLMContext):
        include_thoughts = bool(context.config.reasoning_effort)
        system_prompt, conversation = self._prepare_messages(
            context.messages, include_thoughts=include_thoughts
        )
        tools = construct_anthropic_tools(context)

        response_format = context.config.response_format
        response_format_class = None
        output_format_payload = None

        if response_format is not None:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                response_format_class = response_format
            else:
                output_format_payload = self._build_output_format_payload(
                    response_format
                )

        if response_format_class or output_format_payload:
            response = await self.prompt(context)
            yield self._response_to_chunk(response)
            return

        # Extract context metadata for LLMCall
        llm_call_metadata = {}
        if context.metadata and context.metadata.trace_metadata:
            tm = context.metadata.trace_metadata
            llm_call_metadata = {
                "session": tm.session_id,
                "agent": tm.agent_id,
                "user": tm.user_id,
            }

        last_error: Optional[Exception] = None
        for model_name in self.models:
            tools_payload = self._build_tools_payload(tools, model_name)
            request_kwargs = {
                "model": model_name,
                "system": system_prompt,
                "messages": conversation,
                "max_tokens": context.config.max_tokens or 32000,
            }
            if tools_payload:
                request_kwargs["tools"] = tools_payload

            betas = []
            if self._deferred_tools_enabled() and tools_payload:
                betas.append("advanced-tool-use-2025-11-20")
            beta_kwargs = {"betas": betas} if betas else {}

            # Determine if we should log LLMCall
            should_log_llm_call = db == "STAGE"
            if not should_log_llm_call and llm_call_metadata.get("user"):
                try:
                    user = User.from_mongo(llm_call_metadata.get("user"))
                    should_log_llm_call = user.is_admin()
                except ValueError:
                    pass  # User not found in current DB environment

            # Create LLMCall before streaming starts
            llm_call = None
            start_time = datetime.now(timezone.utc)
            if should_log_llm_call:
                try:
                    truncated_payload = truncate_base64_in_payload(request_kwargs)

                    # Parse session_id - it may be prefixed with DB name like "STAGE-{id}"
                    session_id_raw = llm_call_metadata.get("session")
                    session_oid = None
                    if session_id_raw:
                        if "-" in session_id_raw:
                            session_oid = ObjectId(session_id_raw.split("-", 1)[1])
                        else:
                            session_oid = ObjectId(session_id_raw)

                    llm_call = LLMCall(
                        provider=self.provider_name,
                        model=model_name,
                        request_payload=truncated_payload,
                        start_time=start_time,
                        status="pending",
                        session=session_oid,
                        agent=ObjectId(llm_call_metadata.get("agent"))
                        if llm_call_metadata.get("agent")
                        else None,
                        user=ObjectId(llm_call_metadata.get("user"))
                        if llm_call_metadata.get("user")
                        else None,
                    )
                    llm_call.save()
                    logger.info(
                        f"[ANTHROPIC_STREAM_LLMCALL] Created LLMCall id={llm_call.id}"
                    )
                except Exception as e:
                    logger.error(
                        f"[ANTHROPIC_STREAM_LLMCALL] Failed to create LLMCall: {e}"
                    )

            stop_reason = None
            tool_calls: Dict[int, Dict[str, Any]] = {}
            accumulated_content = ""
            final_message = None
            try:
                if betas:
                    stream_manager = self.client.beta.messages.stream(
                        **request_kwargs,
                        **beta_kwargs,
                    )
                else:
                    stream_manager = self.client.messages.stream(**request_kwargs)

                async with stream_manager as stream:
                    async for event in stream:
                        if event.type == "content_block_start":
                            content_block = getattr(event, "content_block", None)
                            if (
                                content_block
                                and getattr(content_block, "type", None) == "tool_use"
                            ):
                                existing = tool_calls.get(event.index)
                                tool_calls[event.index] = {
                                    "id": content_block.id,
                                    "name": content_block.name,
                                    "arguments": (
                                        existing.get("arguments", "")
                                        if existing
                                        else json.dumps(content_block.input or {})
                                    ),
                                }
                        elif event.type == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta is None:
                                continue
                            if getattr(delta, "type", None) == "text_delta":
                                text = getattr(delta, "text", None)
                                if text:
                                    accumulated_content += text
                                    yield SimpleNamespace(
                                        choices=[
                                            SimpleNamespace(
                                                delta=SimpleNamespace(
                                                    content=text, tool_calls=None
                                                ),
                                                finish_reason=None,
                                            )
                                        ],
                                        usage=None,
                                    )
                            elif getattr(delta, "type", None) == "input_json_delta":
                                partial_json = getattr(delta, "partial_json", "")
                                tool_call = tool_calls.get(event.index)
                                if not tool_call:
                                    tool_call = {
                                        "id": f"toolu_{uuid.uuid4()}",
                                        "name": "",
                                        "arguments": "",
                                    }
                                    tool_calls[event.index] = tool_call
                                tool_call["arguments"] += partial_json
                                if not tool_call.get("name"):
                                    continue
                                yield SimpleNamespace(
                                    choices=[
                                        SimpleNamespace(
                                            delta=SimpleNamespace(
                                                content=None,
                                                tool_calls=[
                                                    SimpleNamespace(
                                                        index=event.index,
                                                        id=tool_call["id"],
                                                        function=SimpleNamespace(
                                                            name=tool_call["name"],
                                                            arguments=partial_json,
                                                        ),
                                                    )
                                                ],
                                            ),
                                            finish_reason=None,
                                        )
                                    ],
                                    usage=None,
                                )
                        elif event.type == "message_delta":
                            delta = getattr(event, "delta", None)
                            if delta and getattr(delta, "stop_reason", None):
                                stop_reason = delta.stop_reason

                    # Get final message to extract usage info
                    final_message = await stream.get_final_message()

                # Update LLMCall with response data
                end_time = datetime.now(timezone.utc)
                llm_call_id = None
                if should_log_llm_call and llm_call:
                    try:
                        duration_ms = int(
                            (end_time - start_time).total_seconds() * 1000
                        )

                        # Extract usage from final message
                        usage = final_message.usage if final_message else None
                        prompt_tokens = usage.input_tokens if usage else None
                        completion_tokens = usage.output_tokens if usage else None
                        total_tokens = (
                            (prompt_tokens or 0) + (completion_tokens or 0)
                            if usage
                            else None
                        )

                        # Calculate cost
                        cost_usd = None
                        if prompt_tokens and completion_tokens:
                            _, _, cost_usd = calculate_cost_usd(
                                model_name,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                            )

                        # Build response payload
                        response_payload = {
                            "content": accumulated_content,
                            "stop": stop_reason,
                            "tokens_spent": total_tokens,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        }
                        if tool_calls:
                            response_payload["tool_calls"] = [
                                {
                                    "id": tc["id"],
                                    "tool": tc["name"],
                                    "args": tc["arguments"],
                                }
                                for tc in tool_calls.values()
                            ]

                        llm_call.update(
                            status="completed",
                            end_time=end_time,
                            duration_ms=duration_ms,
                            response_payload=response_payload,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            cost_usd=cost_usd,
                        )
                        llm_call_id = llm_call.id
                        logger.info(
                            f"[ANTHROPIC_STREAM_LLMCALL] Updated LLMCall id={llm_call.id} "
                            f"status=completed tokens={total_tokens}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[ANTHROPIC_STREAM_LLMCALL] Failed to update LLMCall: {e}"
                        )

                # Include llm_call_id in final chunk
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None, tool_calls=None),
                            finish_reason=stop_reason or "stop",
                        )
                    ],
                    usage=SimpleNamespace(
                        total_tokens=(
                            (final_message.usage.input_tokens or 0)
                            + (final_message.usage.output_tokens or 0)
                        )
                        if final_message and final_message.usage
                        else 0
                    ),
                    llm_call_id=llm_call_id,
                )
                return
            except Exception as exc:
                last_error = exc
                # Update LLMCall with error if it was created
                if llm_call is not None:
                    try:
                        llm_call.update(
                            status="failed",
                            error=str(exc),
                            end_time=datetime.now(timezone.utc),
                        )
                    except Exception as update_err:
                        logger.error(
                            f"[ANTHROPIC_STREAM_LLMCALL] Failed to update LLMCall with error: {update_err}"
                        )
                logger.warning(
                    f"[ANTHROPIC_STREAM] Failed streaming with {model_name}: {exc}"
                )
                continue

        if last_error:
            raise last_error

    def _prepare_messages(
        self, messages: List[ChatMessage], include_thoughts: bool = False
    ) -> (Optional[str], List[Dict[str, Any]]):
        system_prompt_parts: List[str] = []
        conversation: List[Dict[str, Any]] = []

        logger.info(
            f"[ANTHROPIC_PREPARE] === _prepare_messages input ({len(messages)} messages) ==="
        )
        for idx, chat_message in enumerate(messages):
            content_preview = (
                (chat_message.content[:150] + "...")
                if chat_message.content and len(chat_message.content) > 150
                else chat_message.content
            )
            has_system_tag = (
                chat_message.content and "<SystemMessage>" in chat_message.content
                if chat_message.content
                else False
            )
            logger.info(
                f"[ANTHROPIC_PREPARE] input[{idx}]: role={chat_message.role}, "
                f"has_system_tag={has_system_tag}, content_preview={content_preview}"
            )

            schemas = chat_message.anthropic_schema(include_thoughts=include_thoughts)
            if not schemas:
                logger.warning(
                    f"[ANTHROPIC_PREPARE] input[{idx}] returned EMPTY schema - filtered out!"
                )
            for schema in schemas:
                role = schema.get("role")
                if role == "system":
                    system_prompt_parts.append(schema.get("content") or "")
                else:
                    conversation.append(schema)

        logger.info(
            f"[ANTHROPIC_PREPARE] === _prepare_messages output ({len(conversation)} conversation msgs) ==="
        )
        for idx, msg in enumerate(conversation[:5]):
            content = msg.get("content", "")
            if isinstance(content, list):
                content_preview = (
                    str(content)[:150] + "..."
                    if len(str(content)) > 150
                    else str(content)
                )
            else:
                content_preview = (
                    (content[:150] + "...")
                    if content and len(content) > 150
                    else content
                )
            has_system_tag = "<SystemMessage>" in str(content) if content else False
            logger.info(
                f"[ANTHROPIC_PREPARE] output[{idx}]: role={msg.get('role')}, "
                f"has_system_tag={has_system_tag}, content_preview={content_preview}"
            )

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
            "max_tokens": context.config.max_tokens or 32000,
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
        web_search_sources: List[Dict[str, str]] = []

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
            elif block_type == "web_search_tool_result":
                # Extract citations from web search results
                search_results = getattr(block, "content", [])
                for result in search_results:
                    result_type = getattr(result, "type", None)
                    if result_type == "web_search_result":
                        source = {
                            "title": getattr(result, "title", None),
                            "url": getattr(result, "url", None),
                        }
                        if source not in web_search_sources:
                            web_search_sources.append(source)
            elif block_type in {"thinking", "redacted_thinking"}:
                payload = {"type": block_type}
                if hasattr(block, "thinking"):
                    payload["thinking"] = block.thinking
                if hasattr(block, "signature") and block.signature:
                    payload["signature"] = block.signature
                if hasattr(block, "data"):
                    payload["data"] = block.data
                thoughts.append(payload)

        # Add web search as a completed tool call if we have sources
        if web_search_sources:
            tool_calls.insert(
                0,
                ToolCall(
                    id=f"toolu_{uuid.uuid4()}",
                    tool="web_search",
                    args={},
                    result=web_search_sources,
                    status="completed",
                ),
            )

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
