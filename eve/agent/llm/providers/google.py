from __future__ import annotations

import base64
import logging as logger
import os
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import google.genai as genai
import httpx
from bson import ObjectId as ObjectId
from google.genai import types as genai_types

from eve import db
from eve.agent.llm.formatting import construct_gemini_tools
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

# Map eve reasoning_effort values to Gemini thinking_level values
# Note: Gemini only supports LOW and HIGH (no medium)
REASONING_TO_THINKING_LEVEL = {
    "low": "LOW",
    "medium": "LOW",  # Gemini doesn't have medium, map to LOW
    "high": "HIGH",
}


class GoogleProvider(LLMProvider):
    provider_name = "google"

    def __init__(
        self,
        model: str,
        fallbacks: Optional[List[str]] = None,
        instrumentation=None,
    ):
        super().__init__(instrumentation=instrumentation)
        self.models = [model] + [m for m in (fallbacks or []) if m]
        api_key = (
            os.getenv("GOOGLE_GENAI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "GOOGLE_GENAI_API_KEY (or GOOGLE_API_KEY) is required for GoogleProvider"
            )
        self.client = genai.Client(api_key=api_key)
        self.async_models = self.client.aio.models
        self._aio_session = None

    async def _ensure_cleanup(self):
        """Ensure async resources are cleaned up"""
        try:
            # Access the internal httpx client and close it
            if hasattr(self.async_models, "_client"):
                client = self.async_models._client
                if hasattr(client, "aclose"):
                    await client.aclose()
        except Exception:
            pass  # Ignore cleanup errors silently

    def _build_thinking_config(
        self, context: LLMContext
    ) -> Optional[genai_types.ThinkingConfig]:
        """Build ThinkingConfig from eve's reasoning_effort parameter.

        Maps eve's reasoning_effort (low/medium/high) to Gemini's thinking_level.
        Returns None if no thinking configuration is needed.
        """
        reasoning_effort = context.config.reasoning_effort
        if not reasoning_effort:
            return None

        thinking_level = REASONING_TO_THINKING_LEVEL.get(reasoning_effort.lower())
        if not thinking_level:
            return None

        return genai_types.ThinkingConfig(
            include_thoughts=True,
            thinking_level=thinking_level,
        )

    def _build_tools_config(
        self, context: LLMContext
    ) -> Optional[List[genai_types.Tool]]:
        """Build Gemini tools from eve's tool definitions."""
        function_declarations = construct_gemini_tools(context)
        if not function_declarations:
            return None

        # Convert dict declarations to Gemini FunctionDeclaration objects
        gemini_declarations = []
        for decl in function_declarations:
            gemini_declarations.append(
                genai_types.FunctionDeclaration(
                    name=decl["name"],
                    description=decl.get("description", ""),
                    parameters=decl.get("parameters"),
                )
            )

        return [genai_types.Tool(function_declarations=gemini_declarations)]

    def _build_tool_config(
        self, context: LLMContext
    ) -> Optional[genai_types.ToolConfig]:
        """Build ToolConfig to control function calling behavior.

        Maps eve's tool_choice to Gemini's function_calling_config mode.
        """
        if not context.tools:
            return None

        tool_choice = context.tool_choice
        if not tool_choice:
            # Default: AUTO - model decides when to call functions
            return genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(mode="AUTO")
            )

        # Map common tool_choice values
        if tool_choice == "auto":
            mode = "AUTO"
        elif tool_choice == "none":
            mode = "NONE"
        elif tool_choice == "required" or tool_choice == "any":
            mode = "ANY"
        else:
            # If a specific function name is provided, use ANY mode
            # Gemini doesn't support forcing a specific function by name in the same way
            mode = "ANY"

        return genai_types.ToolConfig(
            function_calling_config=genai_types.FunctionCallingConfig(mode=mode)
        )

    async def prompt(self, context: LLMContext) -> LLMResponse:
        system_instruction, contents = await self._prepare_contents(context.messages)

        # Build tools configuration
        tools = self._build_tools_config(context)
        tool_config = self._build_tool_config(context) if tools else None

        # Build thinking configuration (maps reasoning_effort to thinking_level)
        thinking_config = self._build_thinking_config(context)

        # Build the generation config
        config_kwargs = {
            "system_instruction": system_instruction,
            "max_output_tokens": context.config.max_tokens,
        }

        if tools:
            config_kwargs["tools"] = tools
        if tool_config:
            config_kwargs["tool_config"] = tool_config
        if thinking_config:
            config_kwargs["thinking_config"] = thinking_config

        config = genai_types.GenerateContentConfig(**config_kwargs)

        base_input_payload = self._build_input_payload(
            context=context,
            system_instruction=system_instruction,
            contents=contents,
            config=config,
            tools=tools,
            thinking_config=thinking_config,
        )

        # Extract metadata for LLMCall
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
            canonical_name = self._normalize_model_name(model_name)
            stage = (
                self.instrumentation.track_stage(
                    "llm.google", metadata={"model": canonical_name}
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

                    # Build request payload for LLMCall
                    request_payload = {
                        "model": canonical_name,
                        "system_instruction": system_instruction,
                        "contents": self._serialize_contents(contents),
                        "max_output_tokens": context.config.max_tokens,
                    }
                    if tools:
                        request_payload["tools"] = self._serialize_tools(tools)
                    if thinking_config:
                        request_payload["thinking_config"] = {
                            "include_thoughts": thinking_config.include_thoughts,
                            "thinking_level": thinking_config.thinking_level,
                        }

                    # Create LLMCall record before API call
                    should_log_llm_call = db == "STAGE"
                    if not should_log_llm_call and llm_call_metadata.get("user"):
                        try:
                            user = User.from_mongo(llm_call_metadata.get("user"))
                            should_log_llm_call = user.is_admin()
                        except ValueError:
                            pass  # User not found in current DB environment
                    if should_log_llm_call:
                        truncated_payload = truncate_base64_in_payload(request_payload)
                        llm_call = LLMCall(
                            provider=self.provider_name,
                            model=canonical_name,
                            request_payload=truncated_payload,
                            start_time=start_time,
                            status="pending",
                            session=ObjectId(llm_call_metadata.get("session"))
                            if llm_call_metadata.get("session")
                            else None,
                            agent=ObjectId(llm_call_metadata.get("agent"))
                            if llm_call_metadata.get("agent")
                            else None,
                            user=ObjectId(llm_call_metadata.get("user"))
                            if llm_call_metadata.get("user")
                            else None,
                        )
                        llm_call.save()

                    response = await self.async_models.generate_content(
                        model=canonical_name,
                        contents=contents,
                        config=config,
                    )
                    end_time = datetime.now(timezone.utc)
                    duration_ms = int((end_time - start_time).total_seconds() * 1000)
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

                    # Update LLMCall with response data
                    if should_log_llm_call:
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
                        llm_response.llm_call_id = llm_call.id

                    # Clean up async resources before returning
                    await self._ensure_cleanup()
                    return llm_response
            except Exception as exc:
                last_error = exc
                if self.instrumentation:
                    self.instrumentation.log_event(
                        f"Google model {canonical_name} failed",
                        level="warning",
                        payload={"error": str(exc)},
                    )

        # Clean up async resources before raising
        await self._ensure_cleanup()
        if last_error:
            raise last_error
        raise RuntimeError("Failed to obtain Google response")

    async def prompt_stream(self, context: LLMContext):
        response = await self.prompt(context)
        yield self._response_to_chunk(response)

    @staticmethod
    def _get_mime_type(url: str) -> str:
        """Determine MIME type from URL extension"""
        ext = urlparse(url).path.lower().split(".")[-1]
        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
        }
        return mime_types.get(ext, "image/jpeg")

    async def _fetch_image(self, url: str) -> tuple[bytes, str]:
        """Fetch image from URL and return bytes and MIME type"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            mime_type = self._get_mime_type(url)
            return response.content, mime_type

    async def _prepare_contents(
        self, messages: List[ChatMessage]
    ) -> (Optional[str], List[genai_types.Content]):
        system_parts: List[str] = []
        contents: List[genai_types.Content] = []

        for message in messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content.strip())
                continue

            text = message.content or ""
            parts: List[genai_types.Part] = []

            # Handle tool results first - these come as "tool" role messages in eve
            # Must be handled before text content to avoid duplicate content
            if message.role == "tool":
                tool_name = getattr(message, "name", None) or "unknown_tool"
                result_content = message.content or ""

                # Try to parse result as JSON, otherwise use as string
                try:
                    import json
                    result = json.loads(result_content)
                except (json.JSONDecodeError, TypeError):
                    result = {"result": result_content}

                parts.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_name,
                            response=result,
                        )
                    )
                )
                # Function responses should be in "user" role for Gemini
                contents.append(genai_types.Content(role="user", parts=parts))
                continue

            # Add text content if present
            if text.strip():
                parts.append(genai_types.Part(text=text))

            # Add image attachments if present
            if message.attachments:
                for attachment_url in message.attachments:
                    try:
                        # Fetch and encode the image
                        image_bytes, mime_type = await self._fetch_image(attachment_url)
                        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

                        # Create inline data blob
                        blob = genai_types.Blob(mime_type=mime_type, data=encoded_image)
                        parts.append(genai_types.Part(inline_data=blob))
                    except Exception as e:
                        # Log error but continue processing other attachments
                        logger.warning(
                            f"Warning: Failed to fetch image {attachment_url}: {e}"
                        )

            # Handle tool call results in assistant messages
            if message.role == "assistant" and message.tool_calls:
                for tc in message.tool_calls:
                    # Add function call part
                    parts.append(
                        genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name=tc.tool,
                                args=tc.args,
                            )
                        )
                    )

            # Only add content if there are parts
            if parts:
                role = "user" if message.role == "user" else "model"
                contents.append(genai_types.Content(role=role, parts=parts))

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    def _serialize_contents(
        self, contents: List[genai_types.Content]
    ) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for content in contents:
            parts: List[Dict[str, Any]] = []
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    parts.append({"text": text})
                # Serialize function calls
                func_call = getattr(part, "function_call", None)
                if func_call:
                    parts.append({
                        "function_call": {
                            "name": getattr(func_call, "name", ""),
                            "args": dict(getattr(func_call, "args", {})),
                        }
                    })
                # Serialize function responses
                func_response = getattr(part, "function_response", None)
                if func_response:
                    parts.append({
                        "function_response": {
                            "name": getattr(func_response, "name", ""),
                            "response": dict(getattr(func_response, "response", {})),
                        }
                    })
            serialized.append(
                {
                    "role": getattr(content, "role", None),
                    "parts": parts,
                }
            )
        return serialized

    def _serialize_tools(
        self, tools: List[genai_types.Tool]
    ) -> List[Dict[str, Any]]:
        """Serialize tools for logging."""
        serialized = []
        for tool in tools:
            func_decls = getattr(tool, "function_declarations", []) or []
            declarations = []
            for decl in func_decls:
                declarations.append({
                    "name": getattr(decl, "name", ""),
                    "description": getattr(decl, "description", ""),
                    "parameters": getattr(decl, "parameters", {}),
                })
            serialized.append({"function_declarations": declarations})
        return serialized

    def _build_input_payload(
        self,
        *,
        context: LLMContext,
        system_instruction: Optional[str],
        contents: List[genai_types.Content],
        config: genai_types.GenerateContentConfig,
        tools: Optional[List[genai_types.Tool]] = None,
        thinking_config: Optional[genai_types.ThinkingConfig] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "system_instruction": system_instruction,
            "contents": self._serialize_contents(contents),
            "max_output_tokens": getattr(config, "max_output_tokens", None),
            "fallback_models": list(context.config.fallback_models or []),
        }
        if tools:
            payload["tools"] = self._serialize_tools(tools)
        if thinking_config:
            payload["thinking_config"] = {
                "include_thoughts": thinking_config.include_thoughts,
                "thinking_level": thinking_config.thinking_level,
            }
        payload["context_messages"] = serialize_context_messages(context)
        return {k: v for k, v in payload.items() if v not in (None, [], {})}

    def _to_llm_response(
        self, response: genai_types.GenerateContentResponse
    ) -> LLMResponse:
        text_segments: List[str] = []
        tool_calls: List[ToolCall] = []
        thoughts: List[Dict[str, Any]] = []
        stop_reason = None

        for candidate in response.candidates or []:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # Handle text content
                    if getattr(part, "text", None):
                        text_segments.append(part.text)

                    # Handle function calls
                    func_call = getattr(part, "function_call", None)
                    if func_call:
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:24]}",
                                tool=getattr(func_call, "name", ""),
                                args=dict(getattr(func_call, "args", {})),
                                status="pending",
                            )
                        )

                    # Handle thinking/thought content
                    # Gemini returns thought (actual content) and thought_signature
                    # (verification) as separate parts. We only store thinking content
                    # and attach signatures to it. Signatures alone have no displayable
                    # content and should not create UI elements.
                    thought = getattr(part, "thought", None)
                    thought_signature = getattr(part, "thought_signature", None)
                    if thought:
                        thought_entry = {
                            "type": "thinking",
                            "thinking": thought,
                        }
                        # Attach signature if present (for verification)
                        if thought_signature:
                            thought_entry["signature"] = thought_signature
                        thoughts.append(thought_entry)
                    # Note: thought_signature alone (without thought content) is not
                    # stored - it provides no value to the UI and causes empty blocks

            if candidate.finish_reason:
                stop_reason = str(candidate.finish_reason)
                # Normalize stop reason
                if stop_reason == "STOP":
                    stop_reason = "stop"
                elif stop_reason == "MAX_TOKENS":
                    stop_reason = "length"
                elif stop_reason == "TOOL_USE" or stop_reason == "FUNCTION_CALL":
                    stop_reason = "tool_calls"

        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else None
        completion_tokens = usage.candidates_token_count if usage else None
        # Include thinking tokens in total if available (Gemini 3 feature)
        thinking_tokens = (
            getattr(usage, "thinking_token_count", None) if usage else None
        )
        total_tokens = (
            (prompt_tokens or 0)
            + (completion_tokens or 0)
            + (thinking_tokens or 0)
            if usage
            else None
        )

        usage_payload = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            content="\n".join(text_segments).strip(),
            tool_calls=tool_calls if tool_calls else None,
            stop=stop_reason,
            tokens_spent=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage=usage_payload,
            thought=thoughts if thoughts else None,
        )

    def _record_usage(
        self,
        model: str,
        response: genai_types.GenerateContentResponse,
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

        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
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

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if "/" in model_name:
            provider, raw = model_name.split("/", 1)
            if provider in {"google", "gemini"}:
                return raw
        return model_name
