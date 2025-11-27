from __future__ import annotations

import base64
import logging as logger
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import google.genai as genai
import httpx
from bson import ObjectId as BsonObjectId
from google.genai import types as genai_types

from eve.agent.llm.providers import LLMProvider
from eve.agent.llm.util import (
    calculate_cost_usd,
    serialize_context_messages,
)
from eve.agent.session.models import (
    ChatMessage,
    LLMCall,
    LLMContext,
    LLMResponse,
    LLMUsage,
)


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

    async def prompt(self, context: LLMContext) -> LLMResponse:
        if context.tools:
            raise NotImplementedError(
                "Google Gemini provider does not yet support tool usage"
            )

        system_instruction, contents = await self._prepare_contents(context.messages)

        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=context.config.max_tokens,
        )

        base_input_payload = self._build_input_payload(
            context=context,
            system_instruction=system_instruction,
            contents=contents,
            config=config,
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

                    # Create LLMCall record before API call
                    if os.getenv("DB") == "STAGE":
                        llm_call = LLMCall(
                            provider=self.provider_name,
                            model=canonical_name,
                            request_payload=request_payload,
                            start_time=start_time,
                            status="pending",
                            session=BsonObjectId(llm_call_metadata.get("session"))
                            if llm_call_metadata.get("session")
                            else None,
                            agent=BsonObjectId(llm_call_metadata.get("agent"))
                            if llm_call_metadata.get("agent")
                            else None,
                            user=BsonObjectId(llm_call_metadata.get("user"))
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
            serialized.append(
                {
                    "role": getattr(content, "role", None),
                    "parts": parts,
                }
            )
        return serialized

    def _build_input_payload(
        self,
        *,
        context: LLMContext,
        system_instruction: Optional[str],
        contents: List[genai_types.Content],
        config: genai_types.GenerateContentConfig,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "system_instruction": system_instruction,
            "contents": self._serialize_contents(contents),
            "max_output_tokens": getattr(config, "max_output_tokens", None),
            "fallback_models": list(context.config.fallback_models or []),
        }
        payload["context_messages"] = serialize_context_messages(context)
        return {k: v for k, v in payload.items() if v not in (None, [], {})}

    def _to_llm_response(
        self, response: genai_types.GenerateContentResponse
    ) -> LLMResponse:
        text_segments: List[str] = []
        stop_reason = None

        for candidate in response.candidates or []:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "text", None):
                        text_segments.append(part.text)
            if candidate.finish_reason:
                stop_reason = candidate.finish_reason

        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else None
        completion_tokens = usage.candidates_token_count if usage else None
        total_tokens = (
            (prompt_tokens or 0) + (completion_tokens or 0) if usage else None
        )
        usage_payload = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            content="\n".join(text_segments).strip(),
            tool_calls=None,
            stop=stop_reason,
            tokens_spent=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage=usage_payload,
            thought=None,
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
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
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
