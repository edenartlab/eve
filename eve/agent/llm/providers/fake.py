import json
import os
import uuid
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, Optional

from eve.agent.llm.constants import TEST_MODE_TEXT_STRING, TEST_MODE_TOOL_STRING
from eve.agent.llm.providers import LLMProvider
from eve.agent.llm.util import (
    get_last_assistant_message,
    get_last_completed_tool_call,
    get_last_user_prompt,
    is_test_mode_prompt,
)
from eve.agent.session.fake_utils import build_fake_tool_result_payload
from eve.agent.session.models import LLMContext, LLMResponse, ToolCall, LLMUsage
from eve.tool import Tool


def _build_fake_tool_args(tool: Tool) -> Dict[str, Any]:
    """Generate plausible arguments for a tool in fake mode."""
    if getattr(tool, "test_args", None):
        try:
            return tool.prepare_args(tool.test_args.copy())
        except Exception:
            pass

    parameters = getattr(tool, "parameters", {}) or {}
    candidate_args: Dict[str, Any] = {}

    for field_name, parameter in parameters.items():
        default = parameter.get("default")
        param_type = parameter.get("type")
        enum_values = parameter.get("enum") or []

        if default not in (None, "random"):
            candidate_args[field_name] = default
            continue
        if enum_values:
            candidate_args[field_name] = enum_values[0]
            continue
        if default == "random":
            if param_type in {"integer", "number"}:
                minimum = parameter.get("minimum", 0)
                candidate_args[field_name] = minimum
            else:
                candidate_args[field_name] = "sample"
            continue

        if param_type == "boolean":
            candidate_args[field_name] = True
        elif param_type == "integer":
            candidate_args[field_name] = parameter.get("minimum", 1)
        elif param_type == "number":
            candidate_args[field_name] = parameter.get("minimum", 1.0)
        elif param_type == "array":
            candidate_args[field_name] = []
        elif param_type == "object":
            candidate_args[field_name] = {}
        else:
            candidate_args[field_name] = "sample"

    try:
        return tool.prepare_args(candidate_args.copy())
    except Exception:
        return candidate_args


async def async_prompt_fake(
    context: LLMContext,
) -> LLMResponse:
    """Return a simulated LLM response without reaching a provider."""
    prompt_text = get_last_user_prompt(context) or ""
    prompt_clean = prompt_text.strip()
    prompt_lower = prompt_clean.lower()

    last_tool_call = get_last_completed_tool_call(context)
    last_assistant = get_last_assistant_message(context)

    if (
        last_tool_call
        and last_assistant
        and getattr(last_assistant, "finish_reason", None) == "tool_calls"
        and is_test_mode_prompt(prompt_clean)
    ):
        result_summary = last_tool_call.result or []
        body = (
            f"[Simulated Tool Result] Tool '{last_tool_call.tool}' completed with"
            " placeholder output."
        )
        if result_summary and isinstance(result_summary, list):
            try:
                first_output = result_summary[0]["output"][0]
                url = first_output.get("url")
                if url:
                    body += f"\nPreview URL: {url}"
            except Exception:
                pass

        return LLMResponse(
            content=body,
            tool_calls=None,
            stop="stop",
            tokens_spent=0,
            usage=LLMUsage(total_tokens=0, prompt_tokens=0, completion_tokens=0),
            thought=None,
        )

    placeholder = os.getenv(
        "FAKE_LLM_PLACEHOLDER_TEXT", "This is a simulated response."
    )
    tool_calls = None
    stop_reason = "stop"

    wants_tool = (
        prompt_lower == "tool"
        or prompt_clean == TEST_MODE_TOOL_STRING
        or prompt_lower.endswith(TEST_MODE_TOOL_STRING)
        or TEST_MODE_TOOL_STRING in prompt_clean
    )

    if wants_tool and context.tools:
        tool_key, tool = next(iter(context.tools.items()))
        fake_args = _build_fake_tool_args(tool)
        result_payload = build_fake_tool_result_payload(
            tool_key, getattr(tool, "name", None)
        )

        tool_calls = [
            ToolCall(
                id=f"toolu_{uuid.uuid4()}",
                tool=tool_key,
                args=fake_args,
                status="completed",
                result=result_payload,
                cost=0,
            )
        ]
        content = ""
        stop_reason = "tool_calls"
    else:
        if prompt_clean == TEST_MODE_TEXT_STRING:
            body = "[Test Mode] Simulated response."
        elif wants_tool and not context.tools:
            body = "[Simulated Response]\nNo tools available for test mode."
        else:
            body = f"[Simulated Response]\n{placeholder}"
            if prompt_clean:
                body += f"\n\n(Original prompt: {prompt_clean})"
        content = body

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        stop=stop_reason,
        tokens_spent=0,
        usage=LLMUsage(total_tokens=0, prompt_tokens=0, completion_tokens=0),
        thought=None,
    )


async def async_prompt_stream_fake(
    context: LLMContext,
) -> AsyncGenerator[Any, None]:
    """Yield a fake streaming response compatible with the existing consumer."""
    response = await async_prompt_fake(context)

    delta = SimpleNamespace(
        content=response.content if response.content else None,
        tool_calls=None,
    )

    if response.tool_calls and any(
        tc.status != "completed" for tc in response.tool_calls
    ):
        tool_calls = []
        for index, tool_call in enumerate(response.tool_calls):
            function_payload = SimpleNamespace(
                name=tool_call.tool,
                arguments=json.dumps(tool_call.args or {}),
            )
            tool_calls.append(
                SimpleNamespace(
                    index=index,
                    id=tool_call.id,
                    function=function_payload,
                )
            )
        delta = SimpleNamespace(content=None, tool_calls=tool_calls)

    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=delta,
                finish_reason=response.stop or "stop",
            )
        ],
        usage=SimpleNamespace(total_tokens=response.tokens_spent or 0),
    )

    yield chunk


class FakeProvider(LLMProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def prompt(self, context: LLMContext) -> LLMResponse:
        return await async_prompt_fake(context)

    async def prompt_stream(self, context: LLMContext) -> AsyncGenerator[Any, None]:
        async for chunk in async_prompt_stream_fake(context):
            yield chunk
