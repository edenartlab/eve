"""A run whose cost-plus top-up cannot be collected must stop taking turns.

The regression this locks down: the 2-manna floor is charged once per RUN, so
after turn 1 nothing re-asserts the balance. Cost-plus top-ups are deliberately
best-effort — a post-hoc charge must not kill a response the user already
received — but the failure was previously only logged. An account funded with 2
manna therefore bought a full MAX_PROMPT_SESSION_TURNS tool loop: the floor
drains the balance on turn 1, and every later top-up raises and is swallowed.

The failure is now recorded and enforced at the next turn boundary, which keeps
the already-delivered response intact while refusing to start another turn.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from eve.agent.session.models import ChatMessage
from eve.agent.session.runtime import PromptSessionRuntime
from eve.api.errors import APIError

MAX_TURNS = 5


def _runtime(fail_billing_on_turn=None):
    """A runtime carrying only the attributes _prompt_loop touches."""
    r = object.__new__(PromptSessionRuntime)
    r.instrumentation = None
    r.session_run_id = "run-1"
    r.billing_user_id = None
    r.billing_user_doc = None
    r.billed_user_doc = None
    r.rate_limiter = None
    r.stream = False
    r.llm_context = MagicMock()
    r.session = MagicMock()
    r.context = None
    r.active_platform = None
    r.social_post_used = False
    r.social_reprompt_attempted = False

    actor = MagicMock()
    actor.id = "agent-1"
    actor.username = "agent"
    actor.name = "Agent"
    actor.userImage = None
    r.actor = actor

    r._register_active_request = MagicMock()
    r._start_update = MagicMock(return_value="START")
    r._ensure_not_cancelled = MagicMock()
    r._refresh_llm_messages = AsyncMock()
    r._maybe_disable_tools = MagicMock()
    r._charge_manna_for_message = MagicMock()
    r._persist_billing_error_message = MagicMock(
        return_value=ChatMessage(role="eden", content="insufficient manna")
    )
    r._select_provider = MagicMock(return_value="provider")
    # "tool_use" keeps the loop going, so it runs until max_turns or the guard.
    r._non_stream_llm_response = AsyncMock(return_value={"stop_reason": "tool_use"})
    r._maybe_notify_user = AsyncMock()

    async def _no_tool_updates(_message):
        return
        yield  # pragma: no cover - makes this an async generator

    r._process_tool_calls = _no_tool_updates

    turns = {"n": 0}

    async def _persist(_llm_result):
        turns["n"] += 1
        if fail_billing_on_turn is not None and turns["n"] == fail_billing_on_turn:
            # What the cost-plus except block now records.
            r._billing_failed_error = APIError(
                "Insufficient manna", status_code=402
            )
        return ChatMessage(role="assistant", content="ok")

    r._persist_assistant_message = _persist
    return r


async def _drain(runtime):
    return [u async for u in runtime._prompt_loop()]


@pytest.mark.asyncio
async def test_failed_topup_stops_the_run_after_the_current_turn(monkeypatch):
    monkeypatch.setenv("FF_MANNA_BILLING", "1")
    monkeypatch.setenv("MAX_PROMPT_SESSION_TURNS", str(MAX_TURNS))
    runtime = _runtime(fail_billing_on_turn=1)

    await _drain(runtime)

    # Turn 1 completed and was delivered; turn 2 never reached the LLM.
    assert runtime._select_provider.call_count == 1
    runtime._persist_billing_error_message.assert_called_once()


@pytest.mark.asyncio
async def test_a_healthy_run_keeps_taking_turns(monkeypatch):
    """The guard must not fire when every top-up succeeded."""
    monkeypatch.setenv("FF_MANNA_BILLING", "1")
    monkeypatch.setenv("MAX_PROMPT_SESSION_TURNS", str(MAX_TURNS))
    runtime = _runtime(fail_billing_on_turn=None)

    await _drain(runtime)

    assert runtime._select_provider.call_count == MAX_TURNS
    runtime._persist_billing_error_message.assert_not_called()


@pytest.mark.asyncio
async def test_guard_is_inert_when_manna_billing_is_disabled(monkeypatch):
    monkeypatch.delenv("FF_MANNA_BILLING", raising=False)
    monkeypatch.setenv("MAX_PROMPT_SESSION_TURNS", str(MAX_TURNS))
    runtime = _runtime(fail_billing_on_turn=1)

    await _drain(runtime)

    assert runtime._select_provider.call_count == MAX_TURNS
    runtime._persist_billing_error_message.assert_not_called()
