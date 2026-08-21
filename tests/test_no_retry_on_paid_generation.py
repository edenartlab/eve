"""A paid generation must be bought exactly once per manna charge.

Manna is spent up front, once, before the handler runs
(Tool.handle_start_task). Any retry that re-issues the provider request after
the provider has accepted the work therefore buys the same output twice or
three times against a single charge — and refunds only ever return the one
charge, so every extra provider attempt is pure loss.

Covers the two non-fal offenders (fal has its own file,
test_fal_no_double_billing.py):

* ElevenLabs — five paid handlers wrapped generation in
  ``async_exponential_backoff``, which retries a bare ``Exception``: a 422 was
  re-billed exactly like a 503.
* Runway — three handlers wrapped ``*.create`` in tenacity with
  ``retry_if_exception_type((APIConnectionError, APIStatusError))``, and
  ``APIStatusError`` is the base class of every 4xx and 5xx.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import runwayml

from eve.tool import ToolContext
from eve.utils.system_utils import async_retry_if_unbilled

# ---------------------------------------------------------------------------
# ElevenLabs: the shared predicate
# ---------------------------------------------------------------------------


def _http_error(status: int) -> Exception:
    request = httpx.Request("POST", "https://api.elevenlabs.io/v1/music")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


async def _counting(exc):
    """A paid call that always fails, counting how many times it was made."""
    calls = []

    async def call():
        calls.append(1)
        raise exc

    return call, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 404, 422, 500, 503])
async def test_billable_failures_are_not_retried(status):
    """Anything the provider may have generated for is bought once, not thrice.

    5xx is included on purpose: ElevenLabs can render the audio and still fail
    to deliver the response, so a 500 is not provably unbilled.
    """
    call, calls = await _counting(_http_error(status))
    with pytest.raises(httpx.HTTPStatusError):
        await async_retry_if_unbilled(call, max_attempts=3, initial_delay=0)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_read_timeout_is_not_retried():
    """A read timeout means the request WAS sent; the audio may already exist."""
    call, calls = await _counting(httpx.ReadTimeout("timed out"))
    with pytest.raises(httpx.ReadTimeout):
        await async_retry_if_unbilled(call, max_attempts=3, initial_delay=0)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_local_bug_is_not_retried():
    """The bare KeyError('voice') that used to burn three attempts and 4 minutes."""
    call, calls = await _counting(KeyError("voice"))
    with pytest.raises(KeyError):
        await async_retry_if_unbilled(call, max_attempts=3, initial_delay=0)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [httpx.ConnectError("no route"), httpx.ConnectTimeout("nope"), _http_error(429)],
)
async def test_provably_unbilled_failures_are_retried(exc):
    """Never connected, or rejected outright — nothing was generated."""
    call, calls = await _counting(exc)
    with pytest.raises(type(exc)):
        await async_retry_if_unbilled(call, max_attempts=3, initial_delay=0)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_recovers_after_a_transient_connect_failure():
    calls = []

    async def call():
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("no route")
        return "audio"

    assert await async_retry_if_unbilled(call, max_attempts=3, initial_delay=0) == (
        "audio"
    )
    assert len(calls) == 2


def test_vendor_sdk_status_codes_are_understood():
    """The ElevenLabs SDK raises ApiError(status_code=...), not an httpx error."""
    from elevenlabs.core.api_error import ApiError

    from eve.utils.system_utils import _is_provably_unbilled

    assert _is_provably_unbilled(ApiError(status_code=429))
    assert not _is_provably_unbilled(ApiError(status_code=422))
    assert not _is_provably_unbilled(ApiError(status_code=500))


# ---------------------------------------------------------------------------
# ElevenLabs: elevenlabs_speech validates segments before spending anything
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "segments",
    [
        [{"text": "hello"}],  # the prod failure: no 'voice'
        [{"voice": "Charlotte"}],
        [{"text": "hi", "voice": "Charlotte"}, {"text": "there"}],
    ],
)
async def test_speech_rejects_incomplete_segments_before_calling_out(segments):
    from eve.tools.elevenlabs_speech import handler as speech

    with patch.object(speech, "eleven") as eleven:
        with pytest.raises(ValueError) as excinfo:
            await speech.handler(ToolContext(args={"segments": segments}))

    # Actionable, and no provider call was made.
    assert "missing required field" in str(excinfo.value)
    eleven.voices.get_all.assert_not_called()


# ---------------------------------------------------------------------------
# Runway: submit once
# ---------------------------------------------------------------------------


def _runway_error(status: int) -> runwayml.APIStatusError:
    request = httpx.Request("POST", "https://api.dev.runwayml.com/v1/image_to_video")
    response = httpx.Response(status, request=request, text="boom")
    return runwayml.APIStatusError("boom", response=response, body=None)


RUNWAY_CASES = [
    (
        "eve.tools.runway.handler",
        "image_to_video",
        {
            "prompt_text": "a cat",
            "ratio": "16:9",
            "duration": 5,
            "start_image": "https://x/a.png",
        },
    ),
    (
        "eve.tools.runway2.handler",
        "character_performance",
        {
            "ratio": "16:9",
            "character_image": "https://x/a.png",
            "reference_video": "https://x/a.mp4",
            "body_control": True,
            "expression_intensity": 3,
            "seed": 1,
        },
    ),
    (
        "eve.tools.runway3.handler",
        "video_to_video",
        {"input_video": "https://x/a.mp4", "prompt_text": "film noir", "ratio": "16:9"},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 429, 500])
@pytest.mark.parametrize("module,namespace,args", RUNWAY_CASES)
async def test_runway_submits_exactly_once(module, namespace, args, status):
    """A failed create is never re-issued — a second create is a second paid task.

    400 is the one the old predicate got most wrong (APIStatusError is its base
    class, so a deterministic client error was "retryable"); 429 and 500 are
    already retried inside the runwayml SDK, one layer down, before a task
    exists.
    """
    import importlib

    mod = importlib.import_module(module)

    create = AsyncMock(side_effect=_runway_error(status))
    client = MagicMock()
    getattr(client, namespace).create = create

    with patch.object(mod, "AsyncRunwayML", return_value=client):
        with pytest.raises(Exception):
            await mod.handler(ToolContext(args=args))

    assert create.call_count == 1


@pytest.mark.parametrize("module,_namespace,_args", RUNWAY_CASES)
def test_runway_handlers_have_no_retry_wrapper(module, _namespace, _args):
    """Guard the regression at the source: no tenacity in the submit path."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))
    assert "tenacity" not in source
    assert "@retry" not in source
