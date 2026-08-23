"""fal calls must bill exactly once per task.

The regression these lock down: `_call_with_retry` wrapped
`fal_client.subscribe` in a 4-attempt loop, and subscribe SUBMITS — so every
retry minted a new request_id, i.e. another paid generation charged against a
single manna spend. It decided to retry by substring-matching str(e) for any
number in 500..599, so a 422 whose detail carried "240x240" or a docs URL was
read as a server error and re-billed three more times.

fal_client already retries 408/409/429/5xx internally on every HTTP request
(client._should_retry), so the safe retry still happens — one layer down,
before a job exists.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from eve.tools.fal_tool import _format_error_for_user, call_fal


def _fal_error(status: int, detail: str) -> Exception:
    """A FalClientError shaped exactly as fal_client raises it."""
    from fal_client.client import FalClientError

    request = httpx.Request("POST", "https://queue.fal.run/fal-ai/x")
    response = httpx.Response(status, request=request)
    cause = httpx.HTTPStatusError("err", request=request, response=response)
    err = FalClientError(detail)
    err.__cause__ = cause
    return err


def _handle(result=None, exc=None):
    handle = MagicMock()
    handle.request_id = "req-abc"
    handle.iter_events.return_value = iter(())
    if exc is not None:
        handle.get.side_effect = exc
    else:
        handle.get.return_value = result
    return handle


# ---------------------------------------------------------------------------
# exactly-once submission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_submits_once():
    handle = _handle(result={"images": [{"url": "https://x/1.png"}]})
    with patch("fal_client.submit", return_value=handle) as submit:
        out = await call_fal("fal-ai/x", {"prompt": "hi"}, with_logs=False)
    assert submit.call_count == 1
    assert out == {"images": [{"url": "https://x/1.png"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [422, 429, 500, 503])
async def test_failure_after_acceptance_never_resubmits(status):
    """Once fal accepts the job it is billable — including on a 5xx or 429."""
    handle = _handle(exc=_fal_error(status, "boom"))
    with patch("fal_client.submit", return_value=handle) as submit:
        with pytest.raises(ValueError):
            await call_fal("fal-ai/x", {"prompt": "hi"}, with_logs=False)
    assert submit.call_count == 1


@pytest.mark.asyncio
async def test_timeout_after_acceptance_never_resubmits():
    """A timeout means the job most likely RAN and was billed. Never buy it twice."""
    handle = _handle(exc=httpx.ReadTimeout("timed out"))
    with patch("fal_client.submit", return_value=handle) as submit:
        with pytest.raises(ValueError):
            await call_fal("fal-ai/x", {"prompt": "hi"}, with_logs=False)
    assert submit.call_count == 1


@pytest.mark.asyncio
async def test_submit_failure_is_not_retried_here():
    """fal_client retries the submit POST itself; we must not stack another loop."""
    with patch(
        "fal_client.submit", side_effect=_fal_error(500, "bad gateway")
    ) as submit:
        with pytest.raises(ValueError):
            await call_fal("fal-ai/x", {"prompt": "hi"}, with_logs=False)
    assert submit.call_count == 1


# ---------------------------------------------------------------------------
# classification by real status, not by substring
# ---------------------------------------------------------------------------


def test_422_carrying_5xx_digits_is_not_a_server_error():
    """The exact prod string that the old predicate re-billed three times."""
    detail = (
        "[{'loc': ['body', 'image_url'], 'msg': 'Image dimensions are too small. "
        "Minimum dimensions are 240x240 pixels.', 'type': 'image_too_small', "
        "'url': 'https://docs.fal.ai/errors#image_too_small'}]"
    )
    msg = _format_error_for_user(_fal_error(422, detail))
    assert msg.startswith("FAL rejected the request (422)")
    assert "server error" not in msg.lower()
    assert "240x240" in msg  # the actionable detail survives


def test_real_5xx_is_reported_as_a_server_error():
    msg = _format_error_for_user(_fal_error(503, "Service Unavailable"))
    assert "FAL server error (503)" in msg


def test_status_beats_a_message_that_merely_mentions_429():
    msg = _format_error_for_user(_fal_error(400, "seed 429000 is out of range"))
    assert msg.startswith("FAL rejected the request (400)")
    assert "rate limit" not in msg.lower()
