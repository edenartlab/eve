"""fal webhook path: signature verification + task finalization.

The verification recipe (message = request_id\nuser_id\ntimestamp\n
sha256_hex(body), ED25519 against JWKS keys, +/-5 min tolerance) matches fal's
documented scheme; these tests prove our implementation of it with a real
keypair, and prove fal_update_task's claim/status mapping and idempotency.
"""

import hashlib
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eve.tools import fal_tool
from eve.tools.fal_tool import fal_update_task, verify_fal_webhook

# ---------------------------------------------------------------------------
# signature verification
# ---------------------------------------------------------------------------


def _signed_request(body: bytes, ts_offset: int = 0, ts_raw: str = None,
                    user_id: str = "user-456"):
    key = Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes_raw()
    request_id = "req-123"
    timestamp = ts_raw if ts_raw is not None else str(int(time.time()) + ts_offset)
    message = "\n".join(
        [request_id, user_id, timestamp, hashlib.sha256(body).hexdigest()]
    ).encode()
    signature = key.sign(message).hex()
    headers = {
        "X-Fal-Webhook-Request-Id": request_id,
        "X-Fal-Webhook-User-Id": user_id,
        "X-Fal-Webhook-Timestamp": timestamp,
        "X-Fal-Webhook-Signature": signature,
    }
    return headers, pub


def test_signature_verifies():
    body = b'{"request_id": "req-123", "status": "OK"}'
    headers, pub = _signed_request(body)
    with patch.object(fal_tool, "_get_fal_public_keys", return_value=[pub]):
        verify_fal_webhook(body, headers)  # should not raise


def test_tampered_body_rejected():
    body = b'{"status": "OK"}'
    headers, pub = _signed_request(body)
    with patch.object(fal_tool, "_get_fal_public_keys", return_value=[pub]):
        with pytest.raises(ValueError, match="did not verify"):
            verify_fal_webhook(b'{"status": "HACKED"}', headers)


def test_stale_timestamp_rejected():
    body = b"{}"
    headers, pub = _signed_request(body, ts_offset=-600)  # 10 min old
    with patch.object(fal_tool, "_get_fal_public_keys", return_value=[pub]):
        with pytest.raises(ValueError, match="tolerance"):
            verify_fal_webhook(body, headers)


def test_absurd_timestamp_rejected_not_500():
    """A 400-digit timestamp must raise the controlled ValueError, not
    OverflowError escaping to the route."""
    body = b"{}"
    headers, pub = _signed_request(body, ts_raw="9" * 400)
    with patch.object(fal_tool, "_get_fal_public_keys", return_value=[pub]):
        with pytest.raises(ValueError, match="invalid timestamp"):
            verify_fal_webhook(body, headers)


def test_wrong_key_rejected():
    body = b"{}"
    headers, _ = _signed_request(body)
    other = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    with patch.object(fal_tool, "_get_fal_public_keys", return_value=[other]):
        with pytest.raises(ValueError, match="did not verify"):
            verify_fal_webhook(body, headers)


def test_missing_header_rejected():
    body = b"{}"
    headers, pub = _signed_request(body)
    del headers["X-Fal-Webhook-Signature"]
    with patch.object(fal_tool, "_get_fal_public_keys", return_value=[pub]):
        with pytest.raises(ValueError, match="missing header"):
            verify_fal_webhook(body, headers)


def test_user_id_pinning(monkeypatch):
    """With FAL_WEBHOOK_USER_ID set, deliveries signed for OTHER fal accounts
    are rejected before any JWKS fetch; our own pass."""
    body = b"{}"
    headers, pub = _signed_request(body, user_id="attacker-account")
    monkeypatch.setenv("FAL_WEBHOOK_USER_ID", "our-account")
    fetched = MagicMock(side_effect=AssertionError("JWKS must not be fetched"))
    with patch.object(fal_tool, "_get_fal_public_keys", fetched):
        with pytest.raises(ValueError, match="does not match"):
            verify_fal_webhook(body, headers)

    headers, pub = _signed_request(body, user_id="our-account")
    with patch.object(fal_tool, "_get_fal_public_keys", return_value=[pub]):
        verify_fal_webhook(body, headers)  # should not raise


def test_jwks_stale_served_on_fetch_failure(monkeypatch):
    """An expired-but-present key set is served when the JWKS fetch fails, so a
    fal blip doesn't drop legitimate deliveries."""
    import httpx

    stale_keys = [b"k" * 32]
    monkeypatch.setattr(
        fal_tool,
        "_fal_jwks_cache",
        {"keys": stale_keys, "fetched_at": 0.0, "failed_at": 0.0},
    )
    with patch.object(httpx, "get", side_effect=httpx.ConnectError("down")):
        assert fal_tool._get_fal_public_keys() == stale_keys
    # and the failure is negative-cached
    assert fal_tool._fal_jwks_cache["failed_at"] > 0


# ---------------------------------------------------------------------------
# fal_update_task
# ---------------------------------------------------------------------------


def _task(status="pending"):
    task = MagicMock()
    task.status = status
    task.args = {"prompt": "a cat"}
    task.performance = {}
    task.handler_id = "req-1"
    task.createdAt = datetime.now(timezone.utc) - timedelta(seconds=30)
    task.reload = MagicMock()
    return task


def _collection(claim=True, finish=True):
    """Mock tasks3: first find_one_and_update = claim, second = terminal write."""
    col = MagicMock()
    col.find_one_and_update.side_effect = [
        {"_id": "t"} if claim else None,
        {"_id": "t"} if finish else None,
    ]
    return col


def test_error_payload_fails_task_and_refunds():
    task = _task()
    with patch.object(fal_tool, "get_collection", return_value=_collection()):
        out = fal_update_task(task, "ERROR", None, "Invalid status code: 422")
    assert out["status"] == "failed"
    assert task.update.call_args.kwargs["status"] == "failed"
    task.refund_manna.assert_called_once()


def test_claim_lost_is_noop():
    """fal retries deliveries and pollers race the webhook; whoever loses the
    atomic claim must do nothing."""
    task = _task()
    with patch.object(fal_tool, "get_collection", return_value=_collection(claim=False)):
        out = fal_update_task(task, "OK", {"video": {"url": "http://x"}}, None)
    assert out["status"] == task.status
    task.update.assert_not_called()
    task.refund_manna.assert_not_called()


def test_ok_payload_completes_with_creation():
    task = _task()
    tool = MagicMock()
    tool._extract_urls_from_fal_result.return_value = ["https://fal/video.mp4"]
    uploaded = [{"filename": "abc.mp4", "mediaAttributes": {"duration": 3}}]
    col = _collection()
    with patch.object(fal_tool, "get_collection", return_value=col), \
         patch.object(fal_tool.Tool, "load", return_value=tool), \
         patch.object(fal_tool.utils, "upload_result", return_value=uploaded), \
         patch.object(fal_tool, "Creation") as MockCreation:
        MockCreation.return_value.id = "creation-id"
        out = fal_update_task(task, "OK", {"video": {"url": "https://fal/video.mp4"}}, None)
    assert out["status"] == "completed"
    assert task.result[0]["output"][0]["creation"] == "creation-id"
    MockCreation.return_value.save.assert_called_once()
    # terminal write is conditional, not a full-doc save
    terminal_set = col.find_one_and_update.call_args_list[1].args[1]["$set"]
    assert terminal_set["status"] == "completed"
    assert terminal_set["performance"]["runTime"] > 0
    task.refund_manna.assert_not_called()


def test_cancelled_mid_finalization_not_stomped():
    """User cancels (and is refunded) while the upload runs: the completed
    result must be DISCARDED, not written over the cancellation."""
    task = _task()
    tool = MagicMock()
    tool._extract_urls_from_fal_result.return_value = ["https://fal/v.mp4"]
    uploaded = [{"filename": "a.mp4", "mediaAttributes": {}}]
    with patch.object(fal_tool, "get_collection",
                      return_value=_collection(finish=False)), \
         patch.object(fal_tool.Tool, "load", return_value=tool), \
         patch.object(fal_tool.utils, "upload_result", return_value=uploaded), \
         patch.object(fal_tool, "Creation") as MockCreation:
        MockCreation.return_value.id = "c"
        task.reload = MagicMock(side_effect=lambda: setattr(task, "status", "cancelled"))
        out = fal_update_task(task, "OK", {"video": {"url": "https://fal/v.mp4"}}, None)
    assert out["status"] == "cancelled"


def test_ok_with_null_payload_falls_back_to_result_fetch():
    """fal sends status OK + payload null when the result wasn't deliverable
    inline; the run SUCCEEDED and must not be failed+refunded."""
    task = _task()
    tool = MagicMock()
    tool.fal_endpoint = "fal-ai/some/endpoint"
    tool._extract_urls_from_fal_result.side_effect = [
        [],  # from the null payload
        ["https://fal/fetched.mp4"],  # from the queue result fetch
    ]
    uploaded = [{"filename": "f.mp4", "mediaAttributes": {}}]
    with patch.object(fal_tool, "get_collection", return_value=_collection()), \
         patch.object(fal_tool.Tool, "load", return_value=tool), \
         patch.object(fal_tool.fal_client, "result", return_value={"video": {}}) as res, \
         patch.object(fal_tool.utils, "upload_result", return_value=uploaded), \
         patch.object(fal_tool, "Creation") as MockCreation:
        MockCreation.return_value.id = "c"
        out = fal_update_task(task, "OK", None, None)
    res.assert_called_once_with("fal-ai/some/endpoint", "req-1")
    assert out["status"] == "completed"
    task.refund_manna.assert_not_called()


def test_empty_result_fails_and_refunds():
    task = _task()
    tool = MagicMock()
    tool.fal_endpoint = "fal-ai/some/endpoint"
    tool._extract_urls_from_fal_result.return_value = []
    with patch.object(fal_tool, "get_collection", return_value=_collection()), \
         patch.object(fal_tool.Tool, "load", return_value=tool), \
         patch.object(fal_tool.fal_client, "result", side_effect=Exception("422")):
        out = fal_update_task(task, "OK", {}, None)
    assert out["status"] == "failed"
    task.refund_manna.assert_called_once()
