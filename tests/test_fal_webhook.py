"""fal webhook path: signature verification + task finalization.

The verification recipe (message = request_id\nuser_id\ntimestamp\n
sha256_hex(body), ED25519 against JWKS keys, +/-5 min tolerance) matches fal's
documented scheme; these tests prove our implementation of it with a real
keypair, and prove fal_update_task's status mapping and idempotency.
"""

import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eve.tools import fal_tool
from eve.tools.fal_tool import fal_update_task, verify_fal_webhook


# ---------------------------------------------------------------------------
# signature verification
# ---------------------------------------------------------------------------

def _signed_request(body: bytes, ts_offset: int = 0):
    key = Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes_raw()
    request_id = "req-123"
    user_id = "user-456"
    timestamp = str(int(time.time()) + ts_offset)
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


# ---------------------------------------------------------------------------
# fal_update_task
# ---------------------------------------------------------------------------

def _task(status="pending"):
    task = MagicMock()
    task.status = status
    task.args = {"prompt": "a cat"}
    task.performance = {}
    task.reload = MagicMock()
    return task


def test_error_payload_fails_task_and_refunds():
    task = _task()
    out = fal_update_task(task, "ERROR", None, "Invalid status code: 422")
    assert out["status"] == "failed"
    task.update.assert_called_once()
    assert task.update.call_args.kwargs["status"] == "failed"
    task.refund_manna.assert_called_once()


def test_idempotent_when_terminal():
    """fal retries webhook deliveries; a second delivery must no-op."""
    for terminal in ("completed", "failed", "cancelled"):
        task = _task(status=terminal)
        out = fal_update_task(task, "OK", {"video": {"url": "http://x"}}, None)
        assert out["status"] == terminal
        task.update.assert_not_called()
        task.refund_manna.assert_not_called()


def test_ok_payload_completes_with_creation():
    task = _task()
    tool = MagicMock()
    tool._extract_urls_from_fal_result.return_value = ["https://fal/video.mp4"]
    uploaded = [{"filename": "abc.mp4", "mediaAttributes": {"duration": 3}}]
    with patch.object(fal_tool.Tool, "load", return_value=tool), \
         patch.object(fal_tool.utils, "upload_result", return_value=uploaded), \
         patch.object(fal_tool, "Creation") as MockCreation:
        MockCreation.return_value.id = "creation-id"
        out = fal_update_task(task, "OK", {"video": {"url": "https://fal/video.mp4"}}, None)
    assert out["status"] == "completed"
    assert task.status == "completed"
    assert task.result[0]["output"][0]["creation"] == "creation-id"
    MockCreation.return_value.save.assert_called_once()
    task.save.assert_called()
    task.refund_manna.assert_not_called()


def test_empty_result_fails_and_refunds():
    task = _task()
    tool = MagicMock()
    tool._extract_urls_from_fal_result.return_value = []
    with patch.object(fal_tool.Tool, "load", return_value=tool):
        out = fal_update_task(task, "OK", {}, None)
    assert out["status"] == "failed"
    task.refund_manna.assert_called_once()
