import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, List

import fal_client
from pydantic import Field

from .. import utils
from ..mongo import get_collection
from ..task import Creation, Task
from ..tool import Tool, ToolContext, tool_context

logger = logging.getLogger(__name__)


def is_valid_url(value: Any) -> bool:
    """Check if a value is a valid URL string."""
    if not isinstance(value, str):
        return False
    if len(value) < 10:
        return False
    # Basic URL pattern: starts with http:// or https://
    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"[a-zA-Z0-9]"  # at least one alphanumeric char after protocol
    )
    return bool(url_pattern.match(value))


# Retry configuration
MAX_RETRIES = 3
INITIAL_DELAY = 1.0


def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    error_str = str(error).lower()

    # Rate limit errors (429)
    if (
        "429" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
    ):
        return True

    # Server errors (5xx)
    if any(f"{code}" in error_str for code in range(500, 600)):
        return True

    # Network/timeout errors
    if any(
        term in error_str
        for term in ["timeout", "connection", "network", "unavailable"]
    ):
        return True

    return False


def _format_error_for_user(error: Exception) -> str:
    """Format error message for user-friendly display."""
    error_str = str(error).lower()

    if "429" in error_str or "rate limit" in error_str:
        return "Rate limit reached for this model. Please try again later or use a different model."

    if (
        "401" in error_str
        or "unauthorized" in error_str
        or "authentication" in error_str
    ):
        return "Authentication error with FAL API. Please check API credentials."

    if "403" in error_str or "forbidden" in error_str:
        return "Access denied to FAL API. Please check API permissions."

    if any(f"{code}" in error_str for code in range(500, 600)):
        return "FAL API server error. Please try again later."

    if "timeout" in error_str:
        return "Request timed out. Please try again."

    # Return original error for unknown cases
    return str(error)


@tool_context("fal")
class FalTool(Tool):
    fal_endpoint: str
    with_logs: bool = Field(
        default=True, description="Whether to include logs in the response"
    )

    async def _call_with_retry(self, endpoint: str, args: dict) -> dict:
        """
        Call FAL API with exponential backoff retry logic.

        Args:
            endpoint: The FAL API endpoint
            args: Arguments for the API call

        Returns:
            The API response dict

        Raises:
            Exception: With user-friendly error message
        """
        delay = INITIAL_DELAY
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:

                def on_queue_update(update):
                    if isinstance(update, fal_client.InProgress):
                        for log in update.logs:
                            logger.info(log["message"])

                result = await asyncio.to_thread(
                    fal_client.subscribe,
                    endpoint,
                    arguments=args,
                    with_logs=self.with_logs,
                    on_queue_update=on_queue_update if self.with_logs else None,
                )
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"FAL API call failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}"
                )

                # Check if error is retryable and we have retries left
                if _is_retryable_error(e) and attempt < MAX_RETRIES:
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue

                # Non-retryable or max retries reached
                raise Exception(_format_error_for_user(e))

        # Should not reach here, but just in case
        raise Exception(_format_error_for_user(last_error))

    @Tool.handle_run
    async def async_run(self, context: ToolContext):
        check_fal_api_token()
        args = await asyncio.to_thread(self._format_args_for_fal, context.args)

        result = await self._call_with_retry(self.fal_endpoint, args)

        # Extract URLs from common FAL response structures (e.g., {"images": [{"url": "..."}]})
        output_urls = self._extract_urls_from_fal_result(result)

        if output_urls:
            # Upload each URL and return normalized structure
            processed_outputs = []
            for url in output_urls:
                try:
                    logger.info(f"Uploading FAL URL to Eden: {url}")
                    uploaded_data = utils.upload_result(
                        {"output": url},
                        save_thumbnails=True,
                        save_blurhash=True,
                    )
                    processed_outputs.append(uploaded_data.get("output", uploaded_data))
                except Exception as e:
                    logger.error(f"Failed to upload result URL {url}: {e}")
                    continue

            if processed_outputs:
                # Return normalized structure: {"output": [{"url": ...}, ...]}
                return {"output": processed_outputs}

        # Fallback: return raw result wrapped with upload_result
        result = utils.upload_result({"output": result})
        return result

    def _extract_urls_from_fal_result(self, result: dict) -> List[str]:
        """Extract URLs from common FAL API response structures."""
        output_urls = []

        if not isinstance(result, dict):
            return output_urls

        # Check for "images" array (common in image generation endpoints)
        if "images" in result and isinstance(result["images"], list):
            for item in result["images"]:
                if isinstance(item, dict) and "url" in item:
                    url_value = item["url"]
                    if is_valid_url(url_value):
                        output_urls.append(url_value)

        # Check for "video" field (common in video generation endpoints)
        elif "video" in result and isinstance(result["video"], dict):
            if "url" in result["video"]:
                url_value = result["video"]["url"]
                if is_valid_url(url_value):
                    output_urls.append(url_value)

        # Check for direct "url" field
        elif "url" in result:
            url_value = result["url"]
            if is_valid_url(url_value):
                output_urls.append(url_value)

        # Check for "output" field with URL
        elif "output" in result:
            output = result["output"]
            if is_valid_url(output):
                output_urls.append(output)
            elif isinstance(output, list):
                for item in output:
                    if is_valid_url(item):
                        output_urls.append(item)
                    elif isinstance(item, dict) and "url" in item:
                        url_value = item["url"]
                        if is_valid_url(url_value):
                            output_urls.append(url_value)

        return output_urls

    @Tool.handle_start_task
    async def async_start_task(self, task: Task):
        """Submit to fal's queue with a completion webhook; return immediately.

        The work runs on fal's servers — nothing on our side needs to stay
        alive (no held Modal container, no polling). Completion arrives at
        POST /update-fal (eve/api: handle_fal_webhook -> fal_update_task), and
        a periodic sweep (sweep_pending_fal_tasks_fn) re-polls any fal task
        still pending after a few minutes in case a webhook delivery is missed.

        NOTE the historical bug this replaces: webhook_url MUST be the keyword
        argument of fal_client.submit (it becomes the ?fal_webhook= query
        param). It used to be buried inside `arguments`, where fal silently
        ignores unknown keys — so no webhook was ever registered, nothing
        polled, and every direct fal task hung until the 3h watchdog killed it
        as "Timed out".
        """
        check_fal_api_token()
        args = self.prepare_args(task.args)
        args = await asyncio.to_thread(self._format_args_for_fal, args)

        handler = await asyncio.to_thread(
            fal_client.submit,
            self.fal_endpoint,
            arguments=args,
            webhook_url=get_webhook_url(),
        )
        return handler.request_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        """Poll fal until the request finishes (blocking callers only).

        The primary completion path is the webhook; this exists for callers
        that need to block on a task. Finalization goes through
        fal_update_task, which claims the task atomically, so racing the
        webhook is safe.

        NOTE on failure detection: fal's queue status enum is only
        IN_QUEUE / IN_PROGRESS / COMPLETED — a FAILED job also reports
        COMPLETED, and the failure only surfaces when the result fetch raises
        (4xx). There is no FAILED/CANCELED status to branch on (verified
        against fal_client 0.5.9 _parse_status).
        """
        check_fal_api_token()
        request_id = task.handler_id

        while True:
            task.reload()
            if task.status in TERMINAL_STATES:
                return task.model_dump(include={"status", "error", "result"})

            try:
                status = await asyncio.to_thread(
                    fal_client.status,
                    self.fal_endpoint,
                    request_id,
                    with_logs=False,
                )
            except Exception as e:
                # Transient status-endpoint failure (or an unknown status
                # string) — keep polling; the watchdog bounds the loop.
                logger.warning(f"fal status poll failed for {request_id}: {e}")
                await asyncio.sleep(2)
                continue

            if isinstance(status, fal_client.InProgress):
                if task.status != "running":
                    task.update(status="running")
            elif isinstance(status, fal_client.Completed):
                try:
                    result = await asyncio.to_thread(
                        fal_client.result, self.fal_endpoint, request_id
                    )
                except Exception as e:
                    code = _falclient_status_code(e)
                    if code is not None and 400 <= code < 500:
                        # The JOB failed (fal returns 4xx on the result of a
                        # failed run); finalize as an error + refund.
                        return await asyncio.to_thread(
                            fal_update_task, task, "ERROR", None, str(e)
                        )
                    # Transient (5xx / network): retry the loop.
                    logger.warning(f"fal result fetch failed for {request_id}: {e}")
                    await asyncio.sleep(2)
                    continue
                return await asyncio.to_thread(
                    fal_update_task, task, "OK", result, None
                )

            await asyncio.sleep(1)

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        """Cancel the fal queue request (handler_id is the fal request id)."""
        if not task.handler_id:
            return
        try:
            await asyncio.to_thread(
                fal_client.cancel, self.fal_endpoint, task.handler_id
            )
            logger.info(f"FAL cancel sent for {task.handler_id}")
        except Exception as e:
            logger.warning(f"FAL cancel failed for {task.handler_id}: {e}")

    def _format_args_for_fal(self, args: dict):
        """Format the arguments for FAL API call"""
        new_args = args.copy()
        new_args = {k: v for k, v in new_args.items() if v is not None}

        # Handle file uploads if needed
        for key, value in new_args.items():
            if isinstance(value, str) and os.path.isfile(value):
                new_args[key] = fal_client.upload_file(value)

        return new_args

def get_webhook_url():
    env = {
        "PROD": "api-prod",
        "STAGE": "api-stage",
        "WEB3-PROD": "api-web3-prod",
        "WEB3-STAGE": "api-web3-stage",
    }.get(os.getenv("DB"), "api-web3-stage")
    dev = (
        "-dev"
        if os.getenv("DB") in ["WEB3-STAGE", "STAGE"]
        and os.getenv("MODAL_SERVE") == "1"
        else ""
    )

    # /update is the Replicate webhook (replicate signature validation);
    # fal callbacks go to their own endpoint with fal's ED25519 scheme.
    webhook_url = f"https://edenartlab--{env}-fastapi-app{dev}.modal.run/update-fal"
    return webhook_url


def check_fal_api_token():
    if not os.getenv("FAL_KEY"):
        raise Exception("FAL_KEY is not set")


# ---------------------------------------------------------------------------
# Webhook-side finalization
# ---------------------------------------------------------------------------

TERMINAL_STATES = ("completed", "failed", "cancelled")

# How long one finalizer may hold the claim before another may take over
# (crash recovery: the sweep can re-finalize after the lease expires).
_FINALIZE_LEASE = timedelta(minutes=10)


def _falclient_status_code(error: Exception):
    """HTTP status from a FalClientError (raised `from httpx.HTTPStatusError`)."""
    cause = getattr(error, "__cause__", None)
    resp = getattr(cause, "response", None)
    code = getattr(resp, "status_code", None)
    return code if isinstance(code, int) else None


def fal_update_task(task: Task, status: str, payload, error):
    """Finalize a fal task from a webhook payload ({status: OK|ERROR, payload})
    or from a status poll (async_wait / the periodic sweep).

    Finalizers RACE each other by design — fal retries webhook deliveries (its
    delivery timeout is short relative to a video upload), a blocking
    async_wait polls at 1s, and the sweep re-polls stragglers. So the first
    step is an ATOMIC CLAIM on the task document (status not terminal AND no
    live finalize lease); losers no-op. A crashed finalizer's lease expires
    after _FINALIZE_LEASE, letting the sweep retry. Mirrors
    replicate_update_task for the actual output handling.
    """
    tasks = get_collection("tasks3")
    now = datetime.now(timezone.utc)
    claimed = tasks.find_one_and_update(
        {
            "_id": task.id,
            "status": {"$nin": list(TERMINAL_STATES)},
            "$or": [
                {"finalizing_at": None},
                {"finalizing_at": {"$lt": now - _FINALIZE_LEASE}},
            ],
        },
        {"$set": {"finalizing_at": now}},
    )
    if claimed is None:
        # Terminal already, or another finalizer holds the claim.
        task.reload()
        return {"status": task.status}

    if status != "OK":
        error_msg = str(error or "fal returned an error")[:500]
        task.update(status="failed", error=error_msg)
        task.refund_manna()
        return {"status": "failed", "error": error_msg}

    tool = Tool.load(task.tool)
    urls = tool._extract_urls_from_fal_result(payload or {})
    if not urls and task.handler_id:
        # fal delivers status "OK" with payload null (+ payload_error) when the
        # response wasn't serializable/deliverable inline — the run SUCCEEDED
        # and the docs say to fetch the result from the queue instead. Failing
        # here would refund a job fal charged us for.
        try:
            fetched = fal_client.result(tool.fal_endpoint, task.handler_id)
            urls = tool._extract_urls_from_fal_result(fetched or {})
        except Exception as e:
            logger.warning(f"fal result fallback fetch failed for {task.id}: {e}")
    if not urls:
        error_msg = f"fal returned no output: {str(payload)[:300]}"
        task.update(status="failed", error=error_msg)
        task.refund_manna()
        return {"status": "failed", "error": error_msg}

    output = utils.upload_result(urls, save_thumbnails=True, save_blurhash=True)
    result = [{"output": [out]} for out in output]

    for r, res in enumerate(result):
        for o, out in enumerate(res["output"]):
            creation = Creation(
                user=task.user,
                agent=task.agent,
                task=task.id,
                tool=task.tool,
                filename=out["filename"],
                mediaAttributes=out["mediaAttributes"],
                name=task.args.get("prompt"),
                public=task.public,
            )
            creation.save()
            result[r]["output"][o]["creation"] = creation.id

    run_time = (
        datetime.now(timezone.utc) - task.createdAt.replace(tzinfo=timezone.utc)
    ).total_seconds()
    if task.performance.get("waitTime"):
        run_time -= task.performance["waitTime"]
    performance = {**(task.performance or {}), "runTime": run_time}

    # Conditional terminal write (NOT a full-document save): if the user
    # cancelled during the upload window, handle_cancel already refunded —
    # stomping status back to completed would let them keep both the manna
    # and the output. Losing this write is the correct outcome then.
    finished = tasks.find_one_and_update(
        {"_id": task.id, "status": {"$nin": list(TERMINAL_STATES)}},
        {
            "$set": {
                "status": "completed",
                "result": result,
                "performance": performance,
            },
            "$unset": {"finalizing_at": ""},
        },
    )
    if finished is None:
        task.reload()
        logger.warning(
            f"fal task {task.id} reached terminal state ({task.status}) during "
            "finalization; completed result discarded"
        )
        return {"status": task.status}

    task.status = "completed"
    task.result = result
    task.performance = performance
    return {"status": "completed", "result": result}


# ---------------------------------------------------------------------------
# Webhook signature verification (fal's documented scheme)
#
# Headers: X-Fal-Webhook-Request-Id, X-Fal-Webhook-User-Id,
#          X-Fal-Webhook-Timestamp, X-Fal-Webhook-Signature (hex)
# Message: "\n".join(request_id, user_id, timestamp, sha256_hex(raw_body))
# Verify:  ED25519 against any key from fal's JWKS (keys[].x is a base64url
#          ED25519 public key); timestamp must be within +/-5 minutes.
# Verified against fal's docs and two independent SDK implementations.
# ---------------------------------------------------------------------------

FAL_JWKS_URL = "https://rest.alpha.fal.ai/.well-known/jwks.json"
_FAL_JWKS_TTL = 24 * 3600  # fal say keys may rotate; don't cache longer than 24h
_fal_jwks_cache = {"keys": None, "fetched_at": 0.0, "failed_at": 0.0}
_FAL_JWKS_NEGATIVE_TTL = 60  # after a failed fetch, don't re-fetch for this long


def _get_fal_public_keys():
    """fal's ED25519 public keys, cached.

    Resilience matters here because this endpoint is unauthenticated: an
    attacker flooding garbage requests must not be able to turn every request
    into an outbound JWKS fetch. Failures are negative-cached briefly, and a
    stale key set is served rather than failing verification outright (fal say
    keys can rotate, so we never cache success longer than _FAL_JWKS_TTL).
    """
    import base64
    import time

    import httpx

    now = time.time()
    cache = _fal_jwks_cache
    if cache["keys"] and now - cache["fetched_at"] < _FAL_JWKS_TTL:
        return cache["keys"]

    # Within the negative-cache window: serve stale keys if we have them,
    # otherwise fail fast without another outbound request.
    if now - cache["failed_at"] < _FAL_JWKS_NEGATIVE_TTL:
        if cache["keys"]:
            return cache["keys"]
        raise ValueError("fal JWKS unavailable (recent fetch failed)")

    try:
        resp = httpx.get(FAL_JWKS_URL, timeout=10)
        resp.raise_for_status()
        keys = []
        for jwk in resp.json().get("keys", []):
            x = jwk.get("x")
            if not x:
                continue
            pad = "=" * (-len(x) % 4)
            keys.append(base64.urlsafe_b64decode(x + pad))
        if not keys:
            raise ValueError("fal JWKS contained no usable keys")
        cache.update(keys=keys, fetched_at=now)
        return keys
    except Exception:
        cache["failed_at"] = now
        if cache["keys"]:
            # Expired-but-present keys beat dropping a legitimate delivery.
            return cache["keys"]
        raise


def verify_fal_webhook(body: bytes, headers) -> None:
    """Raise ValueError unless this is an authentic, fresh fal webhook."""
    import hashlib
    import time

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    def h(name):
        value = headers.get(name)
        if not value:
            raise ValueError(f"missing header {name}")
        return value

    request_id = h("X-Fal-Webhook-Request-Id")
    user_id = h("X-Fal-Webhook-User-Id")
    timestamp = h("X-Fal-Webhook-Timestamp")
    signature_hex = h("X-Fal-Webhook-Signature")

    try:
        skew = abs(time.time() - int(timestamp))
    except (ValueError, OverflowError):
        # int() raises ValueError on garbage; the subtraction can raise
        # OverflowError on an absurdly large integer. Both are forgeries.
        raise ValueError("invalid timestamp header")
    if skew > 300:
        raise ValueError(f"timestamp outside tolerance ({skew:.0f}s)")

    # A valid fal signature proves "fal signed this", not "this is OUR
    # webhook" — any fal customer can point their own job at our URL and fal
    # will happily sign the delivery. When FAL_WEBHOOK_USER_ID is set, bind
    # deliveries to our account (checked before the JWKS fetch so foreign
    # traffic can't trigger outbound requests).
    expected_user = os.getenv("FAL_WEBHOOK_USER_ID")
    if expected_user and user_id != expected_user:
        raise ValueError("webhook user id does not match this account")

    message = "\n".join(
        [request_id, user_id, timestamp, hashlib.sha256(body).hexdigest()]
    ).encode()
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        raise ValueError("signature is not valid hex")

    for key_bytes in _get_fal_public_keys():
        try:
            Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature, message)
            return
        except (InvalidSignature, ValueError):
            continue
    raise ValueError("signature did not verify against any fal public key")
