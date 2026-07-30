import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, List

import fal_client
from pydantic import Field

from .. import utils
from ..task import Creation, Task

# from ..agent.session.models import Session
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
        fal_update_task, which is idempotent, so racing the webhook is safe.
        """
        check_fal_api_token()
        request_id = task.handler_id

        while True:
            task.reload()
            if task.status in ("completed", "failed", "cancelled"):
                return task.model_dump(include={"status", "error", "result"})

            try:
                status = await asyncio.to_thread(
                    fal_client.status,
                    self.fal_endpoint,
                    request_id,
                    with_logs=False,
                )
            except ValueError as e:
                # fal-client raises ValueError for FAILED / CANCELED statuses
                error_msg = str(e)
                if "CANCELED" in error_msg:
                    task.update(status="cancelled")
                    task.refund_manna()
                    return {"status": "cancelled"}
                if "FAILED" in error_msg:
                    return fal_update_task(task, "ERROR", None, error_msg)
                raise

            if isinstance(status, fal_client.InProgress):
                if task.status != "running":
                    task.update(status="running")
            elif isinstance(status, fal_client.Completed):
                result = await asyncio.to_thread(
                    fal_client.result, self.fal_endpoint, request_id
                )
                return fal_update_task(task, "OK", result, None)

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

    def _get_value_by_path(self, data: Any, path: List[str]) -> Any:
        """Retrieve value from nested data using a list of keys (path)."""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif (
                isinstance(current, list) and key.isdigit() and int(key) < len(current)
            ):
                # This part might need refinement if arrays are complex
                # For now, assumes path doesn't navigate *into* array elements using numeric indices
                # The path finding logic returns the path *to* the array itself
                current = current[int(key)]
            else:
                return None  # Path not found or invalid structure
        return current

    def _process_result(self, result, task):
        """Process the result from FAL API by extracting URLs from common response structures."""

        # Extract URLs using common FAL response patterns
        output_urls = self._extract_urls_from_fal_result(result)

        if not output_urls:
            logger.error(
                f"No output URLs extracted from FAL result for tool {self.name}, task {task.id}. Returning raw result."
            )
            return {"output": result}  # Return raw result if extraction fails

        processed_outputs = []
        # Upload each extracted URL
        for url in output_urls:
            try:
                # upload_result expects a dict structure. We wrap the single URL.
                # It will upload the URL and return metadata.
                logger.info(f"Attempting to upload FAL URL to Eden: {url}")
                uploaded_data = utils.upload_result(
                    {"output": url},  # Pass the URL directly for uploading
                    save_thumbnails=True,
                    save_blurhash=True,
                )
                # Print the result from upload_result to see the structure and final URL
                logger.info(f"Uploaded FAL URL {url} to Eden: {uploaded_data}")
                # Unwrap the "output" key since upload_result preserves the dict structure
                processed_outputs.append(uploaded_data.get("output", uploaded_data))
            except Exception as e:
                logger.error(f"Failed to upload result URL {url}: {e}")
                continue  # Skip this output if upload fails

        if not processed_outputs:
            logger.error(f"No processable outputs found or uploaded for task {task.id}")
            # Return raw result if processing/uploading failed
            return {"output": result}

        # Structure for database: match replicate format - each output gets its own result entry
        # This matches: result = [{"output": [out]} for out in output]
        final_result_structure = [{"output": [out]} for out in processed_outputs]

        # Create creation object(s) based on processed outputs
        for r, res_item in enumerate(final_result_structure):
            for o, output_data in enumerate(res_item["output"]):
                # Ensure output_data is a dict, as expected by Creation logic
                if not isinstance(output_data, dict):
                    logger.warning(
                        f"Skipping creation object for non-dict output: {output_data}"
                    )
                    continue

                name = task.args.get(
                    "prompt", task.args.get("text_input", "")
                )  # Try getting prompt/text_input

                # creation_agent = task.agent
                # session = Session.from_mongo(task.session)
                # if session.parent_session:
                #     parent_session = Session.from_mongo(session.parent_session)
                #     creation_agent = parent_session.agent

                creation = Creation(
                    user=task.user,
                    # agent=creation_agent,
                    agent=task.agent,
                    task=task.id,
                    tool=task.tool,
                    filename=output_data.get("filename"),
                    mediaAttributes=output_data.get("mediaAttributes", {}),
                    name=name,
                    public=task.public,
                )
                creation.save()
                final_result_structure[r]["output"][o]["creation"] = creation.id

        return final_result_structure  # Return the structured result with creation IDs

    # Override the base class method to add debugging before returning
    async def wait(self, task: Task):
        result_data = await self.async_wait(task)
        return result_data


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

def fal_update_task(task: Task, status: str, payload, error):
    """Finalize a fal task from a webhook payload ({status: OK|ERROR, payload})
    or from a status poll (async_wait / the periodic sweep).

    IDEMPOTENT via the terminal-status guard: fal retries webhook deliveries,
    and the sweep or a blocking waiter can race a webhook — only the first
    finalizer runs; the rest no-op. Mirrors replicate_update_task.
    """
    task.reload()
    if task.status in ("completed", "failed", "cancelled"):
        return {"status": task.status}

    if status != "OK":
        error_msg = str(error or "fal returned an error")[:500]
        task.update(status="failed", error=error_msg)
        task.refund_manna()
        return {"status": "failed", "error": error_msg}

    tool = Tool.load(task.tool)
    urls = tool._extract_urls_from_fal_result(payload or {})
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
    task.performance["runTime"] = run_time
    task.status = "completed"
    task.result = result
    task.save()
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
_fal_jwks_cache = {"keys": None, "fetched_at": 0.0}


def _get_fal_public_keys():
    import base64
    import time

    import httpx

    now = time.time()
    if (
        _fal_jwks_cache["keys"]
        and now - _fal_jwks_cache["fetched_at"] < _FAL_JWKS_TTL
    ):
        return _fal_jwks_cache["keys"]

    resp = httpx.get(FAL_JWKS_URL, timeout=10)
    resp.raise_for_status()
    keys = []
    for jwk in resp.json().get("keys", []):
        x = jwk.get("x")
        if not x:
            continue
        pad = "=" * (-len(x) % 4)
        keys.append(base64.urlsafe_b64decode(x + pad))
    if keys:
        _fal_jwks_cache.update(keys=keys, fetched_at=now)
    return keys


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
    except ValueError:
        raise ValueError("invalid timestamp header")
    if skew > 300:
        raise ValueError(f"timestamp outside tolerance ({skew:.0f}s)")

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
