import asyncio
import logging
import os
import re
from typing import Any, List

import fal_client
import modal
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
        """Run the task to completion in a Modal container (blocking subscribe).

        This previously submitted to fal's queue with a webhook and returned
        immediately, but that path could never complete:
          1. `webhook_url` was passed INSIDE `arguments`, while it is a
             keyword-only parameter of fal_client.submit — fal ignores unknown
             input keys, so it never received a callback URL at all; and
          2. the only webhook endpoint (/update) validates Replicate signatures
             and routes to the Replicate handler, so a fal callback would be
             rejected anyway.
        Tasks therefore sat pending until the >3h stuck-task watchdog killed
        them as "Timed out" (every handler:fal tool, on every direct API/studio
        call). Running through Modal's run_task is the same model the modal and
        replicate tools use, and matches the eve/tools/fal/* tools that work.
        """
        check_fal_api_token()
        db = os.getenv("DB", "STAGE").upper()
        func = modal.Function.from_name(
            f"api-{db.lower()}", "run_task", environment_name="main"
        )
        job = await func.spawn.aio(task)
        return job.object_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        """Wait on the Modal job spawned by async_start_task.

        handler_id is now a Modal FunctionCall id (not a fal request id); the
        run_task container updates the task itself, so we just await it.
        """
        fc = modal.functions.FunctionCall.from_id(task.handler_id)
        await fc.get.aio()
        task.reload()
        return task.model_dump(include={"status", "error", "result"})

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        """Cancel the Modal job (handler_id is a Modal FunctionCall id)."""
        if not task.handler_id:
            return
        try:
            fc = modal.functions.FunctionCall.from_id(task.handler_id)
            await fc.cancel.aio()
            logger.info(f"Cancelled Modal job {task.handler_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel Modal job {task.handler_id}: {e}")

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

    webhook_url = f"https://edenartlab--{env}-fastapi-app{dev}.modal.run/update"
    return webhook_url


def check_fal_api_token():
    if not os.getenv("FAL_KEY"):
        raise Exception("FAL_KEY is not set")
