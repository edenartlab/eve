import os
import asyncio
import fal_client
from typing import Dict, List, Any
from pydantic import Field
import logging

from .. import utils
from ..tool import Tool, tool_context
from ..task import Task, Creation

logger = logging.getLogger(__name__)


@tool_context("fal")
class FalTool(Tool):
    fal_endpoint: str
    with_logs: bool = Field(
        default=True, description="Whether to include logs in the response"
    )
    # output_handler: str = "normal" # Removed, logic now uses output_schema

    @Tool.handle_run
    async def async_run(
        self,
        args: Dict,
        user_id: str = None,
        agent_id: str = None,
        session_id: str = None,
    ):
        check_fal_api_token()
        args = self._format_args_for_fal(args)

        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    logger.info(log["message"])

        result = fal_client.subscribe(
            self.fal_endpoint,
            arguments=args,
            with_logs=self.with_logs,
            on_queue_update=on_queue_update if self.with_logs else None,
        )

        result = utils.upload_result({"output": result})
        return result

    @Tool.handle_start_task
    async def async_start_task(self, task: Task, webhook: bool = True):
        check_fal_api_token()
        args = self.prepare_args(task.args)
        args = self._format_args_for_fal(args)

        # Use webhook if provided
        webhook_url = get_webhook_url() if webhook else None
        if webhook_url:
            args["webhook_url"] = webhook_url

        # Submit the request to FAL queue and get the request_id
        handler = fal_client.submit(self.fal_endpoint, arguments=args)

        return handler.request_id

    @Tool.handle_wait
    async def async_wait(self, task: Task):
        check_fal_api_token()
        request_id = task.handler_id

        last_print_time = 0  # Initialize timer

        while True:
            status = fal_client.status(
                self.fal_endpoint, request_id, with_logs=self.with_logs
            )

            if status.status == "FAILED":
                task.update(status="failed", error=status.error)
                task.refund_manna()
                return {"status": "failed", "error": status.error}

            elif status.status == "CANCELED":
                task.update(status="cancelled")
                task.refund_manna()
                return {"status": "cancelled"}

            elif status.status == "PROCESSING":
                task.status = "running"
                task.save()
                # Print running status only every 2 seconds
                current_time = asyncio.get_event_loop().time()
                if current_time - last_print_time >= 2.0:
                    last_print_time = current_time

            elif status.status == "COMPLETED":
                result = fal_client.result(self.fal_endpoint, request_id)
                processed_result = self._process_result(result, task)
                task.status = "completed"
                task.result = processed_result
                task.save()
                return {"status": "completed", "result": processed_result}

            await asyncio.sleep(0.5)  # Poll every 0.5 seconds

    @Tool.handle_cancel
    async def async_cancel(self, task: Task):
        # FAL doesn't have a direct cancel API in their client yet
        # This is a placeholder for when they add that functionality
        pass

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
        """Process the result from FAL API using the output_schema as a direct path."""

        # output_schema is now expected to be a list of keys (the path)
        output_path = self.output_schema

        if not isinstance(output_path, list):
            logger.error(
                f"Invalid output_schema format for tool {self.name}. Expected list, got {type(output_path)}."
            )
            # Return raw result if schema is misconfigured
            return {"output": result}

        # logger.debug(f"Using output path from schema: {output_path}") # Optional: uncomment for debugging
        raw_output_value = self._get_value_by_path(result, output_path)

        if raw_output_value is None:
            logger.error(
                f"Could not extract value using path {output_path} from result for tool {self.name}"
            )
            output_urls = []
        elif isinstance(raw_output_value, str):
            output_urls = [raw_output_value]  # Single URL
            logger.info(f"Extracted single URL: {output_urls}")
        elif isinstance(raw_output_value, list):
            # If the path points to a list, assume it's a list of output URLs
            output_urls = [item for item in raw_output_value if isinstance(item, str)]
            if not output_urls:
                logger.warning(
                    f"Output path {output_path} points to a list, but couldn't extract string URLs directly."
                )
                # Attempt basic extraction if list contains dicts with 'url' key - simplistic fallback
                output_urls = [
                    item["url"]
                    for item in raw_output_value
                    if isinstance(item, dict) and "url" in item
                ]
        else:
            logger.warning(
                f"Unexpected type for output value at path {output_path}: {type(raw_output_value)}"
            )
            output_urls = []

        if not output_urls:
            logger.error(
                f"No output URLs extracted using path {output_path} for tool {self.name}, task {task.id}. Returning raw result."
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
                processed_outputs.append(uploaded_data)
            except Exception as e:
                logger.error(f"Failed to upload result URL {url}: {e}")
                continue  # Skip this output if upload fails

        if not processed_outputs:
            logger.error(f"No processable outputs found or uploaded for task {task.id}")
            # Return raw result if processing/uploading failed
            return {"output": result}

        # Structure for database: list of outputs, each potentially with multiple results (though usually 1 for FAL)
        # We create one 'result' entry containing all processed outputs.
        final_result_structure = [{"output": processed_outputs}]

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
                creation = Creation(
                    user=task.user,
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
    if not os.getenv("FAL_API_KEY"):
        raise Exception("FAL_API_KEY is not set")
