"""
Non-route functions for the API module.
These are Modal functions and helper utilities that are not FastAPI routes.
"""

import logging
import os
import time
import replicate
import sentry_sdk
from bson import ObjectId

from eve import eden_utils
from eve.task import task_handler_func, Task
from eve.tool import Tool
from eve.tools.tool_handlers import load_handler
from eve.tools.replicate_tool import replicate_update_task
from eve.api.runner_tasks import (
    cancel_stuck_tasks,
    generate_lora_thumbnails,
    rotate_agent_metadata,
)
from eve.api.helpers import busy_state_dict

logger = logging.getLogger(__name__)
db = os.getenv("DB", "STAGE").upper()


# Modal scheduled functions


async def cancel_stuck_tasks_fn():
    try:
        await cancel_stuck_tasks()
    except Exception as e:
        print(f"Error cancelling stuck tasks: {e}")
        sentry_sdk.capture_exception(e)


async def generate_lora_thumbnails_fn():
    try:
        await generate_lora_thumbnails()
    except Exception as e:
        print(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)


async def rotate_agent_metadata_fn():
    try:
        await rotate_agent_metadata()
    except Exception as e:
        print(f"Error generating lora thumbnails: {e}")
        sentry_sdk.capture_exception(e)


async def run_scheduled_triggers_fn():
    """Check for and run scheduled triggers every minute"""
    from datetime import datetime, timezone
    from eve.agent.session.models import Trigger
    import aiohttp

    try:
        # Find triggers that need to run
        current_time = datetime.now(timezone.utc)

        # Find active triggers where next_scheduled_run <= current time
        triggers = list(Trigger.find(
            {
                "status": "active",
                "deleted": {"$ne": True},
                "next_scheduled_run": {"$lte": current_time},
            }
        ))

        logger.info(f"Found {len(triggers)} triggers to run")

        for trigger in triggers:
            try:
                logger.info(f"Running trigger {trigger.trigger_id}")

                # Prepare the prompt session request
                request_data = {
                    "session_id": str(trigger.session) if trigger.session else None,
                    "user_id": str(trigger.user),
                    "actor_agent_ids": [str(trigger.agent)],
                    "message": {
                        "role": "system",
                        "content": f"""## Task

You have been given the following instructions. Do not ask for clarification, or stop until you have completed the task.

{trigger.instruction}

""",
                    },
                    "update_config": trigger.update_config,
                }

                # If no session, add creation args
                if not trigger.session:
                    request_data["creation_args"] = {
                        "owner_id": str(trigger.user),
                        "agents": [str(trigger.agent)],
                        "trigger": str(trigger.id),
                    }

                # Make async HTTP POST to prompt session endpoint
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{os.getenv('EDEN_API_URL')}/sessions/prompt",
                        json=request_data,
                        headers={
                            "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                            "Content-Type": "application/json",
                        },
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Failed to run trigger {trigger.trigger_id}: {error_text}"
                            )
                            continue

                        result = await response.json()
                        session_id = result.get("session_id")

                        # Update trigger with session if it was created
                        if not trigger.session and session_id:
                            trigger.session = ObjectId(session_id)

                # Update last_run_time and calculate next_scheduled_run
                trigger.last_run_time = current_time

                # Calculate next run time
                from eve.agent.session.triggers import calculate_next_scheduled_run

                next_run = calculate_next_scheduled_run(trigger.schedule)

                if next_run:
                    trigger.next_scheduled_run = next_run
                else:
                    # No next run (possibly due to end_date)
                    trigger.status = "finished"
                    logger.info(
                        f"Trigger {trigger.trigger_id} has no next run, marking as finished"
                    )

                trigger.save()

                # Handle posting instructions if present
                if trigger.posting_instructions:
                    await handle_trigger_posting(trigger, session_id)

            except Exception as e:
                logger.error(f"Error running trigger {trigger.trigger_id}: {str(e)}")
                sentry_sdk.capture_exception(e)
                continue

    except Exception as e:
        logger.error(f"Error in run_scheduled_triggers: {str(e)}")
        sentry_sdk.capture_exception(e)


async def handle_trigger_posting(trigger, session_id):
    """Handle posting instructions for a trigger"""
    import aiohttp

    posting_instructions = trigger.posting_instructions
    if not posting_instructions:
        return

    try:
        request_data = {
            "session_id": session_id,
            "user_id": str(trigger.user),
            "actor_agent_ids": [str(trigger.agent)],
            "message": {
                "role": "system",
                "content": f"""## Posting instructions
{posting_instructions.get('post_to', '')} channel {posting_instructions.get('channel_id', '')}

{posting_instructions.get('custom_instructions', '')}
""",
            },
            "update_config": trigger.update_config,
        }

        # Add custom tools based on platform
        platform = posting_instructions.get("post_to")
        if platform == "discord" and posting_instructions.get("channel_id"):
            request_data["custom_tools"] = {
                "discord_post": {
                    "parameters": {
                        "channel_id": {"default": posting_instructions["channel_id"]}
                    }
                }
            }
        elif platform == "telegram" and posting_instructions.get("channel_id"):
            request_data["custom_tools"] = {
                "telegram_post": {
                    "parameters": {
                        "channel_id": {"default": posting_instructions["channel_id"]}
                    }
                }
            }

        # Make async HTTP POST to prompt session endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('EDEN_API_URL')}/sessions/prompt",
                json=request_data,
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to run posting instructions for trigger {trigger.trigger_id}: {error_text}"
                    )

    except Exception as e:
        logger.error(
            f"Error handling posting instructions for trigger {trigger.trigger_id}: {str(e)}"
        )
        sentry_sdk.capture_exception(e)


# Modal task functions


async def run(tool_key: str, args: dict, user: str = None, agent: str = None):
    handler = load_handler(tool_key)
    result = await handler(args, user, agent)
    return eden_utils.upload_result(result)


@task_handler_func
async def run_task(tool_key: str, args: dict, user: str = None, agent: str = None):
    handler = load_handler(tool_key)
    return await handler(args, user, agent)


async def run_task_replicate(task: Task):
    task.update(status="running")
    tool = Tool.load(task.tool)
    n_samples = task.args.get("n_samples", 1)
    replicate_model = tool._get_replicate_model(task.args)
    args = tool.prepare_args(task.args)
    args = tool._format_args_for_replicate(args)
    try:
        outputs = []
        for i in range(n_samples):
            task_args = args.copy()
            if "seed" in task_args:
                task_args["seed"] = task_args["seed"] + i
            output = await replicate.async_run(replicate_model, input=task_args)
            outputs.append(output)
        outputs = flatten_list(outputs)
        result = replicate_update_task(task, "succeeded", None, outputs, "normal")
    except Exception as e:
        print(f"Error running replicate: {e}")
        sentry_sdk.capture_exception(e)
        result = replicate_update_task(task, "failed", str(e), None, "normal")
    return result


async def cleanup_stale_busy_states():
    """Clean up any stale busy states in the shared modal.Dict"""
    try:
        current_time = time.time()
        stale_threshold = 300
        logger.info("Starting stale busy state cleanup...")

        # Get all keys from the dictionary first
        all_keys = list(busy_state_dict.keys())  # This is not atomic but necessary
        all_values = list(busy_state_dict.values())
        print(f"Checking keys: {all_keys}")
        print(f"Checking values: {all_values}")

        for key in all_keys:
            try:
                # Get current state
                current_state = busy_state_dict.get(key)
                # Check if state exists and is a dictionary with expected structure
                if (
                    not current_state
                    or not isinstance(current_state, dict)
                    or not all(
                        k in current_state
                        for k in ["requests", "timestamps", "context_map"]
                    )
                ):
                    logger.warning(
                        f"Removing invalid/stale state for key {key}: {current_state}"
                    )
                    # Delete directly if possible and safe
                    if key in busy_state_dict:
                        busy_state_dict.pop(key)
                    continue

                requests = current_state.get("requests", [])
                timestamps = current_state.get("timestamps", {})
                context_map = current_state.get("context_map", {})

                # Ensure correct types after retrieval
                requests = list(requests)
                timestamps = dict(timestamps)
                context_map = dict(context_map)

                stale_requests = []
                active_requests = []
                updated_timestamps = {}
                updated_context_map = {}

                # Iterate over a copy of request IDs
                for request_id in list(requests):
                    timestamp = timestamps.get(request_id, 0)
                    if current_time - timestamp > stale_threshold:
                        stale_requests.append(request_id)
                        logger.info(
                            f"Marking request {request_id} as stale for key {key} (age: {current_time - timestamp:.1f}s)."
                        )
                    else:
                        active_requests.append(request_id)
                        if request_id in timestamps:
                            updated_timestamps[request_id] = timestamps[request_id]
                        if request_id in context_map:
                            updated_context_map[request_id] = context_map[request_id]

                # If any requests were found to be stale, update the state
                if stale_requests:
                    logger.info(
                        f"Cleaning up {len(stale_requests)} stale requests for {key}. Original count: {len(requests)}"
                    )
                    # Update the state in the modal.Dict
                    if not active_requests:
                        # If no active requests left, remove the whole key
                        logger.info(
                            f"Removing key '{key}' as no active requests remain after cleanup."
                        )
                        if key in busy_state_dict:  # Check existence before deleting
                            busy_state_dict.pop(key)
                    else:
                        # Otherwise, update with cleaned lists/dicts
                        new_state = {
                            "requests": active_requests,
                            "timestamps": updated_timestamps,
                            "context_map": updated_context_map,
                        }
                        busy_state_dict.put(key, new_state)
                        logger.info(
                            f"Updated state for key '{key}'. Active requests: {len(active_requests)}"
                        )
                # else: # No stale requests found for this key
                #    logger.debug(f"No stale requests found for key '{key}'.")
            except KeyError:
                logger.warning(
                    f"Key {key} was deleted concurrently during cleanup processing."
                )
                continue  # Key was likely deleted by another process or previous step
            except Exception as key_e:
                logger.error(
                    f"Error processing key '{key}' during cleanup: {key_e}",
                    exc_info=True,
                )
                # Decide how to handle errors: skip key, mark for later deletion, etc.
                # For now, just log and continue to avoid breaking the whole job.

        logger.info("Finished cleaning up stale busy states.")
    except Exception as e:
        logger.error(f"Error in cleanup_stale_busy_states job: {e}", exc_info=True)
        sentry_sdk.capture_exception(e)


# Helper functions


def flatten_list(seq):
    """Flattens a list that is either flat or nested one level deep."""
    return [x for item in seq for x in (item if isinstance(item, list) else [item])]
