# TODO: auto ratio based on the reference video
import asyncio

import runwayml
from loguru import logger
from runwayml import AsyncRunwayML

from eve.tool import ToolContext


async def handler(context: ToolContext):
    client = AsyncRunwayML()
    unsafe_content_error = False

    # Submitted exactly once — deliberately no retry wrapper. The runwayml SDK
    # already retries connection errors and 408/409/429/5xx internally on the
    # create POST (BaseClient._should_retry, DEFAULT_MAX_RETRIES=2), i.e. before
    # a Runway task exists, for free. A retry layered on top of that re-issues
    # the create call, which mints a NEW task id — a second paid generation
    # against a single manna charge. The predicate it used made that worse:
    # APIStatusError is the base class of every 4xx and 5xx, so a deterministic
    # 400 counted as retryable.
    async def create_image_to_video():
        nonlocal unsafe_content_error
        try:
            if context.args["ratio"] == "16:9":
                ratio = "1280:720"
            elif context.args["ratio"] == "4:3":
                ratio = "1104:832"
            elif context.args["ratio"] == "1:1":
                ratio = "960:960"
            elif context.args["ratio"] == "3:4":
                ratio = "832:1104"
            elif context.args["ratio"] == "9:16":
                ratio = "720:1280"
            elif context.args["ratio"] == "9:21":
                ratio = "672:1584"
            else:
                ratio = "1280:720"
            # run Runway client command

            return await client.character_performance.create(
                character={"type": "image", "uri": context.args["character_image"]},
                model="act_two",
                ratio=ratio,
                reference={"type": "video", "uri": context.args["reference_video"]},
                body_control=context.args["body_control"],
                content_moderation={"public_figure_threshold": "low"},
                expression_intensity=context.args["expression_intensity"],
                seed=context.args["seed"],
            )

        except runwayml.APIConnectionError:
            raise Exception("The server could not be reached")

        except runwayml.RateLimitError:
            raise Exception("A 429 status code was received; we should back off a bit.")

        except runwayml.APIStatusError as e:
            # Don't retry client errors (4xx)
            if 400 <= e.status_code < 500:
                # Check if this is a safety/unsafe content error
                error_text = str(e.response.text).lower()
                if (
                    "safety" in error_text
                    or "unsafe content" in error_text
                    or "input.text" in error_text
                    or "safety.input" in error_text
                ):
                    unsafe_content_error = True
                    raise Exception(
                        f"Content moderation rejected the request: {e.response.text}"
                    )

                raise Exception(
                    "Client error received", e.status_code, e.response, e.response.text
                )
            # For 5xx errors, let the retry mechanism handle it
            raise Exception(
                "Server error received", e.status_code, e.response, e.response.text
            )

        except Exception as e:
            raise Exception("An unexpected error occurred", e)

    try:
        task = await create_image_to_video()
    except Exception as e:
        logger.error(f"Failed after retries: {e}")
        logger.error(f"Failed due to unsafe content: {unsafe_content_error}")

        if unsafe_content_error:
            raise e

    if not task:
        raise Exception("No task was returned")

    task_id = task.id

    await asyncio.sleep(5)
    task = await client.tasks.retrieve(task_id)
    while task.status not in ["SUCCEEDED", "FAILED"]:
        await asyncio.sleep(5)
        task = await client.tasks.retrieve(task_id)

    if task.status == "FAILED":
        # Check for unsafe content in task failure
        if task.failure_code and (
            "SAFETY" in task.failure_code
            or "INPUT_PREPROCESSING.SAFETY" in task.failure_code
        ):
            unsafe_content_error = True
            logger.error(f"Content safety check failed: {task.failure_code}")

        logger.error(f"Error: {task.failure}")
        raise Exception(task.failure)

    return {"output": task.output[0]}
