# TODO: auto ratio based on the reference video

# Aleph (video style transfer) — async handler using Runway SDK
# Mirrors the structure/flow of your Act-Two and Gen3/Gen4 handlers.

import asyncio
import runwayml
from runwayml import AsyncRunwayML
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


ASYNC_POLL_INTERVAL_SECS = 5


def _ratio_to_resolution(ratio_str: str) -> str:
    """Map friendly ratio string to Runway resolution 'W:H'."""
    if ratio_str == "21:9":
        return "1584:672"
    if ratio_str == "16:9":
        return "1280:720"
    if ratio_str == "4:3":
        return "1104:832"
    if ratio_str == "1:1":
        return "960:960"
    if ratio_str == "3:4":
        return "832:1104"
    if ratio_str == "9:16":
        return "720:1280"
    # default
    return "1280:720"


async def handler(args: dict, user: str = None, agent: str = None):
    """
    Expected args:
      - input_video: str (required)  # URI to the source video to stylize
      - prompt_text: str (optional)  # Style description, e.g. "film noir, 16mm grain"
      - ratio: str (optional)        # One of: 21:9,16:9,4:3,1:1,3:4,9:16
      - seed: int (optional)
      - style_image: str (optional)  # URI to an image reference for style
      - style_video: str (optional)  # URI to a video reference for style
      - public_figure_threshold: str (optional)  # "low" | "auto" | "high"
    """
    client = AsyncRunwayML()
    unsafe_content_error = False

    # Build references array for Aleph style conditioning.
    references = []
    if args.get("style_image"):
        references.append({"type": "image", "uri": args["style_image"]})
    if args.get("style_video"):
        references.append({"type": "video", "uri": args["style_video"]})
    # Keep as empty array, not None - API expects an array

    ratio = _ratio_to_resolution(args.get("ratio", "16:9"))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0, max=15),
        retry=retry_if_exception_type((runwayml.APIConnectionError, runwayml.APIStatusError)),
        retry_error_callback=lambda retry_state: retry_state.outcome.result(),
    )
    async def create_video_to_video():
        nonlocal unsafe_content_error
        try:
            # Note: Aleph is exposed via the /v1/video_to_video endpoint with model "gen4_aleph".
            # Only pass documented params to avoid undefined behavior.
            return await client.video_to_video.create(
                model="gen4_aleph",
                video_uri=args["input_video"],
                prompt_text=(args.get("prompt_text") or "")[:512],
                ratio=ratio,
                references=references,
                seed=args.get("seed"),
                content_moderation={
                    "public_figure_threshold": args.get("public_figure_threshold", "low")
                },
            )

        except runwayml.APIConnectionError:
            raise Exception("The server could not be reached")

        except runwayml.RateLimitError:
            raise Exception("A 429 status code was received; we should back off a bit.")

        except runwayml.APIStatusError as e:
            # Don't retry client errors (4xx)
            if 400 <= e.status_code < 500:
                # Check for safety/unsafe content
                error_text = (str(getattr(e, "response", "") and e.response.text) or "").lower()
                if (
                    "safety" in error_text
                    or "unsafe content" in error_text
                    or "input.text" in error_text
                    or "safety.input" in error_text
                ):
                    unsafe_content_error = True
                    raise Exception(f"Content moderation rejected the request: {getattr(e, 'response', None) and e.response.text}")
                raise Exception("Client error received", e.status_code, getattr(e, "response", None), getattr(e, "response", None) and e.response.text)
            # For 5xx errors, let the retry mechanism handle it
            raise Exception(
                "Server error received",
                e.status_code,
                getattr(e, "response", None),
                getattr(e, "response", None) and e.response.text,
            )

        except Exception as e:
            raise Exception("An unexpected error occurred", e)

    try:
        task = await create_video_to_video()
    except Exception as e:
        print(f"Failed after retries: {e}")
        print(f"Failed due to unsafe content: {unsafe_content_error}")
        if unsafe_content_error:
            raise e
        # Re-raise the exception if the API call failed
        raise e

    if not task:
        raise Exception("No task was returned")

    task_id = task.id
    print("task id", task_id)

    # Poll until completion
    await asyncio.sleep(ASYNC_POLL_INTERVAL_SECS)
    task = await client.tasks.retrieve(task_id)
    while task.status not in ["SUCCEEDED", "FAILED"]:
        print("status", task.status)
        await asyncio.sleep(ASYNC_POLL_INTERVAL_SECS)
        task = await client.tasks.retrieve(task_id)

    print(task)

    if task.status == "FAILED":
        # Inspect for safety-related failure codes
        if task.failure_code and ("SAFETY" in task.failure_code or "INPUT_PREPROCESSING.SAFETY" in task.failure_code):
            unsafe_content_error = True
            print(f"Content safety check failed: {task.failure_code}")
        print("Error", task.failure)
        raise Exception(task.failure)

    print("task output", task.output)

    # Aleph returns one or more output URLs
    return {"output": task.output[0]}
