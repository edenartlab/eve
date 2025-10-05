# TODO: auto ratio based on the reference video

import asyncio
import runwayml
from runwayml import AsyncRunwayML
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    client = AsyncRunwayML()
    unsafe_content_error = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0, max=15),
        retry=retry_if_exception_type((runwayml.APIConnectionError, runwayml.APIStatusError)),
        retry_error_callback=lambda retry_state: retry_state.outcome.result()
    )
    async def create_image_to_video():
        nonlocal unsafe_content_error
        try:
            print("Runway args", args)
            
            if args["ratio"] == "16:9":
                ratio = "1280:720"
            elif args["ratio"] == "4:3":
                ratio = "1104:832"
            elif args["ratio"] == "1:1":
                ratio = "960:960"
            elif args["ratio"] == "3:4":
                ratio = "832:1104"
            elif args["ratio"] == "9:16":
                ratio = "720:1280"
            elif args["ratio"] == "9:21":
                ratio = "672:1584"
            else:
                ratio = "1280:720"
            # run Runway client command

            return await client.character_performance.create(
                character={
                    "type": "image",
                    "uri": args["character_image"]
                },
                model="act_two",
                ratio=ratio,
                reference={
                    "type": "video",
                    "uri": args["reference_video"]
                },
                body_control=args["body_control"],
                content_moderation={"public_figure_threshold": "low"},
                expression_intensity=args["expression_intensity"],
                seed=args["seed"],
            )

        except runwayml.APIConnectionError as e:
            raise Exception("The server could not be reached")

        except runwayml.RateLimitError as e:
            raise Exception("A 429 status code was received; we should back off a bit.")

        except runwayml.APIStatusError as e:
            # Don't retry client errors (4xx)
            if 400 <= e.status_code < 500:
                
                # Check if this is a safety/unsafe content error
                error_text = str(e.response.text).lower()
                if ("safety" in error_text or 
                    "unsafe content" in error_text or 
                    "input.text" in error_text or
                    "safety.input" in error_text):
                    unsafe_content_error = True
                    raise Exception(f"Content moderation rejected the request: {e.response.text}")

                raise Exception(
                    "Client error received",
                    e.status_code,
                    e.response,
                    e.response.text
                )
            # For 5xx errors, let the retry mechanism handle it
            raise Exception(
                "Server error received",
                e.status_code,
                e.response,
                e.response.text
            )
        
        except Exception as e:
            raise Exception("An unexpected error occurred", e)

    try:
        task = await create_image_to_video()    
    except Exception as e:
        print(f"Failed after retries: {e}")
        print(f"Failed due to unsafe content: {unsafe_content_error}")
        
        # if unsafe_content_error:
        #     print("Retrying...")
        #     task = await create_image_to_video()
    
        # if the error is still there, raise it
        if unsafe_content_error:
            raise e

    if not task:
        raise Exception("No task was returned")

    task_id = task.id
    print(task_id)

    await asyncio.sleep(5)
    task = await client.tasks.retrieve(task_id)
    while task.status not in ["SUCCEEDED", "FAILED"]:
        print("status", task.status)
        await asyncio.sleep(5)
        task = await client.tasks.retrieve(task_id)

    print(task)

    if task.status == "FAILED":
        # Check for unsafe content in task failure
        if (task.failure_code and 
            ("SAFETY" in task.failure_code or "INPUT_PREPROCESSING.SAFETY" in task.failure_code)):
            unsafe_content_error = True
            print(f"Content safety check failed: {task.failure_code}")
        
        print("Error", task.failure)
        raise Exception(task.failure)

    print("task output", task.output)

    return {"output": task.output[0]}
