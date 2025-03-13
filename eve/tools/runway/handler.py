import asyncio
import runwayml
from runwayml import AsyncRunwayML

"""
Todo:
- Error Unsafe content detected. Please try again with a different text, image, or seed.
- 429
"""


async def handler(args: dict, user: str = None, requester: str = None):
    client = AsyncRunwayML()

    try:
        ratio = "1280:768" if args["ratio"] == "16:9" else "768:1280"

        print("RUNWAY ARGS")
        print(args)
        task = await client.image_to_video.create(
            model="gen3a_turbo",
            prompt_image=args["prompt_image"],
            prompt_text=args["prompt_text"][:512],
            duration=int(args["duration"]),
            ratio=ratio,
            watermark=False,
        )
    except runwayml.APIConnectionError as e:
        raise Exception("The server could not be reached")
    except runwayml.RateLimitError as e:
        raise Exception("A 429 status code was received; we should back off a bit.")
    except runwayml.APIStatusError as e:
        raise Exception(
            "Another non-200-range status code was received",
            e.status_code,
            e.response,
            e.response.text
        )
    except Exception as e:
        raise Exception("An unexpected error occurred", e)

    if not task:
        raise Exception("No task was returned")

    task_id = task.id
    print(task_id)

    # time.sleep(10)
    await asyncio.sleep(10)
    task = await client.tasks.retrieve(task_id)
    while task.status not in ["SUCCEEDED", "FAILED"]:
        print("status", task.status)
        # time.sleep(10)
        await asyncio.sleep(10)
        task = await client.tasks.retrieve(task_id)

    # TODO: callback for running state

    print("task finished2", task.status)
    print(task)

    """
    
    task finished2 FAILED
TaskRetrieveResponse(id='48947b97-c260-492e-b662-bec5aa725ebf', created_at=datetime.datetime(2025, 1, 1, 20, 43, 5, 303000, tzinfo=datetime.timezone.utc), status='FAILED', failure='An unexpected error occurred.', failure_code='INTERNAL.BAD_OUTPUT.CODE01', output=None, progress=None, createdAt='2025-01-01T20:43:05.303Z', failureCode='INTERNAL.BAD_OUTPUT.CODE01')
Error An unexpected error occurred.
    
    """

    if task.status == "FAILED":
        print("Error", task.failure)
        raise Exception(task.failure)

    print("task output", task.output)

    return {"output": task.output[0]}
