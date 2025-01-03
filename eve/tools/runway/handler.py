import time
import runwayml
from runwayml import RunwayML

"""
Todo:
- Error Unsafe content detected. Please try again with a different text, image, or seed.
- 429
"""


async def handler(args: dict, db: str):
    client = RunwayML()

    
    

    try:
        ratio = "1280:768" if args["ratio"] == "16:9" else "768:1280"
        
        task = client.image_to_video.create(
            model='gen3a_turbo',
            prompt_image=args["prompt_image"],
            prompt_text=args["prompt_text"][:512],
            duration=int(args["duration"]),
            ratio=ratio,
            watermark=False
        )
    except runwayml.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except runwayml.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except runwayml.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)
        """
        400	BadRequestError
        401	AuthenticationError
        403	PermissionDeniedError
        404	NotFoundError
        422	UnprocessableEntityError
        429	RateLimitError
        >=500	InternalServerError
        N/A	APIConnectionError
        """



    task_id = task.id
    print(task_id)

    time.sleep(10)
    task = client.tasks.retrieve(task_id)
    while task.status not in ['SUCCEEDED', 'FAILED']:
        print("status", task.status)
        time.sleep(10) 
        task = client.tasks.retrieve(task_id)
    
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
    
    return {
        "output": task.output[0]
    }
