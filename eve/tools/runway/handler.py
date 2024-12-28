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
        task = client.image_to_video.create(
            model='gen3a_turbo',
            prompt_image=args["prompt_image"],
            prompt_text=args["prompt_text"][:512]
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

    if task.status == "FAILED":
        print("Error", task.failure)
        raise Exception(task.failure)
    
    print("task output", task.output)
    
    return {
        "output": task.output[0]
    }
