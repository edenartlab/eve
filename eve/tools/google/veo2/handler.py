import time
import requests
import mimetypes
from urllib.parse import urlparse
from google import genai

from ....user import User
from ....agent import Agent


async def handler(args: dict, user: str = None, agent: str = None):
    # If you want to hit the Vertex backend instead of the public Gemini one:
    # client = genai.Client(backend=genai.Backend.VertexAI,
    #                       project_id=os.environ["PROJECT_ID"],
    #                       location="us-central1")

    # Otherwise the default backend honours GOOGLE_API_KEY.
    client = genai.Client() # reads GOOGLE_API_KEY
    
    # for m in client.models.list(): print(m.name)

    if not args.get("prompt") and not args.get("image"):
        raise ValueError("At least one of prompt or image is required")

    # get image from URL and convert to bytes
    image = args.get("image")
    if image:
        response = requests.get(image)
        response.raise_for_status()
        mime_type = response.headers.get('content-type')
        if not mime_type:
            mime_type = mimetypes.guess_type(urlparse(image).path)[0]
        image = genai.types.Image(image_bytes=response.content, mime_type=mime_type)
 
    operation = client.models.generate_videos(
        model="veo-2.0-generate-001",
        image=image,
        prompt=args.get("prompt"),
        config=genai.types.GenerateVideosConfig(
            duration_seconds=args.get("duration"),         # 5-8 seconds
            number_of_videos=args.get("n_samples"),        # 1-4
            aspect_ratio=args.get("aspect_ratio"),         # or "9:16"
            negative_prompt=args.get("negative_prompt"),
            # enhance_prompt=args.get("prompt_enhance"),
            # person_generation="dont_allow",   # safety switch
            # storage_uri="gs://MY_BUCKET/veo/" # optional – if omitted you get bytes
        ),
    )

    # ---- poll until done ----
    while not operation.done:
        time.sleep(5)
        operation = client.operations.get(operation)

    print("Response", operation.response)
    print("Generated videos", operation.response.generated_videos)

    print("__________")

    # ---- download the video(s) ----
    videos = []
    for n, vid in enumerate(operation.response.generated_videos):
        client.files.download(file=vid.video)
        vid.video.save(f"sample_{n}.mp4")
        videos.append(f"sample_{n}.mp4")
    
    return {"output": videos}
