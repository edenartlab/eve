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
    #                       project_id=os.environ.get("PROJECT_ID"),
    #                       location="us-central1")

    # Otherwise the default backend honours GOOGLE_API_KEY.
    # client = genai.Client() # reads GOOGLE_API_KEY
    

    # import os
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    #     # json_file
    # )
    import base64, time, mimetypes, os
    from google import genai                       # pip install -U google-generativeai
    from google.genai.types import GenerateVideosConfig

    PROJECT  = "eden-training-435413"
    REGION   = "us-central1"
    MODEL_ID = "veo-3.0-generate-preview"
    # MODEL_ID = "veo-2.0-generate-001"

    client = genai.Client(
        # vertexai=True,          # ← this makes it use your GCP creds
        # project=PROJECT,
        # location=REGION,
    )


    for m in client.models.list(): print(m.name)

    # raise Exception("Not implemented4")
    
    # 2️⃣ Kick off generation – **no output_gcs_uri parameter**
    op = client.models.generate_videos(
        model=MODEL_ID,
        prompt="Slow tilt-up revealing a serene Martian canyon at golden hour",
        config=GenerateVideosConfig(duration_seconds=8)      # 8-s default codec = WebM
    )


    # 3️⃣ Poll the long-running operation
    while not op.done:
        time.sleep(15)
        op = client.operations.get(op)
        # print(f"state → {op.metadata.state}")


    # 4️⃣ Grab bytes & save locally
    clip = op.result.generated_videos[0].video          # proto message
    mime = clip.mime_type                               # e.g. 'video/mp4'
    ext  = mimetypes.guess_extension(mime) or ".bin"

    # The bytes are directly available in video_bytes
    raw = clip.video_bytes

    with open(f"veo_ou2tput2{ext}", "wb") as f:
        f.write(raw)

    print("Saved →", f.name)

    # raise Exception("Not implemented2")

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

    print("Operation", operation)
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
