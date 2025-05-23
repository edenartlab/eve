import time
from google import genai
from google.genai import types

from eve.user import User
from ....agent import Agent



async def handler(args: dict, user: User, agent: Agent):
    # If you want to hit the Vertex backend instead of the public Gemini one:
    # client = genai.Client(backend=genai.Backend.VertexAI,
    #                       project_id=os.environ["PROJECT_ID"],
    #                       location="us-central1")

    # Otherwise the default backend honours GOOGLE_API_KEY.
    client = genai.Client()          # reads GOOGLE_API_KEY

    print("models")
    for m in client.models.list(): print(m.name)

    operation = client.models.generate_videos(
        model="veo-2.0-generate-001",
        prompt=args.get("prompt"),
        config=types.GenerateVideosConfig(
            duration_seconds=8,          # 5-8 seconds
            # sample_count=1,              # 1-4
            aspect_ratio="16:9",         # or "9:16"
            # person_generation="dont_allow",   # safety switch
            # storage_uri="gs://MY_BUCKET/veo/" # optional – if omitted you get bytes
        ),
    )

    print("OPERATION", operation)

    # ---- poll until done ----
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)

    # ---- download the video(s) ----
    videos = []
    for n, vid in enumerate(operation.response.generated_videos):
        client.files.download(file=vid.video)
        vid.video.save(f"sample_{n}.mp4")
        print("saved", f"sample_{n}.mp4")
        videos.append(f"sample_{n}.mp4")
    
    return {"output": videos}
