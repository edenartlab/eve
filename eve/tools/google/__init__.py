import os
import asyncio
import requests
import tempfile
import mimetypes
from urllib.parse import urlparse
from google import genai
from google.oauth2 import service_account
from google.genai.types import GenerateVideosConfig


async def veo_handler(args: dict, model: str):
    if not args.get("prompt") and not args.get("image"):
        raise ValueError("At least one of prompt or image is required")
    
    # --- setup gcp client ----
    service_account_info = {
        "type": os.environ["GCP_TYPE"],
        "project_id": os.environ["GCP_PROJECT_ID"],
        "private_key_id": os.environ["GCP_PRIVATE_KEY_ID"],
        "private_key": os.environ["GCP_PRIVATE_KEY"].replace("\\n", "\n"),
        "client_email": os.environ["GCP_CLIENT_EMAIL"],
        "client_id": os.environ["GCP_CLIENT_ID"],
        "auth_uri": os.environ["GCP_AUTH_URI"],
        "token_uri": os.environ["GCP_TOKEN_URI"],
        "auth_provider_x509_cert_url": os.environ["GCP_AUTH_PROVIDER_X509_CERT_URL"],
        "client_x509_cert_url": os.environ["GCP_CLIENT_X509_CERT_URL"]
    }

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    client = genai.Client(
        vertexai=True,
        project=os.environ["GCP_PROJECT_ID"],
        location=os.environ["GCP_LOCATION"],
        credentials=credentials,
    )

    print("client", client)

    # ---- list models ----
    # for m in client.models.list(): print(m.name)

    # ---- get image and setup args ----
    config_dict = {
        "duration_seconds": args.get("duration"),         # 5-8 seconds
        "number_of_videos": args.get("n_samples"),        # 1-4
        "enhance_prompt": True,
        # enhance_prompt=args.get("prompt_enhance"),
        # person_generation="dont_allow",   # safety switch
        # storage_uri="gs://MY_BUCKET/veo/" # optional – if omitted you get bytes
    }

    if args.get("aspect_ratio"):
        config_dict["aspect_ratio"] = args.get("aspect_ratio")

    if args.get("negative_prompt"):
        config_dict["negative_prompt"] = args.get("negative_prompt")

    # if args.get("generate_audio"):
    #     config_dict["generate_audio"] = True if args.get("generate_audio") else False
    
    args_dict = {
        "model": model,
        "config": genai.types.GenerateVideosConfig(**config_dict),
    }

    print("HERE ARE")
    print(args_dict)
    print("----")
    print(config_dict)

    image = args.get("image")
    if image:
        response = requests.get(image)
        response.raise_for_status()
        mime_type = response.headers.get('content-type')
        if not mime_type:
            mime_type = mimetypes.guess_type(urlparse(image).path)[0]
        image = genai.types.Image(image_bytes=response.content, mime_type=mime_type)
        args_dict["image"] = image
            
    if args.get("prompt"):
        args_dict["prompt"] = args.get("prompt")


    # ---- start generation ----
    operation = client.models.generate_videos(
        **args_dict
    )
    print("operation", operation)


    # ---- poll until done ----
    while not operation.done:
        await asyncio.sleep(2)
        operation = client.operations.get(operation)


    # ---- download the video(s) ----
    videos = []
    for result in operation.response.generated_videos:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            tmpfile.write(result.video.video_bytes)
            videos.append(tmpfile.name)
    
    return {"output": videos}

