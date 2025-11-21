import asyncio
import mimetypes
import os
import tempfile
from urllib.parse import urlparse

import requests

from google import genai
from google.oauth2 import service_account


def create_gcp_client(gcp_location: str = None):
    """Create and return a Google GenAI client configured for Vertex AI."""
    if gcp_location is None:
        gcp_location = os.environ["GCP_LOCATION"]

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
        "client_x509_cert_url": os.environ["GCP_CLIENT_X509_CERT_URL"],
    }

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    return genai.Client(
        vertexai=True,
        project=os.environ["GCP_PROJECT_ID"],
        location=gcp_location,
        credentials=credentials,
    )


async def veo_handler(args: dict, model: str):
    if not args.get("prompt") and not args.get("image"):
        raise ValueError("At least one of prompt or image is required")

    # --- setup gcp client ----
    client = create_gcp_client()

    # ---- get image and setup args ----
    config_dict = {
        "duration_seconds": args.get("duration"),  # 5-8 seconds
        "number_of_videos": args.get("n_samples"),  # 1-4
        "enhance_prompt": True,
        # enhance_prompt=args.get("prompt_enhance"),
        # person_generation="dont_allow",   # safety switch
        # storage_uri="gs://MY_BUCKET/veo/" # optional – if omitted you get bytes
    }

    if args.get("aspect_ratio"):
        config_dict["aspect_ratio"] = args.get("aspect_ratio")

    if args.get("negative_prompt"):
        config_dict["negative_prompt"] = args.get("negative_prompt")

    if args.get("generate_audio"):
        config_dict["generate_audio"] = True if args.get("generate_audio") else False

    args_dict = {
        "model": model,
        "config": genai.types.GenerateVideosConfig(**config_dict),
    }

    image = args.get("image")
    if image:
        response = requests.get(image)
        response.raise_for_status()
        mime_type = response.headers.get("content-type")
        if not mime_type:
            mime_type = mimetypes.guess_type(urlparse(image).path)[0]
        image = genai.types.Image(image_bytes=response.content, mime_type=mime_type)
        args_dict["image"] = image

    if args.get("prompt"):
        args_dict["prompt"] = args.get("prompt")

    # ---- start generation ----
    operation = client.models.generate_videos(**args_dict)

    # ---- poll until done ----
    while not operation.done:
        await asyncio.sleep(2)
        operation = client.operations.get(operation)

    if not operation.response or not operation.response.generated_videos:
        raise ValueError("No videos generated")
    # ---- download the video(s) ----
    videos = []
    for result in operation.response.generated_videos:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            tmpfile.write(result.video.video_bytes)
            videos.append(tmpfile.name)

    return {"output": videos}
