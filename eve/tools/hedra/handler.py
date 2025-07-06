import os
import time
import logging
import requests
import tempfile
import asyncio
from ... import eden_utils


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class Session(requests.Session):
    def __init__(self, api_key: str):
        super().__init__()

        self.base_url: str = "https://api.hedra.com/web-app/public"
        self.headers["x-api-key"] = api_key

    # @override
    def prepare_request(self, request: requests.Request) -> requests.PreparedRequest:
        request.url = f"{self.base_url}{request.url}"

        return super().prepare_request(request)


async def handler(args: dict, user: str = None, agent: str = None):
    HEDRA_API_KEY = os.getenv("HEDRA_API_KEY")
    session = Session(api_key=HEDRA_API_KEY)

    # Create temp files with appropriate extensions
    temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)

    try:
        # Download files using eden_utils
        image = eden_utils.download_file(args["image"], temp_image.name, overwrite=True)
        audio_file = eden_utils.download_file(
            args["audio"], temp_audio.name, overwrite=True
        )

        logger.info("testing against %s", session.base_url)
        model_id = session.get("/models").json()[0]["id"]
        logger.info("got model id %s", model_id)

        image_response = session.post(
            "/assets",
            json={"name": os.path.basename(image), "type": "image"},
        )
        if not image_response.ok:
            logger.error(
                "error creating image: %d %s",
                image_response.status_code,
                image_response.json(),
            )
        image_id = image_response.json()["id"]
        with open(image, "rb") as f:
            session.post(
                f"/assets/{image_id}/upload", files={"file": f}
            ).raise_for_status()
        logger.info("uploaded image %s", image_id)

        audio_id = session.post(
            "/assets", json={"name": os.path.basename(audio_file), "type": "audio"}
        ).json()["id"]
        with open(audio_file, "rb") as f:
            session.post(
                f"/assets/{audio_id}/upload", files={"file": f}
            ).raise_for_status()
        logger.info("uploaded audio %s", audio_id)

        generation_request_data = {
            "type": "video",
            "ai_model_id": model_id,
            "start_keyframe_id": image_id,
            "audio_id": audio_id,
            "generated_video_inputs": {
                "text_prompt": args["prompt"],
                "resolution": args["resolution"],
                "aspect_ratio": args["aspectRatio"],
            },
        }

        # Add optional parameters if provided
        # if duration is not None:
        #     generation_request_data["generated_video_inputs"]["duration_ms"] = int(duration * 1000)
        # if seed is not None:
        #     generation_request_data["generated_video_inputs"]["seed"] = seed

        generation_response = session.post(
            "/generations", json=generation_request_data
        ).json()
        logger.info("***debug*** generation_response: %s", generation_response)
        generation_id = generation_response["id"]
        while True:
            status_response = session.get(f"/generations/{generation_id}/status").json()
            logger.info("***debug*** status response %s", status_response)
            status = status_response["status"]

            # --- Check for completion or error to break the loop ---
            if status in ["complete", "error"]:
                break

            # Use async sleep to yield control back to the event loop
            await asyncio.sleep(5)

        # --- Process final status (download or log error) ---
        if status == "complete" and status_response.get("url"):
            download_url = status_response["url"]
            # Use asset_id for filename if available, otherwise use generation_id
            output_filename_base = status_response.get("asset_id", generation_id)
            output_filename = f"{output_filename_base}.mp4"
            logger.info(
                f"***debug*** Generation complete. Downloading video from {download_url} to {output_filename}"
            )

            # Use a fresh requests get, not the session, as the URL is likely presigned S3
            # with requests.get(download_url, stream=True) as r:
            #     r.raise_for_status() # Check if the request was successful
            #     with open(output_filename, 'wb') as f:
            #         for chunk in r.iter_content(chunk_size=8192):
            #             f.write(chunk)
            # logger.info(f"Successfully downloaded video to {output_filename}")
            return {"output": download_url}

        elif status == "error":
            logger.error(
                f"***debug*** Video generation failed: {status_response.get('error_message', 'Unknown error')}"
            )
            raise Exception(
                f"Video generation failed: {status_response.get('error_message', 'Unknown error')}"
            )

        else:
            # This case might happen if loop breaks unexpectedly or API changes
            logger.warning(
                f"***debug*** Video generation finished with status '{status}' but no download URL was found."
            )
            raise Exception(
                f"Video generation finished with status '{status}' but no download URL was found."
            )

    except Exception as e:
        logger.error(f"***debug*** Error: {e}")
        raise e

    finally:
        os.unlink(temp_image.name)
        os.unlink(temp_audio.name)
