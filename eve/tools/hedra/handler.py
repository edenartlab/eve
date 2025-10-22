from eve.tool import ToolContext
import os
import logging
import requests
import tempfile
import asyncio
import subprocess
from ... import utils


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


def get_audio_duration(audio_file_path: str) -> float:
    """Get the duration of an audio file in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        logger.info(f"Audio duration: {duration} seconds")
        return duration
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        logger.warning(f"Could not get audio duration using ffprobe: {e}")
        # Fallback: try using wave for .wav files or return None
        return None


async def handler(context: ToolContext):
    HEDRA_API_KEY = os.getenv("HEDRA_API_KEY")
    session = Session(api_key=HEDRA_API_KEY)

    # Create temp files with appropriate extensions
    temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)

    try:
        # Download files using utils
        image = utils.download_file(context.args["image"], temp_image.name, overwrite=True)
        audio_file = utils.download_file(
            context.args["audio"], temp_audio.name, overwrite=True
        )

        logger.info("testing against %s", session.base_url)
        models = session.get("/models").json()

        # Use Hedra Character 3 model for audio-based talking heads (supports auto duration)
        # Otherwise fall back to first model
        character_model = next((m for m in models if "Character" in m.get("name", "")), None)
        model = character_model if character_model else models[0]
        model_id = model["id"]
        logger.info(f"Using model: {model.get('name', 'Unknown')} (ID: {model_id})")

        # Get valid durations for this model
        durations = model.get("durations", [])
        if durations and durations != ["auto"]:
            valid_durations_ms = [int(d) for d in durations]
            logger.info(f"Model valid durations (ms): {valid_durations_ms}")
        else:
            valid_durations_ms = []
            if durations == ["auto"]:
                logger.info("Model supports auto duration from audio")

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

        # Build generated_video_inputs matching the official starter structure
        generated_video_inputs = {
            "text_prompt": context.args["prompt"],
            "resolution": context.args.get("resolution", "540p"),  # Default to 540p if not provided
            "aspect_ratio": context.args.get("aspectRatio", "1:1"),  # Default to 1:1 if not provided
        }

        # Add duration if explicitly provided (in seconds), otherwise get it from audio
        logger.info(f"Args: {args}")
        if context.args.get("duration") is not None:
            duration_seconds = context.args["duration"]
            logger.info(f"Using explicit duration: {duration_seconds}s")
        else:
            # Get audio duration
            audio_duration = get_audio_duration(audio_file)
            if audio_duration is not None:
                duration_seconds = audio_duration
                logger.info(f"Using audio duration: {duration_seconds}s")
            else:
                logger.warning("Could not get audio duration, proceeding without duration")
                duration_seconds = None

        # Convert duration to milliseconds and clamp to valid model durations
        if duration_seconds is not None and durations != ["auto"]:
            requested_duration_ms = int(duration_seconds * 1000)

            # If model has specific valid durations, use the closest one
            if valid_durations_ms:
                # Find the closest valid duration
                duration_ms = min(valid_durations_ms, key=lambda x: abs(x - requested_duration_ms))
                if duration_ms != requested_duration_ms:
                    logger.info(f"Requested duration {requested_duration_ms}ms not valid for this model. Using closest valid duration: {duration_ms}ms")
                else:
                    logger.info(f"Using valid duration: {duration_ms}ms")
            else:
                duration_ms = requested_duration_ms
                logger.info(f"Set duration_ms to {duration_ms}")

            generated_video_inputs["duration_ms"] = duration_ms
        elif durations == ["auto"]:
            logger.info("Skipping duration_ms - model will auto-detect from audio")

        # Add optional seed if provided
        if context.args.get("seed") is not None:
            generated_video_inputs["seed"] = context.args["seed"]

        generation_request_data = {
            "type": "video",
            "ai_model_id": model_id,
            "start_keyframe_id": image_id,
            "audio_id": audio_id,
            "generated_video_inputs": generated_video_inputs,
        }

        logger.info(f"Sending generation request: {generation_request_data}")
        response = session.post("/generations", json=generation_request_data)
        generation_response = response.json()
        logger.info(f"Generation response: {generation_response}")

        # Check if the response contains an error
        if "code" in generation_response or "id" not in generation_response:
            error_msg = generation_response.get("messages", ["Unknown error"])
            raise Exception(f"API error: {error_msg}. Full response: {generation_response}")

        generation_id = generation_response["id"]
        while True:
            status_response = session.get(f"/generations/{generation_id}/status").json()
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

            # Use a fresh requests get, not the session, as the URL is likely presigned S3
            # with requests.get(download_url, stream=True) as r:
            #     r.raise_for_status() # Check if the request was successful
            #     with open(output_filename, 'wb') as f:
            #         for chunk in r.iter_content(chunk_size=8192):
            #             f.write(chunk)
            # logger.info(f"Successfully downloaded video to {output_filename}")
            return {"output": download_url}

        elif status == "error":
            raise Exception(
                f"Video generation failed: {status_response.get('error_message', 'Unknown error')}"
            )

        else:
            # This case might happen if loop breaks unexpectedly or API changes
            raise Exception(
                f"Video generation finished with status '{status}' but no download URL was found."
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

    finally:
        os.unlink(temp_image.name)
        os.unlink(temp_audio.name)
