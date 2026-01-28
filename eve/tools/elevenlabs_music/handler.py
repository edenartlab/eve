import asyncio
import os
from tempfile import NamedTemporaryFile

import httpx

from eve import utils
from eve.tool import ToolContext

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")


async def handler(context: ToolContext):
    prompt = context.args.get("prompt")
    duration = context.args.get("duration", 30)
    composition_plan = context.args.get("composition_plan")
    force_instrumental = context.args.get("force_instrumental", False)
    respect_sections_durations = context.args.get("respect_sections_durations", True)

    if not prompt and not composition_plan:
        raise ValueError("Must provide either 'prompt' or 'composition_plan'")

    if prompt and composition_plan:
        raise ValueError("Cannot use both 'prompt' and 'composition_plan'")

    async def generate_music():
        def _generate():
            if not ELEVEN_API_KEY:
                raise ValueError("ELEVEN_API_KEY environment variable is not set")

            payload = {
                "model_id": "music_v1",
            }

            if composition_plan:
                payload["composition_plan"] = composition_plan
                payload["respect_sections_durations"] = respect_sections_durations
            else:
                payload["prompt"] = prompt
                payload["music_length_ms"] = int(duration * 1000)
                if force_instrumental:
                    payload["force_instrumental"] = True

            response = httpx.post(
                "https://api.elevenlabs.io/v1/music",
                headers={
                    "xi-api-key": ELEVEN_API_KEY,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=600.0,
            )
            response.raise_for_status()
            return response.content

        return await asyncio.to_thread(_generate)

    audio = await utils.async_exponential_backoff(
        generate_music,
        max_attempts=3,
        initial_delay=1,
    )

    if audio is None:
        raise ValueError("Failed to generate music after multiple attempts")

    # Save to file
    audio_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_file.write(audio)
    audio_file.close()

    return {"output": audio_file.name}
