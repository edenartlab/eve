import asyncio
import os
from tempfile import NamedTemporaryFile
from typing import Iterator

from elevenlabs.client import ElevenLabs

from eve import utils
from eve.tool import ToolContext

eleven = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))


async def handler(context: ToolContext):
    async def generate_with_params():
        def _generate():
            audio_generator = eleven.music.compose(
                prompt=context.args["prompt"],
                music_length_ms=context.args["duration"] * 1000,
                model_id="music_v1",
            )
            if isinstance(audio_generator, Iterator):
                return b"".join(audio_generator)
            return audio_generator

        return await asyncio.to_thread(_generate)

    audio_generator = await utils.async_exponential_backoff(
        generate_with_params,
        max_attempts=3,  # context.args["max_attempts"],
        initial_delay=1,  # context.args["initial_delay"],
    )

    audio = audio_generator

    # save to file
    audio_file = NamedTemporaryFile(delete=False)
    audio_file.write(audio)
    audio_file.close()

    return {
        "output": audio_file.name,
    }
