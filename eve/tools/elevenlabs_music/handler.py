import os
from tempfile import NamedTemporaryFile
from elevenlabs.client import ElevenLabs
from typing import Iterator

from eve import utils
from eve.tool import ToolContext


eleven = ElevenLabs(
    api_key=os.getenv("ELEVEN_API_KEY")
)


async def handler(context: ToolContext):
    
    async def generate_with_params():
        audio_generator = eleven.music.compose(
            prompt=context.args["prompt"],
            music_length_ms=context.args["duration"] * 1000,
            model_id="music_v1"
        )
        return audio_generator

    audio_generator = await utils.async_exponential_backoff(
        generate_with_params,
        max_attempts=3, #context.args["max_attempts"],
        initial_delay=1 #context.args["initial_delay"],
    )

    if isinstance(audio_generator, Iterator):
        audio = b"".join(audio_generator)
    else:
        audio = audio_generator

    # save to file
    audio_file = NamedTemporaryFile(delete=False)
    audio_file.write(audio)
    audio_file.close()

    return {
        "output": audio_file.name,
    }
