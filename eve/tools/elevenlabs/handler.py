import asyncio
import base64
import os
import random
from tempfile import NamedTemporaryFile
from typing import List, Literal

import httpx
import instructor
from elevenlabs.client import ElevenLabs

# from elevenlabs.types.voice_settings import VoiceSettings
from openai import OpenAI

from eve import utils
from eve.tool import ToolContext

eleven = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

DEFAULT_VOICE = "XB0fDUnXU5powFXDhCwa"


async def handler(context: ToolContext):
    args = context.args
    stability = args.get("stability", 0.5)
    style = args.get("style", 0.0)
    speed = args.get("speed", 1.0)
    text = args["text"]

    async def generate_with_timestamps():
        def _generate():
            # Get all available voices for resolution
            response = eleven.voices.get_all()
            voices = {v.name: v.voice_id for v in response.voices}
            voice_ids = [v.voice_id for v in response.voices]
            voice_id = args.get("voice", DEFAULT_VOICE)

            if voice_id not in voice_ids:
                if voice_id in voices:
                    voice_id = voices[voice_id]
                else:
                    raise ValueError(
                        f"Voice ID {voice_id} not found, try another one (DEFAULT_VOICE: {DEFAULT_VOICE})"
                    )

            # Get voice name for transcript
            voice_name = None
            for name, vid in voices.items():
                if vid == voice_id:
                    voice_name = name.split(" - ")[0]  # Get first part before " - "
                    break
            if not voice_name:
                voice_name = "Narrator"

            if not ELEVEN_API_KEY:
                raise ValueError("ELEVEN_API_KEY environment variable is not set")

            # Use the with-timestamps endpoint for transcript generation
            api_response = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
                headers={
                    "xi-api-key": ELEVEN_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": stability,
                        "style": style,
                        "speed": speed,
                        "use_speaker_boost": True,
                    },
                    "output_format": "mp3_44100_128",
                },
                timeout=120.0,
            )
            api_response.raise_for_status()
            return api_response.json(), voice_name

        return await asyncio.to_thread(_generate)

    response = await utils.async_exponential_backoff(
        generate_with_timestamps,
        max_attempts=3,
        initial_delay=1,
    )

    if response is None:
        raise ValueError("Failed to generate speech after multiple attempts")

    result, voice_name = response

    # Decode audio from base64
    audio_bytes = base64.b64decode(result["audio_base64"])

    # Save to file
    audio_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    audio_file.write(audio_bytes)
    audio_file.close()

    # Build transcript from alignment data
    alignment = result.get("alignment", {})
    # characters = alignment.get("characters", [])
    start_times = alignment.get("character_start_times_seconds", [])
    end_times = alignment.get("character_end_times_seconds", [])

    # Get overall start and end times
    if start_times and end_times:
        start_time = round(start_times[0], 1)
        end_time = round(end_times[-1], 1)
    else:
        # Fallback: estimate from text length (~150 words per minute)
        word_count = len(text.split())
        duration = (word_count / 150) * 60
        start_time = 0.0
        end_time = round(duration, 1)

    # Format transcript like elevenlabs_dialogue
    transcript = f"{voice_name} {start_time}-{end_time} : {text}"

    return {
        "output": audio_file.name,
        "transcript": transcript,
    }


def clone_voice(name, description, voice_files):
    cloning_files = []
    for file in voice_files:
        if isinstance(file, str) and file.startswith("http"):
            with NamedTemporaryFile(delete=False) as file:
                file = utils.download_file(file, file.name)
                cloning_files.append(file)
        else:
            cloning_files.append(file)
    voice = eleven.clone(name, cloning_files, description)
    for file in cloning_files:
        if file.endswith(".tmp"):
            os.remove(file)
    return voice


def select_random_voice(
    description: str = None,
    gender: str = None,
    autofilter_by_gender: bool = False,
    exclude: List[str] = None,
):
    response = eleven.voices.get_all()
    voices = response.voices
    random.shuffle(voices)

    client = instructor.from_openai(OpenAI())

    if autofilter_by_gender and not gender:
        prompt = f"""You are given the following description of a person:

        ---
        {description}
        ---

        Predict the most likely gender of this person."""

        gender = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Literal["male", "female"],
            max_retries=2,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at predicting the gender of a person based on their description.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    if gender:
        assert gender in ["male", "female"], "Gender must be either 'male' or 'female'"
        voices = [v for v in voices if v.labels.get("gender") == gender]

    if exclude:
        voices = [v for v in voices if v.voice_id not in exclude]

    if not description:
        return random.choice(voices)

    voice_ids = {v.name: v.voice_id for v in voices}
    voice_descriptions = "\n".join(
        [
            f"{v.name}: {', '.join(v.labels.values())}, {v.description or ''}"
            for v in voices
        ]
    )

    prompt = f"""You are given the follow list of voices and their descriptions.

    ---
    {voice_descriptions}
    ---

    You are given the following description of a desired character:

    ---
    {description}
    ---

    Select the voice that best matches the description of the character."""

    selected_voice = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Literal[*voice_ids.keys()],
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at selecting the right voice for a character.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return voice_ids[selected_voice]


def get_voice_summary():
    response = eleven.voices.get_all(show_legacy=False)
    full_description = ""

    ids, names = [], []
    for voice in response.voices:
        id = voice.voice_id
        name = voice.name
        description = voice.description or ""
        labels = voice.labels or {}
        description = ", ".join([v for k, v in labels.items() if v])
        full_description += f"{id} :: {name}, {description}\n"
        ids.append(id)
        names.append(name)

    return ids, names, full_description


def save_to_mongo():
    from eve.mongo import get_collection

    response = eleven.voices.get_all()
    collection = get_collection("voices")
    data = [
        {"key": voice.name, "elevenlabs_id": voice.voice_id}
        for voice in response.voices
    ]
    collection.insert_many(data)


# # if __name__ == "__main__":
#     # example.py
# from elevenlabs.client import ElevenLabs
# from elevenlabs import play
# import os
# from dotenv import load_dotenv
# load_dotenv()


# music_generator = eleven.music.compose(
#     prompt="Create an retro 80s video game style track with low-bit sound effects and a retro synth groove like a Zelda or Mario game",
#     music_length_ms=40000,
# )

# # play(track)
# if isinstance(music_generator, Iterator):
#     audio = b"".join(music_generator)
# else:
#     audio = music_generator

# # save to file
# audio_file = NamedTemporaryFile(delete=False)
# audio_file.write(audio)
# audio_file.close()
