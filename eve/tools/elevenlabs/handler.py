import os
import random
import instructor
from tempfile import NamedTemporaryFile
from typing import List, Literal
from elevenlabs.client import ElevenLabs, VoiceSettings, Voice
from openai import OpenAI
from typing import Iterator

from ... import eden_utils

eleven = ElevenLabs(
    api_key=os.getenv("ELEVEN_API_KEY")
)

DEFAULT_VOICE = "XB0fDUnXU5powFXDhCwa"


async def handler(args: dict, user: str = None, agent: str = None):
    # print("args", args)
    args["stability"] = args.get("stability", 0.5)
    args["similarity_boost"] = args.get("similarity_boost", 0.75)
    args["style"] = args.get("style", 0.0)
    args["use_speaker_boost"] = args.get("use_speaker_boost", True)
    args["max_attempts"] = args.get("max_attempts", 3)
    args["initial_delay"] = args.get("initial_delay", 1)

    # get voice
    response = eleven.voices.get_all()
    voices = {v.name: v.voice_id for v in response.voices}
    voice_id = voices.get(args["voice"], DEFAULT_VOICE)
    
    def generate_with_params():
        return eleven.generate(
            text=args["text"],
            voice=Voice(
                voice_id=voice_id,
                settings=VoiceSettings(
                    stability=args["stability"],
                    similarity_boost=args["similarity_boost"],
                    style=args["style"],
                    use_speaker_boost=args["use_speaker_boost"],
                )
            ),
            model="eleven_multilingual_v2"
        )

    audio = eden_utils.exponential_backoff(
        generate_with_params,
        max_attempts=args["max_attempts"],
        initial_delay=args["initial_delay"],
    )

    if isinstance(audio, Iterator):
        audio = b"".join(audio)

    # save to file
    audio_file = NamedTemporaryFile(delete=False)
    audio_file.write(audio)
    audio_file.close()

    return {
        "output": audio_file.name,
    }


def clone_voice(name, description, voice_urls):
    voice_files = []
    for url in voice_urls:
        with NamedTemporaryFile(delete=False) as file:
            file = eden_utils.download_file(url, file.name)
            voice_files.append(file)
    voice = eleven.clone(name, voice_files, description)    
    for file in voice_files:
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
        voices = [v for v in voices if v.labels.get('gender') == gender]

    if exclude:
        voices = [v for v in voices if v.voice_id not in exclude]
        
    if not description:
        return random.choice(voices)

    voice_ids = {v.name: v.voice_id for v in voices}
    voice_descriptions = "\n".join([f"{v.name}: {', '.join(v.labels.values())}, {v.description or ''}" for v in voices])

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

    #return voice_ids[selected_voice]
    return selected_voice


def get_voice_summary():
    response = eleven.voices.get_all()
    names = [voice.name for voice in response.voices]
    full_description = ""
    
    for voice in response.voices:
        name = voice.name
        description = voice.description or ""
        labels = voice.labels or {}
        description = (description or "") + ", "
        description += ", ".join([f"{k}: {v}" for k, v in labels.items()])    
        full_description += f"{name}: {description}\n"
    
    return names, full_description
