import os
import random
import instructor
from tempfile import NamedTemporaryFile
from typing import List, Literal
from elevenlabs.client import ElevenLabs
# from elevenlabs.types.voice_settings import VoiceSettings
from openai import OpenAI
from typing import Iterator


import elevenlabs
print("versio", elevenlabs.__version__)


from eve import utils

eleven = ElevenLabs(
    api_key=os.getenv("ELEVEN_API_KEY")
)

DEFAULT_VOICE = "XB0fDUnXU5powFXDhCwa"


async def handler(args: dict, user: str = None, agent: str = None):
    # print("args", args)
    args["stability"] = args.get("stability", 0.5)
    args["style"] = args.get("style", 0.0)
    args["speed"] = args.get("speed", 1.0)
    # args["similarity_boost"] = args.get("similarity_boost", 0.75)
    # args["use_speaker_boost"] = args.get("use_speaker_boost", True)
    # args["max_attempts"] = args.get("max_attempts", 3)
    # args["initial_delay"] = args.get("initial_delay", 1)

    # get voice
    response = eleven.voices.get_all()
    voices = {v.name: v.voice_id for v in response.voices}
    # voice_id = voices.get(args["voice"], DEFAULT_VOICE)
    voice_ids = [v.voice_id for v in response.voices]
    voice_id = args.get("voice", DEFAULT_VOICE)
    if voice_id not in voice_ids:
        # check if voice is a name
        if voice_id in voices:
            voice_id = voices[voice_id]
        else:
            raise ValueError(f"Voice ID {voice_id} not found, try another one (DEFAULT_VOICE: {DEFAULT_VOICE})")
    
    async def generate_with_params():
        audio_generator = eleven.text_to_speech.convert(
            text=args["text"],
            voice_id=voice_id,
            voice_settings={
                "stability": args["stability"],
                "style": args["style"],
                "speed": args["speed"],
                "use_speaker_boost": True, #args["use_speaker_boost"],
                # similarity_boost=args["similarity_boost"],
            },
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        return audio_generator

    audio_generator = await utils.async_exponential_backoff(
        generate_with_params,
        max_attempts=3, #args["max_attempts"],
        initial_delay=1 #args["initial_delay"],
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
    
    print(", ".join(ids))
    print(", ".join(names))
    print(full_description)

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

