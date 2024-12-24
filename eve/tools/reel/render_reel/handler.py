import math
import asyncio
import tempfile
import json
import random
from pprint import pprint
from io import BytesIO
from pydub import AudioSegment
from pydub.utils import ratio_to_db
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List, Optional, Literal
import requests
import instructor



from pydub import AudioSegment
from pydub.utils import ratio_to_db
from io import BytesIO

from .... import llm
from ....thread import UserMessage
from ....agent import Agent
from ..common import ReelStoryboard

from ....tools.elevenlabs import handler as elevenlabs
from ....tool import Tool
from .... import s3, eden_utils

from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

INSTRUCTIONS = """You are creating a storyboard or comprehensive description of a short film or “Reel” of generally 1 to 2 minutes long, in the schema given to you. This will be used to produce a final video.

Some guidelines:
- Avoid layering an overall reel voiceover on top of scene-level voiceovers unless there is a good reason to have both.
- Each video clip or scene can include:
  - A visual description of what is happening.
  - Camera motion details, if important.
  - Foley/sound effects details, if any.
  - Vocals, which can be multiple lines or multiple speakers.
- Create characters only when they are named. Do not include NPCs or extras.

Your Task:
- You will receive a user prompt describing an idea for a reel. It may be vague or highly detailed, or a mix of both.
- A user may just give you a premise or bare idea for a story, in which case you should make something short and simple, unless they ask you to be creative with the details.
- If they give you a lot of details, you should stay as authentic and faithful to the request as possible.
- The length of the overall voiceover if there is one, should not exceed the sum total duration of the video across all scenes, at approximately 20-30 words per 10 seconds of video (150-180 words per minute). If there is an overall voiceover, aim for at least half the total duration."""

DEFAULT_SYSTEM_MESSAGE = """You are an advanced AI that creates captivating storyboards and short films, authentically to a user's desire or vision."""


from eve.base import VersionableBaseModel, generate_edit_model, apply_edit

# async def handler(args: dict, db: str):
agent_name = None   # args.get("agent_name")
# prompt = args.get("prompt")

if agent_name:
    # agent = Agent.load(agent_name, db=db)
    # system_message = agent.description or DEFAULT_SYSTEM_MESSAGE
    pass
else:
    system_message = DEFAULT_SYSTEM_MESSAGE






# reel = await llm.async_prompt(
#     messages=[
#         UserMessage(content=INSTRUCTIONS),
#         UserMessage(content=prompt)
#     ],
#     system_message=system_message,
#     model="claude-3-5-sonnet-20241022",
#     response_model=ReelStoryboard
# )
# import json
# print(json.dumps(reel.model_dump(), indent=4))

# return {
#     "output": reel.model_dump()
# }








# raise Exception("stop")
# return {
#     "output": reel.model_dump()
# }


# @retry(
#     stop=stop_after_attempt(2),
#     wait=wait_exponential(multiplier=1, min=5, max=10),
#     reraise=True
# )
async def generate_single_video(args, db):
    return "https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/7b4da1c185a0146b5ff583d068f14a54a71a8bc729f5d38f588a0e446d8efe14.mp4"
    runway = Tool.load("runway", db=db)
    video = await runway.async_run(args, db=db)
    return video['output'][0]["url"]

async def generate_video(args, db):
    try:
        return await generate_single_video(args, db)
    except RetryError:
        raise Exception(f"Failed to generate image after 3 attempts with args: {args}")




# @retry(
#     stop=stop_after_attempt(2),
#     wait=wait_exponential(multiplier=1, min=5, max=10),
#     reraise=True
# )
async def generate_single_image(args, db):
    print("kles go!", args)
    flux = Tool.load("flux_dev_lora", db=db)
    # return "https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/0322f959d789720089bbb27582bff6b6c3604da4bc0ffdc9c7fe5e43306621e4.png"
    image = await flux.async_run(args, db=db)
    image = eden_utils.prepare_result(image, db=db)
    return image['output'][0]["url"]

async def generate_image(args, db):
    try:
        return await generate_single_image(args, db)
    except RetryError:
        raise Exception(f"Failed to generate image after 3 attempts with args: {args}")



def generate_reel_data(db: str):
    flux = Tool.load("flux_dev", db=db)
    return flux.run({
        "prompt": "generate a reel storyboard",
        "model": "gpt-4o-mini"
    })

async def handler(args: dict, db: str):

    reel_data = {"aspect_ratio": "16:9", "characters": [{"name": "abraham", "description": "an old grizzled mans voice", "visual_description": "a visual description of abraham"}], "title": "Desert Mirages", "brief": "A surreal and poetic journey through a desert encounter where three mysterious figures bring transformation and magic to a nomadic tent.", "characters": [{"name": "Mirage Beings", "description": "Three ethereal figures that bend light and reality around them, bringing transformation to the desert.", "visual_description": "Three tall, crystalline humanoid figures with translucent, light-refracting bodies that shimmer and shift"}, {"name": "Desert Woman", "description": "A mysterious woman whose presence bridges the real and surreal within the tent.", "visual_description": "A graceful silhouette in flowing robes, her form visible through translucent tent fabric"}], "clips": [{"scene_description": "Wide shot of an empty desert landscape with heat waves rippling across golden dunes. Three tall, prismatic figures emerge from the mirages, their bodies refracting light like living crystals.", "camera_motion": "slow push in", "sound_effects": "whispered wind, crystalline chimes", "vocals": None, "duration_in_seconds": 15}, {"scene_description": "Inside a sun-lit tent, multiple impossible shadows of the three figures dance across the fabric walls, their movements creating kaleidoscopic patterns.", "camera_motion": "slow circular pan", "sound_effects": "fabric rustling, distant wind chimes", "vocals": None, "duration_in_seconds": 12}, {"scene_description": "Close-up shots of hands breaking bread in perfect synchronization, the bread giving off steam that forms mysterious symbols in the air. Earthen vessels of milk swirl with iridescent patterns.", "camera_motion": None, "sound_effects": "soft bread breaking, liquid swirling", "vocals": None, "duration_in_seconds": 15}, {"scene_description": "The Desert Woman's silhouette appears behind the tent fabric, her shadow splitting into multiple layers that ripple and dance as she moves, each layer showing a different gesture of joy.", "camera_motion": "gentle sway", "sound_effects": "ethereal laughter, fabric rustling", "vocals": None, "duration_in_seconds": 18}, {"scene_description": "Time-lapse of desert plants erupting from sand around the tent, flowers blooming in accelerated motion while the three figures gradually dissolve into pure light, their prismatic essence merging with the blooming landscape.", "camera_motion": "slow upward tilt", "sound_effects": "accelerated growth sounds, crystalline shimmer", "vocals": None, "duration_in_seconds": 30}], "overall_voiceover": {"speaker": "abraham", "text": "there is a voiceover"}, "music_prompt": "Format: Ambient World | Genre: Ethereal | Sub-genre: Middle Eastern Fusion | Instruments: Oud, Singing Bowls, Atmospheric Synths | Moods: Mystical, Dreamy, Transcendent | Styles: Desert Dawn | BPM: 70", "visual_prompt": "Ethereal magical realism | sun-bleached color palette | prismatic light effects | dreamy soft focus | Arabian aesthetics", "target_length_in_seconds": 90}

    # reel_data = reel.model_dump()


    reel = ReelStoryboard(**reel_data)

    reel_data = json.dumps(reel.model_dump(), indent=2)

    # round target duration up to nearest 5 seconds
    target_duration = reel.target_length_in_seconds or 30
    target_duration = round(target_duration / 5) * 5
    
    overall_voiceover = None

    if False and reel.overall_voiceover:
        from ....tools import select_random_voice

        speaker = reel.overall_voiceover.speaker
        text = reel.overall_voiceover.text

        character = reel.characters[0]
        assert speaker == character.name

        voice = select_random_voice(character.description)
        speech_audio = await elevenlabs.handler({
            "text": text,
            "voice_id": voice
        }, db=db)

        print("speech_audio", speech_audio)

        if speech_audio.get("error"):
            raise Exception(f"Speech generation failed: {speech_audio['error']}")
        
        # with open(speech_audio['output'], 'rb') as f:
        #     speech_audio = AudioSegment.from_file(BytesIO(f.read()))
        
        # duration = len(speech_audio) / 1000
        # print("duration", duration)
        
        # min_silence_duration = 2
        # new_duration = round((duration + min_silence_duration) / 5) * 5
        # target_duration = max(target_duration, new_duration)
        # target_duration = min(target_duration, 500)  # musicgen
        
        
        # amount_silence = target_duration - duration
        
        
        # if new_duration > duration:
        #     amount_silence = new_duration - duration
        #     silence = AudioSegment.silent(duration=amount_silence * 1000 * 0.5)
        #     speech_audio = silence + speech_audio + silence
        # duration = len(speech_audio) / 1000

        # audio_url, _ = s3.upload_audio_segment(speech_audio)
        # print("audio_url", audio_url)

        # audio = speech_audio


    if False and reel.music_prompt:
        musicgen = Tool.load("musicgen", db=db)
        stable_audio = Tool.load("stable_audio", db=db)
        print("MUSIC GEN 111", reel.music_prompt)
        
        music_audio = await stable_audio.async_run({
            "prompt": reel.music_prompt,
            "duration": 90 #int(duration)
        }, db=db)

        print("MUSIC GEN 555", music_audio)
        if music_audio.get("error"):
            raise Exception(f"Music generation failed: {music_audio['error']}")
        
        # music_audio = eden_utils.prepare_result(music_audio, db=db)
        # print("MUSIC AUDIO 55", music_audio)

        
        # temp_file = tempfile.NamedTemporaryFile(delete=False)
        # music_file = eden_utils.download_file(music_audio['output'][0]['url'], temp_file.name+".mp3")
        # print("MUSIC FILE 77", temp_file.name)
        # with open(music_file, 'rb') as f:
        #     music_audio = AudioSegment.from_file(BytesIO(f.read()))
        # #os.remove(temp_file.name)
        # print("MUSIC AUDIO 66", music_audio)
        # print("MUSIC AUDIO 66 LENGTH", temp_file.name)

        # speech_boost = 5
        # if audio:
        #     diff_db = ratio_to_db(audio.rms / music_audio.rms)
        #     music_audio = music_audio + diff_db
        #     audio = audio + speech_boost
        #     audio = music_audio.overlay(audio)  
        # else:
        #     audio = music_audio

        # if audio:
        #     audio_url1, _ = s3.upload_audio_segment(audio)

        # music_audio = await musicgen.async_run({
        #     "prompt": reel.music_prompt,
        #     "duration": 90 #int(duration)
        # }, db=db)
        # # music_audio = {'output': {'mediaAttributes': {'mimeType': 'audio/mpeg', 'duration': 20.052}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/430eb06b9a9bd66bece456fd3cd10f8c6d99fb75c1d05a1da6c317247ac171c6.mp3'}, 'status': 'completed'}


    import copy
    for clip in reel.clips[:1]:
        print("CLIP", clip)


    # width, height = 1024, 1024
    # if reel.aspect_ratio == "16:9":
    #     width, height = 1280, 768
    # elif reel.aspect_ratio == "9:16":
    #     width, height = 768, 1280
    

    flux_args = [{
        "prompt": f"{clip.scene_description}, {reel.visual_prompt}",
        "aspect_ratio": reel.aspect_ratio,
        "seed": random.randint(0, 2147483647)
    } for clip in reel.clips][:2]

    # if use_lora:
    #     flux_args.update({
    #         "use_lora": True,
    #         "lora": lora,
    #         "lora_strength": lora_strength
    #     })

    images = await asyncio.gather(*[
        generate_image(copy.deepcopy(args), db) for args in flux_args
    ])

    print("IMAGES", images)






    
    runway_args = [{
        "prompt_image": image_url,
        "prompt_text": f"{clip.camera_motion}, {clip.scene_description}",
        "duration": str(min(10, round((clip.duration_in_seconds or 5) / 5) * 5)),
        "ratio": reel.aspect_ratio
    } for image_url, clip in zip(images, reel.clips)]

    videos = await asyncio.gather(*[
        generate_video(copy.deepcopy(args), db) for args in runway_args
    ])

    print("VIDEOS", videos)



    # if audio:
    #     audio_url, _ = s3.upload_audio_segment(audio)
    
    # print("audio_url", audio_url)



    # print(reel)




"""




"""