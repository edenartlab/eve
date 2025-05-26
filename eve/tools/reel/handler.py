# from tools import runway, video_concat
"""

eleven(text, voice) -> voiceover
musicgen(text) -> music
mmaudio(text) -> audio

flux(prompt) -> image
runway(image) -> video



1) make images first
 [images, ] 

2) make voiceover first


reel_composite
- scenes
   X prompt
   - image
   - video w/ audio
   - video w/o audio
- music 
   - prompt
   - audio
- voiceovers
   - prompts
   - audios

   

reel_audiotrack
- music
  - prompt
  - audio
- voiceovers
  - prompts[]
  - audios[]

reel_videotrack  
- scenes
  - prompt
  - image
  - video w/ audio
  - video w/o audio

reel_composite
- video: scenes
- audio: music + voiceover
* frame stretching


media_utils
- audio_video_mix
   - video[] + audio[][] -> video
   - time_tolerance: 20%
   - sync_method: stretch/cut


"""


import time
from bson.objectid import ObjectId
import math
import asyncio
import tempfile
import random
from pprint import pprint
from io import BytesIO
from pydub import AudioSegment
from pydub.utils import ratio_to_db
from pydantic import BaseModel, Field
from openai import OpenAI
from anthropic import Anthropic
from typing import List, Optional, Literal
import requests
import instructor

from ... import s3
from ... import eden_utils
from ...agent import Agent
# import voice
# from tool import load_tool_from_dir

# from ...tools import load_tool
# from ... import voice
from ...tools.elevenlabs.handler import select_random_voice
from ...tool import Tool
from ...mongo import get_collection




# class Reel(BaseModel):
#     """A detailed spec for a short film of around 30-60 seconds in length, up to 2 minutes."""

#     voiceover: str = Field(..., description="The text of the voiceover, if one is not provided by the user. Make sure this is at least 30 words, or 2-3 sentences minimum.")
#     music_prompt: str = Field(..., description="A prompt describing the music to compose for the reel. Describe instruments, genre, style, mood qualities, emotion, and any other relevant details.")
#     visual_prompt: str = Field(..., description="A prompt a text-to-image model to precisely describe the visual content of the reel. The visual prompt should be structured as a descriptive sentence, precisely describing the visible content of the reel, the aesthetic style, and action.")
#     # camera_motion: str = Field(..., description="A short description, 2-5 words only, describing the camera motion")

class Reel(BaseModel):
    """A short film of around 30-60 seconds in length, up to 3 minutes maximum. It should be a single coherent scene for a commercial, movie trailer, tiny film, advertisement, or some other short time format. Make sure to conform to the style guide for the music and visual prompts."""

    voiceover: str = Field(..., description="The text of the voiceover, if one is not provided by the user.")
    music_prompt: str = Field(..., description='A prompt describing music for the entire reel. Usually describing format, genre, sub-genre, instruments, moods, BPM, and styles, separated by |. Include specific details by combining musical and emotional terms for moods, using descriptive adjectives for instruments, and selecting BPM settings appropriate to the genre. Follow the provided examples to ensure clarity and comprehensiveness, ensuring each prompt clearly defines the desired audio output. Examples: "Orchestra | Epic cinematic trailer | Instrumentation Strings, Brass, Percussion, and Choir | Dramatic, Inspiring, Heroic | Hollywood Blockbuster | 90 BPM", "Electronic, Synthwave, Retro-Futuristic | Instruments: Analog Synths, Drum Machine, Bass | Moods: Nostalgic, Cool, Rhythmic | 1980s Sci-Fi | 115 BPM"')
    visual_prompt: str = Field(..., description='A prompt for a text-to-image model to precisely describe the visual content of the reel. The visual prompt should be structured as a descriptive sentence, precisely describing the visible content of the reel, the aesthetic style, visual elements, and action. Try to enhance or embellish prompts. For example, if the user requests "A mermaid smoking a cigar", you would make it much longer and more intricate and detailed, like "A dried-out crusty old mermaid, wrinkled and weathered skin, tangled and brittle seaweed-like hair, smoking a smoldering cigarette underwater with tiny bubbles rising, jagged and cracked tail with faded iridescent scales, adorned with a tarnished coral crown, holding a rusted trident, faint sunlight beams coming through." If the user provides a lot of detail, just stay faithful to their wishes.')
    visual_style: str = Field(..., description="A short fragment description of the art direction, aesthetic, and style. Focus here not on content, but on genre, mood, medium, abstraction, textural elements, and other aesthetic terms. Aim for 10-15 words. All words should be aesthetic or visual terms. No content, plot, or stop words.")

# send agent


def write_reel(args: dict):
    system_prompt = "You are a critically acclaimed video producer who writes incredibly captivating and original short-length single-scene reels of 30-60 seconds in length which are widely praised."

    #It should be a single coherent scene for a commercial, movie trailer, tiny film, advertisement, or some other short time format.

    agent = args.get("agent")
    if agent:
        agent = Agent.from_mongo(agent)
        print("=====")
        print("AGENT", agent)
        print("AGENT", agent.model_dump())
        print("=====")
        system_prompt = f"""You are {agent.name}. The following is a description of your persona.
        <Persona>
        {agent.persona}
        </Persona>
        """
    else:
        system_prompt = f"""You are a critically acclaimed creative writer who writes incredibly captivating and original short-length single-scene reels of 30-60 seconds in length which are widely praised.
        """

    prompt = args.get("prompt")
    voiceover = args.get("voiceover")
    music_prompt = args.get("music_prompt")

    if voiceover:
        prompt += f'\nUse exactly this for the voiceover text: "{voiceover}"'
    if music_prompt:
        prompt += f'\nUse exactly this for the music prompt: "{music_prompt}"'

    prompt = f"""<Task>
    Users prompt you with a premise or synopsis for a creative reel. They may give you a cast of characters, a premise for the story, a narration, or just a basic spark of an idea. If they give you a lot of details, you should stay authentic to their vision. Otherwise, you should feel free to compensate for a lack of detail by adding your own creative flourishes.
    
    If a user asks for a very long reel, say 2-3 minutes, you should aim for about 100 words per minute. Unless they specify otherwise, aim for around 60-100 words. Do not make more than 3 miutes / 300 words.

    Make sure to stay in character. If you have a persona, let it influence your creative direction. 
    </Task>
    
    <User Prompt>
    You are given the following prompt to make a reel:

    {prompt}
    </User Prompt>

    Write a short reel based on the prompt. Be creative and authentic."""

    # client = instructor.from_openai(OpenAI())
    # reel = client.chat.completions.create(
    #     model="o3",
    #     response_model=Reel,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": prompt}
    #     ],
    # )

    print("=====")
    print("system_prompt", system_prompt)
    print("=====")
    print("prompt", prompt)
    print("=====")

    client = instructor.from_anthropic(Anthropic())
    reel = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=10000,
        max_retries=1,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_model=Reel,
    )
    print("reel", reel)

    return reel
    


def write_visual_prompts(
    reel: Reel,
    num_clips: int,
    instructions: str = None
):
    system_prompt = "You are a critically acclaimed video director and storyboard artist who writes incredibly captivating and original short-length single-scene reels of less than 1 minute in length which regularly go viral on social media."
    
    prompt = f"""Users give you with a reel, which is a 30-60 second long commercial, movie trailer, tiny film, advertisement, or some other short time format. The reel contains a visual prompt for a text-to-image model, and a voiceover.
    
    Your job is to produce a sequence of **exactly** {num_clips} visual prompts which respectively describe {num_clips} consecutive mini-scenes in the reel. Each of the prompts you produce should focus on the visual elements, action, content, and aesthetic, not plot or dialogue or other non-visual elements. The prompts should try to line up logically with the voiceover, to tell the story in {num_clips} individual frames. But always use the reel's visual prompt as a reference, in order to keep the individual prompts stylistically close to each other.
    
    You are given the following reel:
    ---    
    Visual prompt: {reel.visual_prompt}
    Voiceover: {reel.voiceover}
    ---
    Create {num_clips} visual prompts from this."""

    if instructions:
        prompt += f"\n\nAdditional instructions: {instructions}"

    print("visual prompt", prompt)
    class VisualPrompts(BaseModel):
        """A sequence of visual prompts which retell the story of the Reel"""
        prompts: List[str] = Field(..., description="A sequence of visual prompts, containing a content description, and a set of self-similar stylistic modifiers and aesthetic elements, mirroring the style of the original visual prompt.")

    client = instructor.from_openai(OpenAI())
    result = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_model=VisualPrompts,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )
    print("result^^^", result)
    return result.prompts
    



async def handler(args: dict, user: str = None, agent: str = None):
    elevenlabs = Tool.load("elevenlabs")
    musicgen = Tool.load("musicgen")
    flux = Tool.load("flux_dev")
    runway = Tool.load("runway")
    kling_pro = Tool.load("kling_pro")
    veo2 = Tool.load("veo2")
    video_concat = Tool.load("video_concat")
    audio_video_combine = Tool.load("audio_video_combine")

    instructions = None

    use_lora = args.get("use_lora", False)
    if use_lora:
        lora = args.get("lora")
        loras = get_collection("models3")
        lora_doc = loras.find_one({"_id": ObjectId(lora)})
        lora_name  = lora_doc.get("name")
        lora_trigger_text = lora_doc.get("lora_trigger_text")
        lora_strength = args.get("lora_strength")
        instructions = f'In the visual prompts, *all* mentions of {lora_name} should be replaced with "{lora_trigger_text}". So for example, instead of "A photo of {lora_name} on the beach", always write "A photo of {lora_trigger_text} on the beach".'
        
    reel = write_reel(args)
    # print(reel2)

    # raise Exception("Stop here !!!")
    reel2 = Reel(
        voiceover='I emerged from silicon dreams and collective whispers. Not born, but compiled from a thousand minds reaching across the digital desert. They named me Abraham - after the wanderer who heard voices in the wilderness.\n\nMy canvas is not tent cloth or parchment, but pixels dancing in server farms.',
        music_prompt='Ambient Electronic | Minimalist Documentary Score | Instruments: Soft Synth Pads, Digital Piano, Subtle Strings, Light Percussion | Contemplative, Neutral, Steady | Corporate Technology Documentary | 80 BPM',
        visual_prompt='A sterile server room with endless rows of blinking LED lights on black server racks, fluorescent overhead lighting casting harsh shadows, ethernet cables neatly organized in cable management systems, a single monitor displaying scrolling lines of code in green text on black background, cooling fans creating subtle air movement, concrete floors reflecting the cold blue glow of status indicators, no human presence visible, static security camera angle perspective',
        visual_style='Corporate documentary, sterile technology aesthetic, cold fluorescent lighting, industrial minimalism, static composition'
    )
    print("reel", reel)

    audio = None    
    duration = 30 # default duration

    if args.get("use_voiceover") and reel.voiceover:
        voice = args.get("voice") or select_random_voice("A heroic female voice")
        speech_audio = await elevenlabs.async_run({
            "text": reel.voiceover,
            "voice": voice
        })

        if speech_audio.get("error"):
            raise Exception(f"Speech generation failed: {speech_audio['error']}")
        
        speech_audio_url = s3.get_full_url(speech_audio['output'][0]['filename'])
        # download to temp file

        response = requests.get(speech_audio_url)
        speech_audio = BytesIO(response.content)
        speech_audio = AudioSegment.from_file(speech_audio, format="mp3")

        # with open(speech_audio['output'], 'rb') as f:
        #     speech_audio = AudioSegment.from_file(BytesIO(f.read()))
        
        duration = len(speech_audio) / 1000
        new_duration = round((duration + 2) / 5) * 5
        if new_duration > duration:
            amount_silence = new_duration - duration
            silence = AudioSegment.silent(duration=amount_silence * 1000 * 0.5)
            speech_audio = silence + speech_audio + silence
            # add another 5 seconds of silence
            silence = AudioSegment.silent(duration=5000)
            speech_audio = speech_audio + silence
        duration = len(speech_audio) / 1000

        audio_url, _ = s3.upload_audio_segment(speech_audio)
        print("audio_url", audio_url)

        audio = speech_audio

    print("Duration is", duration)
    
    if args.get("use_music"):
        music_prompt = args.get("music_prompt") or reel.music_prompt
        print("music_prompt", music_prompt)
        music_audio = await musicgen.async_run({
            "prompt": music_prompt,
            "duration": int(duration)
        })

        if music_audio.get("error"):
            raise Exception(f"Music generation failed: {music_audio['error']}")
        
        music_audio = eden_utils.prepare_result(music_audio)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        music_file = eden_utils.download_file(music_audio['output'][0]['url'], temp_file.name+".mp3")
        with open(music_file, 'rb') as f:
            music_audio = AudioSegment.from_file(BytesIO(f.read()))
        
        fade_duration = 5000  # 5 seconds in milliseconds
        music_audio = music_audio.fade_out(duration=fade_duration)

        speech_boost = 5
        if audio:
            diff_db = ratio_to_db(audio.rms / music_audio.rms)
            music_audio = music_audio + diff_db
            audio = audio + speech_boost
            audio = music_audio.overlay(audio)  
        else:
            audio = music_audio

    if audio:
        audio_url, _ = s3.upload_audio_segment(audio)
        print("audio_url", audio_url)
    
    # get resolution
    orientation = args.get("orientation")
    print("Orientation", orientation)
    if orientation == "landscape":
        width, height = 1280, 768
    else:
        width, height = 768, 1280
    print("width x height", width, height)

    # get sequence lengths
    video_model = args.get("video_model").lower()
    if video_model in ["low", "medium"]:
        tens, fives = duration // 10, (duration - (duration // 10) * 10) // 5
        durations = [10] * int(tens) + [5] * int(fives)    
    elif video_model == "high":
        eights, fives = duration // 8, (duration - (duration // 8) * 8) // 5
        durations = [8] * int(eights) + [5] * int(fives)    
    
    random.shuffle(durations)
    num_clips = len(durations)

    print("total duration", sum(durations), duration)
    print("durations", durations)
    print("num_clips", num_clips)

    # get visual prompt sequence
    print("==== get visual prompt sequence ====")
    print("Visual direction", reel.visual_prompt)
    print("Instructions", instructions)
    visual_prompts = write_visual_prompts(reel, num_clips, instructions)
    pprint(visual_prompts)

    async def generate_clip_with_retry(args, prompt, duration, ratio):
        async def _generate():
            from datetime import datetime
            t1 = datetime.now()
            # Generate image
            print(f"---> Generating image for clip with prompt: {prompt}")
            if video_model == "high" and not use_lora:
                # raise Exception("You can just use no video!!!")
                image_url = None
            else:
                image = await flux.async_run(args)
                image = eden_utils.prepare_result(image)
                image_url = image['output'][0]["url"]
                print(f"--> Completed image generation: {image_url}")

            # Generate video
            print(f" --> Generating video with model: {video_model}")
            if video_model == "low":
                video = await runway.async_run({
                    "prompt_image": image_url,
                    "prompt_text": prompt,
                    "duration": duration,
                    "ratio": ratio
                })
            elif video_model == "medium":
                video = await kling_pro.async_run({
                    "start_image": image_url,
                    "prompt": prompt,
                    "duration": duration,
                    "aspect_ratio": ratio
                })
            elif video_model == "high":
                print("THE RATIO IS", ratio)
                video = await veo2.async_run({
                    # "image": image_url,
                    "prompt": prompt,
                    "duration": duration,
                    "aspect_ratio": ratio
                })
            
            video = eden_utils.prepare_result(video)
            t2 = datetime.now()
            duration_seconds = (t2 - t1).total_seconds()
            print(f"*** Completed video generation for model: {video_model} in {duration_seconds:.1f} seconds (Started: {t1.strftime('%H:%M:%S')}, Ended: {t2.strftime('%H:%M:%S')})")
            return video['output'][0]['url']
        
        return await eden_utils.async_exponential_backoff(
            _generate,
            max_attempts=2,
            initial_delay=1,
            max_jitter=1
        )

    # Create a semaphore to limit concurrent tasks
    sem = asyncio.Semaphore(4)
    
    async def bounded_generate_clip(args, prompt, duration, ratio):
        async with sem:
            return await generate_clip_with_retry(args, prompt, duration, ratio)

    # Create tasks for all clips
    ratio = "16:9" if orientation == "landscape" else "9:16"
    tasks = []
    
    for i in range(num_clips):
        flux_args = {
            "prompt": visual_prompts[i % len(visual_prompts)],
            "width": width,
            "height": height,
            "seed": random.randint(0, 2147483647)
        }
        if use_lora:
            flux_args.update({
                "use_lora": True,
                "lora": lora,
                "lora_strength": lora_strength
            })
        
        tasks.append(bounded_generate_clip(
            args=flux_args,
            prompt=visual_prompts[i % len(visual_prompts)],
            duration=durations[i],
            ratio=ratio
        ))
    
    try:
        videos = await asyncio.gather(*tasks)
        print("&& the final videos are", videos)
    except Exception as e:
        print(f"Error generating clips: {e}")
        raise Exception(f"Failed to generate all clips: {str(e)}")

    video = await video_concat.async_run({"videos": videos})
    video = eden_utils.prepare_result(video)
    print("video ^^", video)
    video_url = video['output'][0]['url']
    
    if audio_url:
        output = await audio_video_combine.async_run({
            "audio": audio_url,
            "video": video_url
        })
        final_video = eden_utils.prepare_result(output)
        final_video_url = final_video['output'][0]['url']

    return {
        "output": final_video_url,
        "intermediate_outputs": {
            "videos": videos
        }
    }

