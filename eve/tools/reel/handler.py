import os
import time
import logging
from elevenlabs.client import ElevenLabs
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
import uuid

from ... import s3
from ... import utils
from ...agent import Agent
# import voice
# from tool import load_tool_from_dir

# from ...tools import load_tool
# from ... import voice
from ...tools.elevenlabs.handler import select_random_voice
from ...tool import Tool, ToolContext
from ...mongo import get_collection
from loguru import logger

# Suppress verbose httpx logging - only show warnings and errors
logging.getLogger("httpx").setLevel(logging.WARNING)

NUM_PARALLEL_GENERATIONS = 2


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


def write_reel(context.args: dict, agent: Agent = None):
    system_prompt = "You are a critically acclaimed video producer who writes incredibly captivating and original short-length single-scene reels of 30-60 seconds in length which are widely praised."

    #It should be a single coherent scene for a commercial, movie trailer, tiny film, advertisement, or some other short time format.

    if agent:
        system_prompt = f"""You are {agent.name}. The following is a description of your persona.
        <Persona>
        {agent.persona}
        </Persona>
        """
    else:
        system_prompt = f"""You are a critically acclaimed creative writer who writes incredibly captivating and original short-length single-scene reels of 30-60 seconds in length which are widely praised.
        """

    prompt = context.args.get("prompt")
    voiceover = context.args.get("voiceover")
    music_prompt = context.args.get("music_prompt")

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

    client = instructor.from_anthropic(Anthropic())
    reel = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=3000,
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

    class VisualPrompts(BaseModel):
        """A sequence of visual prompts which retell the story of the Reel"""
        prompts: List[str] = Field(..., description="A sequence of visual prompts, containing a content description, and a set of self-similar stylistic modifiers and aesthetic elements, mirroring the style of the original visual prompt.")

    # client = instructor.from_openai(OpenAI())
    # result = client.chat.completions.create(
    #     model="gpt-4o-2024-08-06",
    #     response_model=VisualPrompts,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": prompt}
    #     ],
    # )
    client = instructor.from_anthropic(Anthropic())
    result = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=3000,
        max_retries=1,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_model=VisualPrompts,
    )
    
    return result.prompts
    



async def handler(context: ToolContext):
    elevenlabs = Tool.load("elevenlabs")
    musicgen = Tool.load("musicgen")
    flux = Tool.load("flux_dev")
    runway = Tool.load("runway")
    kling_pro = Tool.load("kling_pro")
    veo2 = Tool.load("veo2")
    video_concat = Tool.load("video_concat")
    audio_video_combine = Tool.load("audio_video_combine")

    instructions = None

    use_lora = context.args.get("use_lora", False)
    if use_lora:
        lora = context.args.get("lora")
        loras = get_collection("models3")
        lora_doc = loras.find_one({"_id": ObjectId(lora)})
        lora_name  = lora_doc.get("name")
        lora_trigger_text = lora_doc.get("lora_trigger_text")
        lora_strength = context.args.get("lora_strength")
        instructions = f'In the visual prompts, *all* mentions of {lora_name} should be replaced with "{lora_trigger_text}". So for example, instead of "A photo of {lora_name} on the beach", always write "A photo of {lora_trigger_text} on the beach".'

    agent = context.args.get("agent")
    if agent:
        agent = Agent.from_mongo(context.agent)

    reel = write_reel(context.args, agent)

    audio = None    
    duration = 30 # default duration

    if context.args.get("use_voiceover") and reel.voiceover:
        
        # if voice is provided, use it
        if context.args.get("voice"):
            voice = context.args.get("voice")
        # otherwise, if agent has a voice, use it
        elif agent and agent.voice:
            eleven = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
            try:
                voice = eleven.voices.get(agent.voice)
                voice = voice.id
            except Exception as e:
                logger.error(f"Error getting voice: {e}")
                voice = select_random_voice("Voice of a narrator")
        # otherwise, select a random voice
        else:
            voice = select_random_voice("Voice of a narrator")

        
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

        audio = speech_audio

    
    if context.args.get("use_music"):
        music_prompt = context.args.get("music_prompt") or reel.music_prompt
        music_audio = await musicgen.async_run({
            "prompt": music_prompt,
            "duration": int(duration)
        })

        if music_audio.get("error"):
            raise Exception(f"Music generation failed: {music_audio['error']}")
        
        music_audio = utils.prepare_result(music_audio)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        music_file = utils.download_file(music_audio['output'][0]['url'], temp_file.name+".mp3")
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
    
    # get resolution
    orientation = context.args.get("orientation")
    if orientation == "landscape":
        width, height = 1280, 768
    else:
        width, height = 768, 1280

    # get sequence lengths
    video_model = context.args.get("video_model").lower()
    if video_model in ["low", "medium"]:
        tens, fives = duration // 10, (duration - (duration // 10) * 10) // 5
        durations = [10] * int(tens) + [5] * int(fives)    
    elif video_model == "high":
        eights, fives = duration // 8, (duration - (duration // 8) * 8) // 5
        durations = [8] * int(eights) + [5] * int(fives)    
    
    random.shuffle(durations)
    num_clips = len(durations)
    durations = durations[:num_clips]

    # get visual prompt sequence
    visual_prompts = write_visual_prompts(reel, num_clips, instructions)
    

    async def generate_clip_with_retry(context.args, prompt, duration, ratio):
        async def _generate():
            from datetime import datetime
            from eve.agent.session.models import LLMContextMetadata, LLMTraceMetadata
            
            # Create trace metadata for this clip generation
            trace_metadata = LLMTraceMetadata(
                user_id=None,  # We don't have user context here
                agent_id=None,  # We don't have agent context here
                session_id=str(uuid.uuid4()),  # Generate unique session ID for this clip
            )

            # Create LLM context metadata
            metadata = LLMContextMetadata(
                session_id=str(uuid.uuid4()),
                trace_name="clip_generation",
                # trace_id=f"clip_generation_{trace_metadata.session_id}",
                generation_name="clip_generation",
                # generation_id=f"clip_generation_{trace_metadata.session_id}",
                trace_metadata=trace_metadata
            )
            
            t1 = datetime.now()
            # Generate image
            
            if video_model == "high" and not use_lora:
                image_url = None
            else:
                image = await flux.async_run(context.args)
                image = utils.prepare_result(image)
                image_url = image['output'][0]["url"]

            # Generate video with fallback logic
            fallback_map = {
                "low": "medium",     # runway -> kling_pro
                "medium": "low",     # kling_pro -> runway  
                "high": "medium"     # veo2 -> kling_pro
            }
            
            async def generate_video_with_model(model_type, prompt, duration, ratio, image_url):
                """Generate video with specified model type"""
                
                if model_type == "low":
                    return await runway.async_run({
                        "start_image": image_url,
                        "prompt_text": prompt,
                        "duration": duration if duration in [10, 5] else 10,
                        "ratio": ratio
                    })
                elif model_type == "medium":
                    return await kling_pro.async_run({
                        "start_image": image_url,
                        "prompt": prompt,
                        "duration": duration if duration in [10, 5] else 10,
                        "aspect_ratio": ratio
                    })
                elif model_type == "high":
                    if image_url:
                        return await veo2.async_run({
                            "image": image_url,
                            "prompt": prompt,
                            "duration": duration if duration < 8 else 8,
                            "aspect_ratio": ratio
                        })
                    else:
                        return await veo2.async_run({
                            "prompt": prompt,
                            "duration": duration if duration < 8 else 8,
                            "aspect_ratio": ratio
                        })
            
            # Try primary model, fallback to alternative if it fails
            video = None
            current_model = video_model
            
            for attempt in range(2):  # Try primary, then fallback
                video = await generate_video_with_model(current_model, prompt, duration, ratio, image_url)
                
                # Check if video generation was successful
                if video and "output" in video and video["output"]:
                    break
                else:
                    error_msg = video.get("error", "Unknown error") if video else "No response"
                    
                    if attempt == 0 and current_model in fallback_map:
                        current_model = fallback_map[current_model]
                    else:
                        raise Exception(f"All video generation attempts failed. Last error: {error_msg}")
            
            if not video or "output" not in video:
                raise Exception("Video generation failed for unknown reasons")

            video = utils.prepare_result(video)
            t2 = datetime.now()
            duration_seconds = (t2 - t1).total_seconds()
            
            # Log with proper session context
            try:
                return video['output'][0]['url']
            except Exception as e:
                logger.error(f"Error preparing video: {e}")
                raise Exception(f"Failed to generate video: {str(e)}")
        
        return await utils.async_exponential_backoff(
            _generate,
            max_attempts=2,
            initial_delay=1,
            max_jitter=1
        )

    # Create a semaphore to limit concurrent tasks
    sem = asyncio.Semaphore(NUM_PARALLEL_GENERATIONS)
    
    async def bounded_generate_clip(context.args, prompt, duration, ratio):
        async with sem:
            return await generate_clip_with_retry(context.args, prompt, duration, ratio)

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
    except Exception as e:
        logger.error(f"Error generating clips: {e}")
        raise Exception(f"Failed to generate all clips: {str(e)}")

    video = await video_concat.async_run({"videos": videos})
    video = utils.prepare_result(video)
    video_url = video['output'][0]['url']
    
    if audio_url:
        output = await audio_video_combine.async_run({
            "audio": audio_url,
            "video": video_url
        })
        final_video = utils.prepare_result(output)
        final_video_url = final_video['output'][0]['url']

    return {
        "output": final_video_url,
        "intermediate_outputs": {
            "videos": videos
        }
    }

