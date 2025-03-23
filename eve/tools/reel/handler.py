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
from typing import List, Optional, Literal
import requests
import instructor

from ... import s3
from ... import eden_utils
# import voice
# from tool import load_tool_from_dir

# from ...tools import load_tool
# from ... import voice
from ...tools.elevenlabs.handler import select_random_voice
from ...tool import Tool
from ...mongo import get_collection




class Character(BaseModel):
    name: str = Field(..., description="The name of the character")
    description: str = Field(..., description="A short description of the character")


def extract_characters(prompt: str):
    client = instructor.from_openai(OpenAI())
    characters = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_model=Optional[List[Character]],
        messages=[
            {
                "role": "system",
                "content": "Extract and resolve a list of characters/actors/people from the following story premise. Do not include inanimate objects, places, or concepts. Only named or nameable characters.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )    
    return characters or []
    

def prompt_variations(prompt: str, n: int):
    client = instructor.from_openai(OpenAI())

    class PromptVariations(BaseModel):
        prompts: List[str] = Field(..., description="A unique variation of the original prompt")

    user_message = f"You are given the following prompt for a short-form video: {prompt}. Generate EXACTLY {n} variations of this prompt. Don't get too fancy or creative, just state the same thing in different ways, using synonyms or different phrase constructions."
    client = instructor.from_openai(OpenAI())
    prompts = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_model=PromptVariations,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who generates variations of a prompt for a short-form video.",
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )
    print("PROMPTS", prompts)
    return prompts.prompts



def write_reel23(
    prompt: str, 
    characters: List[Character],
    narration: str,
    music: bool,
    music_prompt: str
):
    
    if characters or narration:
        names = [c.name for c in characters]
        speaker_type, speaker_description = Literal[*names], "Name of the speaker, if any voiceover."
        speech_type, speech_description = str, "If there is a voiceover, the text of the speech."
    else:
        speaker_type, speaker_description = Optional[None], "Leave this blank since there are no speakers."
        speech_type, speech_description = Optional[None], "Leave this blank since there are no speakers."

    if music:
        music_type, music_description = str, "A short and concise 1-sentence description of the music for the reel, structured as a prompt. Use descriptive words to convey the mood and genre of the music."
    else:
        music_type, music_description = Optional[None], "Leave this blank since there is no music."
    
    class Reel(BaseModel):
        image_prompt: str = Field(..., description="A short and concise 1-sentence description of the visual content for the reel, structured as a prompt, focusing on visual elements and action, not plot or dialogue")
        music_prompt: music_type = Field(..., description=music_description)
        speaker: speaker_type = Field(..., description=speaker_description)
        speech: speech_type = Field(..., description=speech_description)

    system_prompt = f"""You are a critically acclaimed screenwriter who writes incredibly captivating and original short-length single-scene reels of less than 1 minute in length which regularly go viral on Instagram, TikTok, Netflix, and YouTube.
    
    Users will prompt you with a premise or synopsis for a reel, as well as optionally a cast of characters, including their names and biographies.
    
    You will then write a script for a reel based on the information provided.
    
    Do not include an introduction or restatement of the prompt, just go straight into the reel itself."""

    client = instructor.from_openai(OpenAI())
    
    reel = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_model=Reel,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )

    # override music prompt if provided by user
    if music and music_prompt:
        reel.music_prompt = music_prompt

    if narration:
        reel.speech = narration

    return reel
    









class Reel(BaseModel):
    """A reel is a short film of 30-60 seconds in length. It should be a single coherent scene for a commercial, movie trailer, tiny film, advertisement, or some other short time format."""

    voiceover: str = Field(..., description="The text of the voiceover, if one is not provided by the user. Make sure this is at least 30 words, or 2-3 sentences minimum.")
    music_prompt: str = Field(..., description="A prompt describing the music to compose for the reel. Describe instruments, genre, style, mood qualities, emotion, and any other relevant details.")
    visual_prompt: str = Field(..., description="A prompt a text-to-image model to precisely describe the visual content of the reel. The visual prompt should be structured as a descriptive sentence, precisely describing the visible content of the reel, the aesthetic style, and action.")
    # camera_motion: str = Field(..., description="A short description, 2-5 words only, describing the camera motion")








def write_reel(
    prompt: str,
    voiceover: str = None,
    music_prompt: str = None,
):
    system_prompt = "You are a critically acclaimed video director who writes incredibly captivating and original short-length single-scene reels of 30-60 seconds in length which regularly go viral on social media."
    print("make the reel !!!\n\n")
    if voiceover:
        prompt += f'\nUse this for the voiceover text: "{voiceover}"'
    if music_prompt:
        prompt += f'\nUse this for the music prompt: "{music_prompt}"'

    prompt = f"""Users prompt you with a premise or synopsis for a reel. They may give you a cast of characters, a premise for the story, a narration, or just a basic spark of an idea. If they give you a lot of details, you should stay authentic to their vision. Otherwise, you should feel free to compensate for a lack of detail by adding your own creative flourishes. Make sure the voiceover is at least 30 words, or 2-3 sentences minimum.
    
    You are given the following prompt to make a short reeL:
    ---    
    {prompt}
    ---
    Create a short reel based on the prompt."""

    class Reel(BaseModel):
        """A reel is a short film of 30-60 seconds in length. It should be a single coherent scene for a commercial, movie trailer, tiny film, advertisement, or some other short time format. Make sure to conform to the style guide for the music and visual prompts."""

        voiceover: str = Field(..., description="The text of the voiceover, if one is not provided by the user.")
        music_prompt: str = Field(..., description='A prompt describing music for the entire reel. Usually describing format, genre, sub-genre, instruments, moods, BPM, and styles, separated by |. Include specific details by combining musical and emotional terms for moods, using descriptive adjectives for instruments, and selecting BPM settings appropriate to the genre. Follow the provided examples to ensure clarity and comprehensiveness, ensuring each prompt clearly defines the desired audio output. Examples: "Orchestra | Epic cinematic trailer | Instrumentation Strings, Brass, Percussion, and Choir | Dramatic, Inspiring, Heroic | Hollywood Blockbuster | 90 BPM", "Electronic, Synthwave, Retro-Futuristic | Instruments: Analog Synths, Drum Machine, Bass | Moods: Nostalgic, Cool, Rhythmic | 1980s Sci-Fi | 115 BPM"')
        visual_prompt: str = Field(..., description='A prompt for a text-to-image model to precisely describe the visual content of the reel. The visual prompt should be structured as a descriptive sentence, precisely describing the visible content of the reel, the aesthetic style, visual elements, and action. Try to enhance or embellish prompts. For example, if the user requests "A mermaid smoking a cigar", you would make it much longer and more intricate and detailed, like "A dried-out crusty old mermaid, wrinkled and weathered skin, tangled and brittle seaweed-like hair, smoking a smoldering cigarette underwater with tiny bubbles rising, jagged and cracked tail with faded iridescent scales, adorned with a tarnished coral crown, holding a rusted trident, faint sunlight beams coming through." If the user provides a lot of detail, just stay faithful to their wishes.')
        visual_style: str = Field(..., description="A short fragment description of the art direction, aesthetic, and style. Focus here not on content, but on genre, mood, medium, abstraction, textural elements, and other aesthetic terms. Aim for 10-15 words")
        # camera_motion: str = Field(..., description="A short description, 2-5 words only, describing the camera motion")


    # return Reel(
    #     voiceover='In the heart of a hidden forest, Verdelis stumbled upon a realm where reality twisted into magic. Her eyes widened at the sight of a mystical creature, shimmering with ethereal elegance, its eyes holding ancient secrets and untold stories. In this moment, the ordinary paused, and an extraordinary bond was born.', music_prompt='A mystical, enchanting orchestral piece with soft strings and ethereal woodwinds, creating a sense of wonder and discovery. The music is gentle and flowing, capturing the magical atmosphere of the forest encounter.', visual_prompt="A serene, enchanted forest with dappled sunlight filtering through lush green leaves. The scene shows Verdelis, a young adventurer dressed in earth-toned attire, floating gracefully through the trees. She encounters a mystical creature—a unicorn-like being with shimmering iridescent skin and an elegant presence. The forest is vibrant with colors, and there's a magical aura surrounding the creature, creating an ethereal glow that illuminates the scene, capturing a moment of awe and wonder."
    # )

    client = instructor.from_openai(OpenAI())
    reel = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_model=Reel,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
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
        
    reel = write_reel(
        prompt=args.get("prompt"),
        voiceover=args.get("voiceover"),
        music_prompt=args.get("music_prompt"),
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


    print("THE DURATION IS", duration)
    

    if args.get("use_music"):
        print("music_prompt", args.get("music_prompt"))
        music_prompt = args.get("music_prompt") or reel.music_prompt
        print("music_prompt", music_prompt)
        print("run")
        music_audio = await musicgen.async_run({
            "prompt": music_prompt,
            "duration": int(duration)
        })
        print("run2")
        print("music_audio", music_audio)
        # music_audio = {'output': {'mediaAttributes': {'mimeType': 'audio/mpeg', 'duration': 20.052}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/430eb06b9a9bd66bece456fd3cd10f8c6d99fb75c1d05a1da6c317247ac171c6.mp3'}, 'status': 'completed'}

        if music_audio.get("error"):
            raise Exception(f"Music generation failed: {music_audio['error']}")
        
        music_audio = eden_utils.prepare_result(music_audio)
        print("MUSIC AUDIO 55", music_audio)

        
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        music_file = eden_utils.download_file(music_audio['output'][0]['url'], temp_file.name+".mp3")
        print("MUSIC FILE 77", temp_file.name)
        with open(music_file, 'rb') as f:
            music_audio = AudioSegment.from_file(BytesIO(f.read()))
        #os.remove(temp_file.name)
        
        # fadeout music last 3 seconds
        
        print("MUSIC AUDIO 66", music_audio)
        print("MUSIC AUDIO 66 LENGTH", temp_file.name)

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

    print("lfg", audio)

    if audio:
        print("go1")
        audio_url, _ = s3.upload_audio_segment(audio)
        print("audio_url", audio_url)
    
    print("go2")
    # get resolution
    orientation = args.get("orientation")
    print("TE ORIENTATION IS", orientation)
    if orientation == "landscape":
        width, height = 1280, 768
    else:
        width, height = 768, 1280
    print("width", width)
    print("height", height)

    # get sequence lengths
    print("==== get sequence lengths ====")
    print("duration", duration)
    tens, fives = duration // 10, (duration - (duration // 10) * 10) // 5
    durations = [10] * int(tens) + [5] * int(fives)    
    random.shuffle(durations)
    num_clips = len(durations)
    print("durations", durations)
    print("num_clips", num_clips)


    # get visual prompt sequence
    print("==== get visual prompt sequence ====")
    print("reel.visual_prompt", reel.visual_prompt)
    print("THJE INSTRUCTIONS ARE", instructions)
    visual_prompts = write_visual_prompts(reel, num_clips, instructions)
    pprint(visual_prompts)




    flux_args = {
        "prompt": reel.visual_prompt,
        "width": width,
        "height": height
    }

    if use_lora:
        flux_args.update({
            "use_lora": True,
            "lora": lora,
            "lora_strength": lora_strength
        })


    flux_args = [{**flux_args} for _ in range(num_clips)]
    for i in range(num_clips):
        print("FLUX ARGS", i)
        flux_args[i]["prompt"] = visual_prompts[i % len(visual_prompts)]
        flux_args[i]["prompt"] += ", " + reel.visual_style
        flux_args[i]["seed"] = random.randint(0, 2147483647)

    print("FLUX ARGS!!!")
    pprint(flux_args)

    images = []
    for i in range(num_clips):
        image = await flux.async_run(flux_args[i])
        image = eden_utils.prepare_result(image)
        print("IMAGE ==1", image)
        output_url = image['output'][0]["url"]
        images.append(output_url)
    # images =['https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/6af97716cf3a4703877576e07823d5c6492a0355c2c7a55148b8f6a4cc8d97a7.png', 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/4bbcee84993883fe767502a29cdbe615e5f16b962de5d92a77e50ca466ef6564.png']

    print("IMAGES!!")
    print(images)


    # videos = ['https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/ccf83bd781685d8a457535c28d28c6c1dc1740486b7ad937813013558b95d4fe.mp4', 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/2d22e7328a8a2ad72d16e42d766b9cf67b6c50be129ad8b3733b33eda0f1e369.mp4']
    videos = []
    for i, image in enumerate(images):
        print("i", i)
        print("image", image)
        print("flux_args", flux_args[i])
        print("durations", durations[i])
        print("ok?", orientation)
        print("OK!!!!", {
            "prompt_image": image,
            "prompt_text": flux_args[i]["prompt"],
            "duration": str(durations[i]),
            "ratio": "16:9" if orientation == "landscape" else "9:16"
        })
        video = await runway.async_run({
            "prompt_image": image,
            "prompt_text": flux_args[i]["prompt"],
            "duration": durations[i],
            "ratio": "16:9" if orientation == "landscape" else "9:16"
        })
        print("video!!", video)
        video = eden_utils.prepare_result(video)
        print("video", video)
        video = video['output'][0]['url']
        videos.append(video)

    video = await video_concat.async_run({"videos": videos})
    video = eden_utils.prepare_result(video)
    print("video", video)
    video_url = video['output'][0]['url']
    
    if audio_url:
        output = await audio_video_combine.async_run({
            "audio": audio_url,
            "video": video_url
        })
        print("OUTPTU!")
        print(output)
        final_video = eden_utils.prepare_result(output)
        print(final_video)
        final_video_url = final_video['output'][0]['url']
        print("a 5")
        # output_url, _ = s3.upload_file(output)
        print("a 888")


    print("this is updating...")    

    return {
        "output": final_video_url,
        "intermediate_outputs": {
            "images": images,
            "videos": videos
        }
    }

