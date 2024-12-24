# from tools import runway, video_concat
"""

scenarios
- single prompt
- prompt



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
        """A reel is a short film of 30-60 seconds in length. It should be a single coherent scene for a commercial, movie trailer, tiny film, advertisement, or some other short time format."""

        voiceover: str = Field(..., description="The text of the voiceover, if one is not provided by the user.")
        music_prompt: str = Field(..., description="A prompt describing the music to compose for the reel. Describe instruments, genre, style, mood qualities, emotion, and any other relevant details.")
        visual_prompt: str = Field(..., description="A prompt a text-to-image model to precisely describe the visual content of the reel. The visual prompt should be structured as a descriptive sentence, precisely describing the visible content of the reel, the aesthetic style, and action.")
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
    


# async def go():
#     speech_audio = await elevenlabs.handler({
#         "text": "this is a test",
#         "voice_id": "j6Fbg1nV1BgnjZqPvN1d"
#     }, db="STAGE")
#     return speech_audio

# import asyncio
# asyncio.run(go())

from bson.objectid import ObjectId

async def handler(args: dict, db: str):
    
    from ...tools import select_random_voice
    from ...tools.elevenlabs import handler as elevenlabs
    from ...tool import Tool
    from ...mongo import get_collection

    musicgen = Tool.load("musicgen", db=db)
    flux = Tool.load("flux_dev", db=db)
    runway = Tool.load("runway", db=db)
    video_concat = Tool.load("video_concat", db=db)
    audio_video_combine = Tool.load("audio_video_combine", db=db)

    instructions = None

    use_lora = args.get("use_lora", False)
    if use_lora:
        lora = args.get("lora")
        loras = get_collection("models", db=db)
        lora_doc = loras.find_one({"_id": ObjectId(lora)})
        lora_name  = lora_doc.get("name")
        caption_prefix = lora_doc["args"]["caption_prefix"]
        lora_strength = args.get("lora_strength")
        instructions = f'In the visual prompts, *all* mentions of {lora_name} should be replaced with "{caption_prefix}". So for example, instead of "A photo of {lora_name} on the beach", always write "A photo of {caption_prefix} on the beach".'
        
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
        speech_audio = await elevenlabs.handler({
            "text": reel.voiceover,
            "voice_id": voice
        }, db=db)

        if speech_audio.get("error"):
            raise Exception(f"Speech generation failed: {speech_audio['error']}")
        
        with open(speech_audio['output'], 'rb') as f:
            speech_audio = AudioSegment.from_file(BytesIO(f.read()))
        
        duration = len(speech_audio) / 1000
        new_duration = round((duration + 2) / 5) * 5
        if new_duration > duration:
            amount_silence = new_duration - duration
            silence = AudioSegment.silent(duration=amount_silence * 1000 * 0.5)
            speech_audio = silence + speech_audio + silence
        duration = len(speech_audio) / 1000

        audio_url, _ = s3.upload_audio_segment(speech_audio)
        print("audio_url", audio_url)

        audio = speech_audio


    print("THE DURATION IS", duration)
    

    if args.get("use_music"):
        music_prompt = args.get("music_prompt") or reel.music_prompt
        music_audio = await musicgen.async_run({
            "prompt": music_prompt,
            "duration": int(duration)
        }, db=db)
        # music_audio = {'output': {'mediaAttributes': {'mimeType': 'audio/mpeg', 'duration': 20.052}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/430eb06b9a9bd66bece456fd3cd10f8c6d99fb75c1d05a1da6c317247ac171c6.mp3'}, 'status': 'completed'}

        if music_audio.get("error"):
            raise Exception(f"Music generation failed: {music_audio['error']}")
        
        music_audio = eden_utils.prepare_result(music_audio, db=db)
        print("MUSIC AUDIO 55", music_audio)

        
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        music_file = eden_utils.download_file(music_audio['output'][0]['url'], temp_file.name+".mp3")
        print("MUSIC FILE 77", temp_file.name)
        with open(music_file, 'rb') as f:
            music_audio = AudioSegment.from_file(BytesIO(f.read()))
        #os.remove(temp_file.name)
        print("MUSIC AUDIO 66", music_audio)
        print("MUSIC AUDIO 66 LENGTH", temp_file.name)

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
        flux_args[i]["seed"] = random.randint(0, 2147483647)

    print("FLUX ARGS!!!")
    pprint(flux_args)

    images = []
    for i in range(num_clips):
        image = await flux.async_run(flux_args[i], db=db)
        image = eden_utils.prepare_result(image, db=db)
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
            "duration": str(durations[i]),
            "ratio": "16:9" if orientation == "landscape" else "9:16"
        }, db=db)
        print("video!!", video)
        video = eden_utils.prepare_result(video, db=db)
        print("video", video)
        video = video['output'][0]['url']
        videos.append(video)


    
    video = await video_concat.async_run({"videos": videos}, db=db)
    video = eden_utils.prepare_result(video, db=db)
    print("video", video)
    video_url = video['output'][0]['url']
    
    if audio_url:
        output = await audio_video_combine.async_run({
            "audio": audio_url,
            "video": video_url
        }, db=db)
        print("OUTPTU!")
        print(output)
        final_video = eden_utils.prepare_result(output, db=db)
        print(final_video)
        final_video_url = final_video['output'][0]['url']
        print("a 5")
        # output_url, _ = s3.upload_file(output)
        print("a 6")




    return {
        "output": final_video_url,
        "intermediate_outputs": {
            "images": images,
            "videos": videos
        }
    }


async def handler2(args: dict, db: str):
    # try:
    if 1:

        vid = voice.select_random_voice("A gruff and intimidating voice") 
        print("vid", vid)


        

        prompt = args.get("prompt")
        music = args.get("use_music")
        music_prompt = (args.get("music_prompt") or "").strip()
        
        narrator = args.get("use_narrator")
        narration = (args.get("narration") or "").strip() if narrator else ""
        narration = narration[:600]
        if narration: # remove everything after the last space
            last_space_idx = narration.rindex(" ")
            narration = narration[:last_space_idx]
        
        min_duration = args.get("min_duration")
        
        # resolution = args.get("resolution", "none")
        # width = args.get("width", None)
        # height = args.get("width", None)

        # print("resolution", resolution)
        # print("width", width)
        # print("height", height)
        

        orientation = args.get("orientation")
        if orientation == "landscape":
            width = 1280
            height = 768
        else:
            width = 768
            height = 1280

        
        speech_boost = 5

        if not min_duration:
            raise Exception("min_duration is required")

        print("ALL ARGS ARE", args)
        
        characters = extract_characters(prompt)

        if narrator:
            characters.append(Character(name="narrator", description="The narrator of the reel is a voiceover artist who provides some narration for the reel"))
        
        print("characters :: ", characters)

        voices = {
            c.name: voice.select_random_voice(c.description) 
            for c in characters
        }

        story = write_reel(prompt, characters, narration, music, music_prompt)

        print("story", story)
        
        duration = min_duration

        print("characters", characters)
        print("voices", voices)
        print("story", story)

        metadata = {
            "reel": story.model_dump(),
            "characters": [c.model_dump() for c in characters],
        }

        print("metadata", metadata)

        speech_audio = None
        music_audio = None
        print("NEXT")
        # generate speech
        print(" ---1-1-1 lets go")
        print(voices)
        # print(voices[story.speaker])

        if story.speech:
            speech_audio = voice.run(
                text=story.speech,
                voice_id=voices[story.speaker]
            )
            print("generated speech", story.speech)
            speech_audio = AudioSegment.from_file(BytesIO(speech_audio))
            silence1 = AudioSegment.silent(duration=2000)
            silence2 = AudioSegment.silent(duration=3000)
            speech_audio = silence1 + speech_audio + silence2
            duration = max(duration, len(speech_audio) / 1000)
            metadata["speech"], _ = s3.upload_audio_segment(speech_audio)
            
        # # generate music
        if music and story.music_prompt:
            from eve.tool import Tool
            musicgen = Tool.load("musicgen", db="STAGE")
            music = await musicgen.async_run({
                "prompt": story.music_prompt,
                "duration": int(duration)
            }, db=db)
            print("THE MUSIC IS DONE!")
            print(music)
            print("generated music", story.music_prompt)
            music_bytes = requests.get(music[0]['url']).content
            music_audio = AudioSegment.from_file(BytesIO(music_bytes))
            metadata["music"], _ = s3.upload_audio_segment(music_audio)

        # mix audio
        audio = None
        if speech_audio and music_audio:
            diff_db = ratio_to_db(speech_audio.rms / music_audio.rms)
            music_audio = music_audio + diff_db
            speech_audio = speech_audio + speech_boost
            audio = music_audio.overlay(speech_audio)        
        elif speech_audio:
            audio = speech_audio
        elif music_audio:
            audio = music_audio
        
        print("THE AUDIO IS DONE!")
        print(audio)




        print("MAKE THE VIDEO!")
        
        flux_args = {
            "prompt": story.image_prompt,
            "width": width,
            "height": height
        }        
        print("flux_args", flux_args)
        use_lora = args.get("use_lora", False)
        if use_lora:
            lora = args.get("lora")
            lora_strength = args.get("lora_strength")
            flux_args.update({
                "use_lora": True,
                "lora": lora,
                "lora_strength": lora_strength
            })

        print("flux_args", flux_args)


        num_clips = math.ceil(duration / 10)
        print("num_clips", num_clips)

        flux_args = [flux_args.copy()] * num_clips

        if num_clips > 1:
            prompts = prompt_variations(prompt, num_clips)        
            print("ORIGINAL PROMPT", prompt)
            print("-----")
            print("PROMPT VARIATIONS")
            for p, new_prompt in enumerate(prompts):
                print(p)
                print("-----")
                flux_args[p]["prompt"] = new_prompt
            

        txt2img = load_tool("../../workflows/workspaces/flux/workflows/flux_dev")
        images = []
        for i in range(num_clips):
            print("i", i)
            image = await txt2img.async_run(flux_args[i], db=db)
            print("THE IMAGE IS DONE!")
            print(image)
            output_url = image[0]["url"]
            images.append(output_url)

        print("run runway")
        runway = load_tool("tools/runway")

        # print("images", images)

        # num_clips = 1
        # images = ["https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/53bc5b8d715c6b243db787ab2ca15718f983dd80811f470f2a8e9aa4c8f518cc.png"]
        # orientation = "portrait"
        # duration = 5

        videos = []
        dur = 10
        for i in range(num_clips):
            if i == num_clips - 1 and duration % 10 < 5:
                dur = 5
            print("video", i)
            video = await runway.async_run({
                "prompt_image": images[i],
                "prompt_text": "A panorama of a sand castle", #story.image_prompt,
                "duration": str(dur),
                "ratio": "16:9" if orientation == "landscape" else "9:16"
            }, db=db)
            print("video is done", i)
            print(video)
            videos.append(video[0])

        print("videos", videos)

        # download videos
        # videos = [eden_utils.get_file_handler(".mp4", v) for v in videos]

        video_concat = load_tool("tools/media_utils/video_concat")
        video = await video_concat.async_run({"videos": [v["url"] for v in videos]}, db=db)
        print("video", video)
        video = video[0]['url']


        # txt2vid = load_tool("../workflows/workspaces/video/workflows/txt2vid")
        # video = await txt2vid.async_run({
        #     "prompt": story.image_prompt,
        #     "n_frames": 128,
        #     "width": width,
        #     "height": height
        # }, db=db)
        print("THE VIDEO IS DONE!")
        # video = [{'mediaAttributes': {'mimeType': 'video/mp4', 'width': 1280, 'height': 768, 'aspectRatio': 1.6666666666666667, 'duration': 31.6}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/75bf55b76a8e4cadbf824b4eee1673a8c41c24f6688a1d5f2f90723c237c4ae6.mp4'}]
        print(video)
        # output_url = video[0]["url"]
        # output_url = video
        # video = "output.mp4"

        # print("txt2vid", output_url)

        print("a 1")
        if audio:
            print("a 2")
            buffer = BytesIO()
            print("a 3")
            audio.export(buffer, format="mp3")
            print("a 4")
            # print("URL IS", video[0]["url"])
            output = eden_utils.make_audiovideo_clip(video, buffer)
            print(output)
            print("a 5")
            # output_url, _ = s3.upload_file(output)
            print("a 6")

        # print("output_url", output_url)
        print("metadata", metadata)

        print("LETS GO!!!! ...")
        # print("output_url", output_url)
        print("story", story)
        print("characters", characters)
        print("images", ["images"])
        print("videos", ["videos"])
        print("music", music)
        zz = {
            "output": output,
            "intermediate_outputs": {
                "story": story.model_dump(),
                "characters": [c.model_dump() for c in characters],
                "images": images,
                "videos": videos,
                "music": music,
                # "speech": speech_audio
            }
        }

        # zz = {'output': '/var/folders/h_/8038q2513yz414f7j3yqy_580000gn/T/tmpkjf59iem.mp4', 'intermediate_outputs': {'story': {'image_prompt': 'A cinematic asteroid view of Mars hurtling through space and colliding dramatically with Earth, causing an immense explosion.', 'music_prompt': 'Intense orchestral music building to a crescendo, evoking tension and epic disaster.', 'speaker': 'narrator', 'speech': "Witness the catastrophic collision of Mars and Earth, a cosmic dance of destruction, captured with stunning simulation, as the red planet meets our blue world in an inevitable, fiery embrace. Watch as continents crumble and atmospheres collide, forever altering the solar system's story."}, 'characters': [{'name': 'narrator', 'description': 'The narrator of the reel is a voiceover artist who provides some narration for the reel'}], 'images': ['https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/6532b48aa71c98b56a9ab41f63a24c09029527360af26b9e089218de4043e8f8.png', 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/e423f8290876ee4694f811bb1716e5d70acdf6ab6b6ea3480357ca5ae6af2f2b.png', 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/3892e536589147b729ff8d31ae93457f24361cb01450de707a700ef798828bc8.png'], 'videos': [{'mediaAttributes': {'mimeType': 'video/mp4', 'width': 1280, 'height': 768, 'aspectRatio': 1.6666666666666667, 'duration': 10.54}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/5020c31bf1fdf2f590113a75148a021aae38eb809532d2799b9c434f3548f832.mp4'}, {'mediaAttributes': {'mimeType': 'video/mp4', 'width': 1280, 'height': 768, 'aspectRatio': 1.6666666666666667, 'duration': 10.54}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/2559354dcfbfe2921e468580e8ed66823f332924191bdeb5f6aed4d3ae4a19ba.mp4'}, {'mediaAttributes': {'mimeType': 'video/mp4', 'width': 1280, 'height': 768, 'aspectRatio': 1.6666666666666667, 'duration': 10.54}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/1ddbcdbfaa3c4a8ab218a79cbbf1d95f92cc105b0d5e30fe8c5cd0bc8f00bfa4.mp4'}], 'music': [{'mediaAttributes': {'mimeType': 'audio/mpeg', 'duration': 28.044}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/e3b1438800d80293a2cc87a6371cd6947ad9e10bd449b5bfe27e4891dbab9448.mp3'}]}}

        # zz = {'output': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/911d8cbe1775cfa52ddf3900fa2d5e55698de63860eb00a4be246baf5c174912.mp4', 'intermediate_outputs': {'story': {'image_prompt': 'A dramatic simulation showing Mars approaching and colliding with Earth, with both planets breaking apart and creating a cosmic explosion.', 'music_prompt': "213413", 'speaker': "None222", 'speech': "2342"}, 'characters': ["SDFA"], 'images': ['images'], 'videos': ['videos'], 'music': "ddd"}}

        # zz = {
        #     'output': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/911d8cbe1775cfa52ddf3900fa2d5e55698de63860eb00a4be246baf5c174912.mp4',
        #     'intermediate_outputs': {
        #         'story': {'image_prompt': 'A dramatic simulation showing Mars approaching and colliding with Earth, with both planets breaking apart and creating a cosmic explosion.'}
        #     }
        # }

        print("zz", zz)

        from pprint import pprint
        pprint(zz)

        return zz


    # except asyncio.CancelledError as e:
    #     print("asyncio CancelledError")
    #     print(e)
    # except Exception as e:
    #     print("normal error")
    #     print(e)
        

# import eve.eden_utils
# zz = {'output': '/Users/gene/Eden/dev/eve/97468b465a993c272b8d12990095027ec67f86ddfea6093c36be8925503d41a4.mp4', 'intermediate_outputs': {'story': {'image_prompt': 'A cinematic asteroid view of Mars hurtling through space and colliding dramatically with Earth, causing an immense explosion.', 'music_prompt': 'Intense orchestral music building to a crescendo, evoking tension and epic disaster.', 'speaker': 'narrator', 'speech': "Witness the catastrophic collision of Mars and Earth, a cosmic dance of destruction, captured with stunning simulation, as the red planet meets our blue world in an inevitable, fiery embrace. Watch as continents crumble and atmospheres collide, forever altering the solar system's story."}, 'characters': [{'name': 'narrator', 'description': 'The narrator of the reel is a voiceover artist who provides some narration for the reel'}], 'images': ['https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/6532b48aa71c98b56a9ab41f63a24c09029527360af26b9e089218de4043e8f8.png', 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/e423f8290876ee4694f811bb1716e5d70acdf6ab6b6ea3480357ca5ae6af2f2b.png', 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/3892e536589147b729ff8d31ae93457f24361cb01450de707a700ef798828bc8.png'], 'videos': [{'mediaAttributes': {'mimeType': 'video/mp4', 'width': 1280, 'height': 768, 'aspectRatio': 1.6666666666666667, 'duration': 10.54}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/5020c31bf1fdf2f590113a75148a021aae38eb809532d2799b9c434f3548f832.mp4'}, {'mediaAttributes': {'mimeType': 'video/mp4', 'width': 1280, 'height': 768, 'aspectRatio': 1.6666666666666667, 'duration': 10.54}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/2559354dcfbfe2921e468580e8ed66823f332924191bdeb5f6aed4d3ae4a19ba.mp4'}, {'mediaAttributes': {'mimeType': 'video/mp4', 'width': 1280, 'height': 768, 'aspectRatio': 1.6666666666666667, 'duration': 10.54}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/1ddbcdbfaa3c4a8ab218a79cbbf1d95f92cc105b0d5e30fe8c5cd0bc8f00bfa4.mp4'}], 'music': [{'mediaAttributes': {'mimeType': 'audio/mpeg', 'duration': 28.044}, 'url': 'https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/e3b1438800d80293a2cc87a6371cd6947ad9e10bd449b5bfe27e4891dbab9448.mp3'}]}}
# eve.eden_utils.upload_result(zz, "STAGE")
