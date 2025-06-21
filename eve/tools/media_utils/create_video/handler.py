"""
VIDEO TODO:
- negative_prompt
- n_samples
- vid2vid_sdxl, video_FX, texture_flow, mmaudio?

AUDIO
- elevenlabs
- stable_audio
- mmaudio
- ace_step_musicgen
- zonos
- transcription

MEDIA_EDITOR
- combine audio
- extract/remove/split tracks
- speed up/slow down?


----

prompt (negative prompt)
starting image (not always supported)
ending_image
aspect ratio (auto)
duration
seed


output [image, video]
prompt
 - negative prompt
starting image
 - controlnet
ending_image (VIDEO)
aspect ratio 
duration (VIDEO)
lora (IMAGE)
seed

"""

import os
from eve.api.api import create
from eve.tool import Tool
from eve.models import Model

from eve.s3 import get_full_url
from eve.eden_utils import get_media_attributes
    



def get_closest_aspect_ratio_preset(aspect_ratio: float, presets: dict) -> str:
    closest_preset = None
    min_difference = float('inf')
    for preset_name, preset_ratio in presets.items():
        difference = abs(aspect_ratio - preset_ratio)
        if difference < min_difference:
            min_difference = difference
            closest_preset = preset_name    
    return closest_preset


async def handler(args: dict, user: str = None, agent: str = None):
    runway = Tool.load("runway")
    kling_pro = Tool.load("kling_pro")
    veo2 = Tool.load("veo2")
    veo3 = Tool.load("veo3")
    hedra = Tool.load("hedra")
    create = Tool.load("create")
    mmaudio = Tool.load("mmaudio")

    args1 = {   
        # "prompt": "Banny looks straight into the camera looking suave, frontal",
        
        # "prompt": "A desert caravan with camels walks through the desert under dusk",
        "prompt": "A little league baseball game, batters hits a home run, crowd goes wild, he runs towards base",
        # "start_image": "https://d14i3advvh2bvd.cloudfront.net/48f4b354bd5711b4d2234a9b8f05b193e186df8639ca72273c068e8b4f1910f1.png",
        # "end_image": "https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg",
        "n_samples": 1,
        # "aspect_ratio": "9:16",
        # "lora": "6766760643808b38016c64ce",  # flux banny
        "aspect_ratio": "auto",
        "quality": "fast",
        "add_sound_effects": True,
        # "talking_head": True,
        # "audio": "https://edenartlab-stage-data.s3.amazonaws.com/4452e52569d517cac1c25c8817a052224bd74f4133d2471cb3bb8d6a23a27efb.mp3",
    }
    # args = args1


    prompt = args["prompt"]
    n_samples = args.get("n_samples", 1)
    start_image = args.get("start_image", None)
    end_image = args.get("end_image", None)
    seed = args.get("seed", None)
    lora_strength = args.get("lora_strength", 0.75)
    aspect_ratio = args.get("aspect_ratio", "auto")
    quality = args.get("quality", "standard")
    duration = args.get("duration", 5)
    talking_head = args.get("talking_head", False)
    audio = args.get("audio", None)
    sound_effects = args.get("sound_effects", False)
    
    if end_image:
        assert start_image, "Must provide start_image if end_image is provided"

    # get loras
    loras = get_loras(args.get("lora"), args.get("lora2"))

    print("\n\n\n\n\n--------------------------------")
    print("INIT IMAGE", start_image)
    print("LORAS", loras)
    print("--------------------------------")


    # Rules
    if talking_head and audio:
        video_tool = hedra
    elif quality == "fast":
        video_tool = runway
    elif quality == "standard":
        video_tool = kling_pro
    elif quality == "high_quality":
        video_tool = veo2

    # If Loras and no start image, generate start image with Lora
    if not start_image and (video_tool == runway or loras):
        print("Generating start image with Lora")
        args = {
            "prompt": prompt,
        }
        if loras:
            args.update({
                "lora": str(loras[0].id),
                "lora_strength": lora_strength,
            })
        result = await create.async_run(args)
        start_image = get_full_url(result["output"][0]["filename"])

    # Get start image attributes if any
    if start_image:
        start_image_attributes, _ = get_media_attributes(start_image)
    else:
        start_image_attributes = None
    
    # Runway
    if video_tool == runway:
        if aspect_ratio == "auto":
            presets = {"16:9": 16/9, "9:16": 9/16} if end_image else {"21:9": 21/9, "16:9": 16/9, "4:3": 4/3, "1:1": 1/1, "3:4": 3/4, "9:16": 9/16}
            aspect_ratio = get_closest_aspect_ratio_preset(start_image_attributes["aspectRatio"], presets)

        args = {
            "prompt_text": prompt,
            "start_image": start_image, # Runway requires start image
            "model": "gen4_turbo",
            "duration": 10 if duration > 7.5 else 5,
            "ratio": aspect_ratio,
        }

        if aspect_ratio != "auto":
            args["ratio"] = aspect_ratio

        # If ending image, must use gen3a_turbo
        if end_image:
            args.update({
                "end_image": end_image,
                "model": "gen3a_turbo",
            })

        if seed:
            args["seed"] = seed

        print("Running Runway", args)
        # result = await runway.async_run(args)
        result = {
            "output": [
                {
                    "filename": "bad88c52fcee21daef8f605c204aea8cbb990cad2c4324c4cbb02ce6a374247c.mp4"
                }
            ]
        }

    # Kling Pro
    elif video_tool == kling_pro:
        if aspect_ratio == "auto":
            if start_image:
                presets = {"16:9": 16/9, "1:": 1/1, "9:16": 9/16}
                aspect_ratio = get_closest_aspect_ratio_preset(start_image_attributes["aspectRatio"], presets)
            else:
                aspect_ratio = "16:9"
    
        args = {
            "prompt": prompt,
            # "n_samples": n_samples,
            "aspect_ratio": aspect_ratio,
            "quality": "high",
            "duration": duration,
        }

        if start_image:
            args.update({
                "start_image": start_image,
            })

        if end_image:
            args.update({
                "end_image": end_image,
                "quality": "medium",
            })

        print("Running Kling Pro", args)
        result = await kling_pro.async_run(args)


    # Veo3
    elif video_tool == veo2:
        args = {
            "prompt": prompt,
            # "n_samples": n_samples,
            "duration": min(duration, 8),
        }
        print("Veo2 args", args)

        if start_image:
            args.update({
                "image": start_image,
            })

        # if end_image:
        #     args.update({
        #         "end_image": end_image,
        #     })

        if aspect_ratio == "auto":
            args["aspect_ratio"] = "16:9"

        print("Running Veo2", args)
        result = await veo2.async_run(args)
        

    elif video_tool == hedra:
        if aspect_ratio == "auto":
            if start_image:
                aspect_ratio = get_closest_aspect_ratio_preset(
                    start_image_attributes["aspectRatio"], {"16:9": 16/9, "1:1": 1/1, "9:16": 9/16}
                )
            else:
                aspect_ratio = "16:9"
        
        args = {
            "image": start_image,
            "prompt": prompt,
            "audio": audio,
            "aspectRatio": aspect_ratio,
        }

        print("Running Hebra", args)
        result = await hedra.async_run(args)
        
    else:
        raise Exception("Invalid video tool", video_tool)

    final_result = get_full_url(result["output"][0]["filename"])

    print("final result", final_result)

    # Add sound effects at end if asked for
    if sound_effects:
        args = {
            "prompt": sound_effects,
            "video": final_result,
            "duration": min(duration, 16),
        }
        if seed:
            args["seed"] = seed
        print("Running MMAudio", args)
        sound_fx = await mmaudio.async_run(args)
        final_result = get_full_url(sound_fx["output"][0]["filename"])
        print("Final result with sound effects", final_result)

    return {
        "output": final_result
    }


def get_loras(lora1, lora2):
    loras = []
    for lora_id in [lora1, lora2]:
        if lora_id:
            lora = Model.from_mongo(lora_id)
            if not lora:
                raise Exception(f"Lora {lora_id} not found on {os.getenv('ENV')}")        
            loras.append(lora)

    if len(loras) == 2 and "sdxl" in [lora.base_model for lora in loras]:
        raise Exception("Second Lora is not supported for SDXL")
        
    return loras

