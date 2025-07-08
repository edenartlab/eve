"""
VIDEO TODO:
- fix cost formula
- add lora2. when two face loras, use double character
- negative_prompt
- start_image_strength
- n_samples
- vid2vid_sdxl, video_FX, texture_flow
- no start image and aspect ratio == auto, predict good aspect ratio

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

"""

import os
from eve.api.api import create
from eve.tool import Tool
from eve.models import Model
from eve.user import User

from eve.s3 import get_full_url
from eve.eden_utils import get_media_attributes
    

async def handler(args: dict, user: str = None, agent: str = None):

    print("THE AGENT IS", agent)
    print("THE USER IS", user)

    
    # if specific user is provided, check if they have access to veo3
    if user:
        user = User.from_mongo(user)
        print(user.featureFlags)
        veo3_enabled = "tool_access_veo3" in user.featureFlags
        print("VEO3 ENABLED", veo3_enabled)


    runway = Tool.load("runway")
    kling_pro = Tool.load("kling_pro")
    veo2 = Tool.load("veo2")
    veo3 = Tool.load("veo3")
    hedra = Tool.load("hedra")
    create = Tool.load("create")
    mmaudio = Tool.load("mmaudio")

    # args1 = {   
    #     "prompt": "Banny looks straight into the camera looking suave, frontal",
        
    #     # "prompt": "A desert caravan with camels walks through the desert under dusk",
    #     # "prompt": "A little league baseball game, batters hits a home run, crowd goes wild, he runs towards base",
    #     # "start_image": "https://d14i3advvh2bvd.cloudfront.net/48f4b354bd5711b4d2234a9b8f05b193e186df8639ca72273c068e8b4f1910f1.png",
    #     # "end_image": "https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg",
    #     # "n_samples": 1,
    #     # "aspect_ratio": "9:16",
    #     "lora": "6766760643808b38016c64ce",  # flux banny
    #     "aspect_ratio": "auto",
    #     "quality": "fast",
    #     # "sound_effects": "Baseball field, little league, bats crunching, applause",
    #     "talking_head": True,
    #     "audio": "https://edenartlab-stage-data.s3.amazonaws.com/4452e52569d517cac1c25c8817a052224bd74f4133d2471cb3bb8d6a23a27efb.mp3",
    # }
    # args = args1


    prompt = args["prompt"]
    # n_samples = args.get("n_samples", 1)
    start_image = args.get("start_image", None)
    end_image = args.get("end_image", None)
    seed = args.get("seed", None)
    lora_strength = args.get("lora_strength", 0.75)
    aspect_ratio = args.get("aspect_ratio", "auto")
    quality = args.get("quality", "standard")
    duration = args.get("duration", 5)
    talking_head = args.get("talking_head", False)
    audio = args.get("audio", None)
    sound_effects = args.get("sound_effects", None)
    
    if end_image:
        assert start_image, "Must provide start_image if end_image is provided"

    # get loras
    loras = get_loras(args.get("lora"), args.get("lora2"))

    # Rules
    if talking_head and audio:
        video_tool = hedra
    elif quality == "fast":
        video_tool = runway
    elif quality == "standard":
        video_tool = kling_pro
    elif quality == "high_quality":
        if sound_effects:
            video_tool = veo3
        else:
            if start_image:
                video_tool = veo2
            else:
                video_tool = veo3

    print("Tool selected", video_tool.key)

    # If there is no start image, generate one for any of the following reasons:
    # - Using Runway, because it requires one
    # - Lora is set, so we want to do img2vid with a lora-applied image instead of txt2vid
    # Otherwise, can just do txt2vid without a start image
    if not start_image:
        if video_tool == runway or loras:
            print("Generating start image with Lora")
            args = {"prompt": prompt}
            if loras:
                args.update({
                    "lora": str(loras[0].id),
                    "lora_strength": lora_strength,
                })
            try:
                result = await create.async_run(args)
                start_image = get_full_url(result["output"][0]["filename"])
            except Exception as e:
                raise Exception("Error generating start image for img2vid. Try generating it yourself first with the 'create' tool, and then use it as start_image. Original error: {}".format(e))

    # Get start image attributes if any
    if start_image:
        start_image_attributes, _ = get_media_attributes(start_image)
    else:
        start_image_attributes = None
    
    
    #########################################################
    # Runway
    if video_tool == runway:

        # Runway can only produce 5 or 10s videos        
        duration = 10 if duration > 7.5 else 5
        
        # Snap aspect ratio to closest Runway preset
        aspect_ratio = snap_aspect_ratio_to_model(
            aspect_ratio, 
            "runway3" if end_image else "runway4", 
            start_image_attributes
        )

        args = {
            "prompt_text": prompt,
            "start_image": start_image, # Runway requires start image
            "model": "gen4_turbo",
            "duration": duration,
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
        result = await runway.async_run(args)


    #########################################################
    # Kling 
    elif video_tool == kling_pro:

        # Kling can only produce 5 or 10s videos        
        duration = 10 if duration > 7.5 else 5

        # Snap aspect ratio to closest Kling Pro preset
        aspect_ratio = snap_aspect_ratio_to_model(
            aspect_ratio, 
            "kling", 
            start_image_attributes
        )
    
        args = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "quality": "high",  # use Kling 2 optimistically
            "duration": duration
        }

        if start_image:
            args.update({
                "start_image": start_image,
            })

        # If an end image is requested, fall back to Kling 1.6 Pro which supports it
        if end_image:
            args.update({
                "end_image": end_image,
                "quality": "medium",
            })

        print(f"Running Kling Pro {args['quality']}", args)
        result = await kling_pro.async_run(args)


    #########################################################
    # Veo-2
    elif video_tool == veo2:

        # Veo can only produce 5-8s videos
        duration = min(duration, 8)

        # Snap aspect ratio to closest Veo2 preset
        aspect_ratio = snap_aspect_ratio_to_model(
            aspect_ratio, 
            "veo2", 
            start_image_attributes
        )

        args = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        if start_image:
            args.update({
                "image": start_image,
            })

        # if end_image:
        #     args.update({
        #         "end_image": end_image,
        #     })

        print("Running Veo2", args)
        result = await veo2.async_run(args)


    #########################################################
    # Veo-3
    elif video_tool == veo3:

        # Veo can only produce 5-8s videos
        duration = min(duration, 8)

        args = {
            "prompt": f"{prompt}. {sound_effects}",
            "duration": duration,
            # "aspect_ratio": aspect_ratio,
        }

        # if start_image:
        #     args.update({
        #         "image": start_image,
        #     })

        if seed:
            args["seed"] = seed

        print("Running Veo3", args)
        result = await veo3.async_run(args)
        

    #########################################################
    # Hebra
    elif video_tool == hedra:
        
        # Snap aspect ratio to closest Hebra preset
        aspect_ratio = snap_aspect_ratio_to_model(
            aspect_ratio, 
            "hedra", 
            start_image_attributes
        )
        
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


    #########################################################
    # Final video is now generated
    final_video = get_full_url(result["output"][0]["filename"])
    print("final result", final_video)


    # If sound effects are requested, try to add them
    if sound_effects and video_tool.key != "veo3":
        try:
            args = {
                "prompt": sound_effects,
                "video": final_video,
                "duration": min(duration, 16),
            }
            if seed:
                args["seed"] = seed
            print("Running MMAudio", args)
            sound_fx = await mmaudio.async_run(args)
            final_video = get_full_url(sound_fx["output"][0]["filename"])
            print("Final result with sound effects", final_video)
        
        except Exception as e:
            print("Error adding sound effects, just return video without it. Error: {}".format(e))

    return {
        "output": final_video
    }


def get_loras(lora1, lora2):
    """
    Get loras from their IDs
    """
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


def get_closest_aspect_ratio_preset(aspect_ratio: float, presets: dict) -> str:
    """
    Get closest aspect ratio preset from a list of presets
    """
    closest_preset = None
    min_difference = float('inf')
    for preset_name, preset_ratio in presets.items():
        difference = abs(aspect_ratio - preset_ratio)
        if difference < min_difference:
            min_difference = difference
            closest_preset = preset_name    
    return closest_preset


def snap_aspect_ratio_to_model(aspect_ratio, model_name, start_image_attributes):
    """
    Snap aspect ratio to closest preset for a given model
    """

    presets = {
        "runway3": {"16:9": 16/9, "9:16": 9/16},
        "runway4": {"21:9": 21/9, "16:9": 16/9, "4:3": 4/3, "1:1": 1/1, "3:4": 3/4, "9:16": 9/16},
        "kling": {"16:9": 16/9, "1:1": 1/1, "9:16": 9/16},
        "veo2": {"16:9": 16/9, "9:16": 9/16},
        "hedra": {"16:9": 16/9, "1:1": 1/1, "9:16": 9/16},
    }[model_name]
    
    if aspect_ratio == "auto":

        # If there is a start image, snap to its aspect ratio
        if start_image_attributes:
            aspect_ratio = get_closest_aspect_ratio_preset(
                start_image_attributes["aspectRatio"], 
                presets
            )

        # Otherwise, default to 16:9
        # todo: make this smarter
        else:
            aspect_ratio = "16:9"

    # If aspect ratio is set but not in presets, snap to closest preset
    else:    
        if aspect_ratio not in presets:
            ar = aspect_ratio.split(":")
            aspect_ratio_ = float(ar[0]) / float(ar[1])
            aspect_ratio = get_closest_aspect_ratio_preset(
                aspect_ratio_,
                presets
            )

    return aspect_ratio