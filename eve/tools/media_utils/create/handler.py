"""
TODO:
 - face_swap, flux_inpainting, outpaint, remix_flux_schnell
 - deal with moderation errors for flux_kontext and openai tools, and errors in general
 - negative prompting
 - make init image strength a parameter
 - txt2img has "style image" / ip adapter
 - video start_image_strength
 - enforce n_samples
 - vid2vid_sdxl, video_FX, texture_flow
 - no start image and aspect ratio == auto, predict good aspect ratio
"""

import os
from bson import ObjectId
from eve.s3 import get_full_url
from eve.tool import Tool
from eve.models import Model
# from eve.api.api import create
from eve.user import User
from eve.utils import get_media_attributes


def is_image_file(url: str) -> bool:
    """Check if URL points to an image file based on extension"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.tif', '.ico']
    url_lower = url.lower()
    # Remove query parameters if present
    url_lower = url_lower.split('?')[0]
    return any(url_lower.endswith(ext) for ext in image_extensions)


def is_video_file(url: str) -> bool:
    """Check if URL points to a video file based on extension"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg', '.3gp']
    url_lower = url.lower()
    # Remove query parameters if present
    url_lower = url_lower.split('?')[0]
    return any(url_lower.endswith(ext) for ext in video_extensions)


def validate_media_types(reference_images, reference_video):
    """Validate that reference images are images and reference video is a video"""
    # Check reference images
    if reference_images:
        for i, img in enumerate(reference_images):
            if isinstance(img, str) and not is_image_file(img):
                if is_video_file(img):
                    raise Exception(f"Reference image {i+1} ({img}) is not an image, it's a video")
                # Don't error on URLs without extensions, they might be valid
    
    # Check reference video
    if reference_video:
        if isinstance(reference_video, str) and not is_video_file(reference_video):
            if is_image_file(reference_video):
                raise Exception(f"The reference video {reference_video} is not a video, it's an image")
            # Don't error on URLs without extensions, they might be valid


async def handler(args: dict, user: str = None, agent: str = None):
    print("args", args)
    print("user", user)
    print("agent", agent)

    # Validate media types
    reference_images = args.get("reference_images", [])
    reference_video = args.get("reference_video", None)
    validate_media_types(reference_images, reference_video)

    output_type = args.get("output", "image")

    if output_type == "image":
        return await handle_image_creation(args, user, agent)
    elif output_type == "video":
        return await handle_video_creation(args, user, agent)
    else:
        raise Exception(f"Invalid output type: {output_type}")

async def handle_image_creation(args: dict, user: str = None, agent: str = None):
    """Handle image creation - copied from original create tool handler"""

    # load tools
    flux_schnell = Tool.load("flux_schnell")
    flux_dev_lora = Tool.load("flux_dev_lora")
    flux_dev = Tool.load("flux_dev")
    flux_kontext = Tool.load("flux_kontext")
    flux_double_character = Tool.load("flux_double_character")
    nano_banana = Tool.load("nano_banana")
    txt2img = Tool.load("txt2img")
    openai_image_edit = Tool.load("openai_image_edit")
    openai_image_generate = Tool.load("openai_image_generate")
    seedream3 = Tool.load("seedream3")
    # get args
    prompt = args["prompt"]
    n_samples = args.get("n_samples", 1)
    reference_images = args.get("reference_images", [])
    init_image = reference_images[0] if len(reference_images) > 0 else None
    extras = args.get("extras", [])
    text_precision = "text_precision" in extras
    double_character = "double_character" in extras
    controlnet = "controlnet" in extras
    seed = args.get("seed", None)
    aspect_ratio = args.get("aspect_ratio", "auto")
    model_preference = args.get("model_preference", "").lower()

    # get loras
    loras = get_loras(args.get("lora"), args.get("lora2"))
    lora_strength = args.get("lora_strength", 0.8)
    lora2_strength = args.get("lora2_strength", 0.8)

    # check if both loras are faces
    two_faces = len(loras) == 2 and all(
        [
            (lora.args.get("mode") or lora.args.get("concept_mode")) == "face"
            for lora in loras
        ]
    )

    intermediate_outputs = {}

    # Determine tool
    if init_image:
        if text_precision:
            if loras:
                image_tool = openai_image_edit  # preceded by flux_dev_lora call
            else:
                image_tool = openai_image_edit
        else:            
            # just use one of the image editing tools for now, even when there's a lora
            # init image takes precedence over lora
            if False: #loras:
                if loras[0].base_model == "sdxl":
                    image_tool = txt2img
                else:
                    image_tool = flux_dev_lora
            elif controlnet:
                image_tool = (
                    flux_dev  # todo: controlnet vs instructions is kind of a hack
                )
            else:
                image_tool = {
                    "flux": flux_kontext,
                    "seedream": seedream3,
                    "openai": openai_image_edit,
                    "sdxl": txt2img,
                    "nano_banana": nano_banana,
                }.get(model_preference, nano_banana)

    else:
        if text_precision:
            if loras:
                image_tool = openai_image_edit
            else:
                image_tool = openai_image_generate
        else:
            if loras:
                if loras[0].base_model == "sdxl":
                    image_tool = txt2img
                else:
                    image_tool = flux_dev_lora
            else:
                image_tool = {
                    "flux": flux_dev_lora,
                    "seedream": seedream3,
                    "openai": openai_image_generate,
                    "sdxl": txt2img,
                }.get(model_preference, flux_dev_lora)

    # Switch from Flux Dev Lora to Flux Dev if and only if 2 LoRAs or Controlnet
    if image_tool == flux_dev_lora:
        if two_faces or double_character:
            image_tool = flux_double_character
        elif len(loras) > 1 or controlnet:
            image_tool = flux_dev

    tool_calls = []

    #########################################################
    # Txt2Img
    if image_tool == txt2img:
        args = {
            "prompt": prompt,
            "n_samples": n_samples,
            "enforce_SDXL_resolution": True,
        }

        if seed:
            args["seed"] = seed

        if loras:
            args.update(
                {
                    "use_lora": True,
                    "lora": str(loras[0].id),
                    "lora_strength": lora_strength,
                }
            )

        if init_image:
            args.update(
                {
                    "init_image": init_image,
                    "use_init_image": True,
                    "denoise": 0.8,
                }
            )
            if controlnet:
                args.update(
                    {
                        "use_controlnet": True,
                        "controlnet_strength": 0.6,
                    }
                )

        args.update(aspect_ratio_to_dimensions(aspect_ratio))

        result = await txt2img.async_run(args, save_thumbnails=True)

    #########################################################
    # Flux Schnell
    elif image_tool == flux_schnell:
        if aspect_ratio == "auto":
            aspect_ratio = "1:1"

        args = {
            "prompt": prompt,
            "n_samples": n_samples,
            "aspect_ratio": aspect_ratio,
        }

        if seed:
            args["seed"] = seed

        print("Running flux_schnell", args)
        result = await flux_schnell.async_run(args, save_thumbnails=True)

    #########################################################
    # Flux Dev Lora
    elif image_tool == flux_dev_lora:
        args = {
            "prompt": prompt,
            "n_samples": n_samples,
        }

        if seed:
            args["seed"] = seed

        if init_image:
            args.update(
                {
                    "init_image": init_image,
                    "prompt_strength": 0.8 if init_image else 1.0,
                }
            )
        else:
            if aspect_ratio != "auto":
                args["aspect_ratio"] = aspect_ratio

        if loras:
            args.update(
                {
                    "lora": str(loras[0].id),
                    "lora_strength": lora_strength,
                }
            )
        else:
            args.update({"lora_strength": 0.0})

        print("Running flux_dev_lora", args)
        result = await flux_dev_lora.async_run(args, save_thumbnails=True)

    #########################################################
    # Flux Dev
    elif image_tool == flux_dev:
        args = {
            "prompt": prompt,
            "denoise": 1.0 if init_image else 0.8,
            "n_samples": n_samples,
            "speed_quality_tradeoff": 0.7,
        }

        if seed:
            args["seed"] = seed

        if init_image:
            args.update(
                {
                    "init_image": init_image,
                    "use_init_image": True,
                    "denoise": 0.75,
                }
            )
        else:
            if aspect_ratio == "auto":
                aspect_ratio = "1:1"

        if controlnet:
            args.update(
                {
                    "use_controlnet": True,
                    "controlnet_strength": 0.6,
                }
            )

        if loras:
            args.update(
                {
                    "use_lora": True,
                    "lora": str(loras[0].id),
                    "lora_strength": lora_strength,
                }
            )
        else:
            args.update({"lora_strength": 0.0})

        if loras and len(loras) > 1:
            args.update(
                {
                    "use_lora2": True,
                    "lora2": str(loras[1].id),
                    "lora2_strength": lora2_strength,
                }
            )
        else:
            args.update({"lora2_strength": 0.0})

        args.update(aspect_ratio_to_dimensions(aspect_ratio))

        print("Running flux_dev", args)
        result = await flux_dev.async_run(args, save_thumbnails=True)
        # Todo: incorporate style_image / style_strength ?

    #########################################################
    # Flux Double Character
    elif image_tool == flux_double_character:
        # Set the two LoRAs
        if len(loras) < 2:
            raise Exception("flux_double_character requires exactly 2 LoRAs")

        for l, lora in enumerate(loras):
            prompt = prompt.replace(lora.name, f"subj_{l+1}")

        args = {
            "prompt": prompt,
            "n_samples": n_samples,
            "speed_quality_slider": 0.4,
            "lora": str(loras[0].id),
            "lora2": str(loras[1].id),
        }
        args.update(aspect_ratio_to_dimensions(aspect_ratio))

        if seed:
            args["seed"] = seed

        # Note: flux_double_character doesn't support init_image, so we ignore it
        print("Running flux_double_character", args)
        result = await flux_double_character.async_run(args, save_thumbnails=True)

    #########################################################
    # Flux Kontext
    elif image_tool == flux_kontext:
        if aspect_ratio == "auto":
            aspect_ratio = "match_input_image"

        args = {
            "prompt": prompt,
            "init_image": init_image,
            "n_samples": n_samples,
            "aspect_ratio": aspect_ratio,
            "fast": False,
        }

        if seed:
            args["seed"] = seed

        print("Running flux_kontext", args)
        result = await flux_kontext.async_run(args, save_thumbnails=True)

    #########################################################
    # Nano Banana
    elif image_tool == nano_banana:
        args = {
            "prompt": prompt,
            "image_input": reference_images,  # Use all reference images
            "n_samples": n_samples,
            "output_format": "png"
        }

        if seed:
            args["seed"] = seed

        print("Running Nano Banana !!!", args)
        result = await nano_banana.async_run(args, save_thumbnails=True)

    #########################################################
    # OpenAI Image Generate
    elif image_tool == openai_image_generate:
        args = {
            "prompt": prompt,
            "n_samples": n_samples,
            "quality": "high",
        }
        if aspect_ratio in ["21:9", "16:9", "3:2", "4:3"]:
            args["size"] = "1536x1024"
        elif aspect_ratio in ["3:4", "2:3", "9:16", "9:21"]:
            args["size"] = "1024x1536"
        elif aspect_ratio in ["5:4", "1:1", "4:5"]:
            args["size"] = "1024x1024"
        else:
            args["size"] = "auto"

        if user:
            args["user"] = str(user)

        print("Running openai_image_generate", args)
        result = await openai_image_generate.async_run(args, save_thumbnails=True)

    #########################################################
    # OpenAI Image Edit
    elif image_tool == openai_image_edit:
        if loras:
            try:
                args_pre = {
                    "prompt": prompt,
                    "n_samples": n_samples,
                    "lora": str(loras[0].id),
                    "lora_strength": lora_strength,
                }

                if init_image:
                    args_pre.update(
                        {
                            "init_image": init_image,
                            "prompt_strength": 1.0 if init_image else 0.8,
                        }
                    )
                else:
                    if aspect_ratio != "auto":
                        args_pre["aspect_ratio"] = aspect_ratio

                if loras[0].base_model == "sdxl":
                    args_pre.update(
                        {
                            "enforce_SDXL_resolution": True,
                            "use_lora": True,
                        }
                    )
                    if init_image:
                        args_pre.update(
                            {
                                "use_init_image": True,
                            }
                        )
                        if controlnet:
                            args_pre.update(
                                {
                                    "use_controlnet": True,
                                    "controlnet_strength": 0.6,
                                }
                            )
                    result = await txt2img.async_run(args_pre)
                    print("init image result", result)
                    filename = result["output"][0]["filename"]
                    init_image = get_full_url(filename)
                    tool_calls.append(
                        {"tool": txt2img.key, "args": args_pre, "output": init_image}
                    )
                    intermediate_outputs["lora_init_image"] = result["output"]
                else:
                    result = await flux_dev_lora.async_run(args_pre)
                    print("init image result", result)
                    filename = result["output"][0]["filename"]
                    init_image = get_full_url(filename)
                    tool_calls.append(
                        {
                            "tool": flux_dev_lora.key,
                            "args": args_pre,
                            "output": init_image,
                        }
                    )
                    intermediate_outputs["lora_init_image"] = result["output"]

                prompt = f"This was the prompt for the image you see here: {prompt}. Regenerate this exact image in this exact style, as faithfully to the original image as possible, except completely redo any poorly rendered or illegible text rendered that doesn't match what's in the prompt."

            except Exception as e:
                print(
                    "Error in flux_dev_lora step, so just using openai_image_generate",
                    e,
                )
                raise e

        args = {
            "prompt": prompt,
            "n_samples": n_samples,
            "input_fidelity": "high",
            "size": "auto",
        }

        if user:
            args["user"] = str(user)

        if init_image:
            args["image"] = [init_image]
            print("Running openai_image_edit", args)
            result = await openai_image_edit.async_run(args, save_thumbnails=True)

        else:
            print("No init image, fall back on openai_image_generate", args)
            result = await openai_image_generate.async_run(args, save_thumbnails=True)

    #########################################################
    # Seedream 3
    elif image_tool == seedream3:
        args = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio if aspect_ratio != "auto" else "16:9",
            "size": "regular",
        }

        if init_image:
            args["image"] = init_image
            args.pop("aspect_ratio", None)

        if seed:
            args["seed"] = seed

        print("Running Seedream3", args)
        result = await seedream3.async_run(args, save_thumbnails=True)

    else:
        raise Exception("Invalid args", args, image_tool)

    #########################################################
    # Final result
    print("result", result)
    if result.get("status") == "failed":
        raise Exception(f"Error in /create: {result.get('error')}")
    
    final_result = get_full_url(result["output"][0]["filename"])
    print("final result", final_result)

    # Add sub tool call to tool_calls
    tool_calls.append({"tool": image_tool.key, "args": args, "output": final_result})

    # insert args urls
    for tool_call in tool_calls:
        for key, value in tool_call["args"].items():
            if key in ["init_image", "start_image", "end_image", "image"]:
                if isinstance(value, dict) and "filename" in value:
                    tool_call["args"][key] = get_full_url(value["filename"])
                else:
                    tool_call["args"][key] = value

    final_result = {
        "output": final_result, 
        "subtool_calls": tool_calls, 
        "intermediate_outputs": intermediate_outputs
    }
    if intermediate_outputs:
        final_result["intermediate_outputs"] = intermediate_outputs
    
    return final_result


async def handle_video_creation(args: dict, user: str = None, agent: str = None):
    """Handle video creation - copied from create_video tool handler"""

    # veo3 is enabled by default
    # if a specific user is provided (e.g. from the website or api), check if paying user has access to veo3 and disable it if not
    veo3_enabled = True
    if user:
        user = User.from_mongo(user)
        
        # if agent's owner pays, check their feature flags, otherwise user's
        if agent:
            agent = Agent.from_mongo(agent)
            if agent.owner_pays in ["full", "deployments"]:
                paying_user = User.from_mongo(agent.owner)
            else:
                paying_user = user
        
        veo3_enabled = any([
            t for t in paying_user.featureFlags 
            if t in ["tool_access_veo3", "preview"]
        ])

    runway = Tool.load("runway")
    runway3 = Tool.load("runway3")  # Load Runway Aleph
    kling = Tool.load("kling")
    kling_pro = Tool.load("kling")
    seedance1 = Tool.load("seedance1")
    veo2 = Tool.load("veo2")
    veo3 = Tool.load("veo3") if veo3_enabled else None
    hedra = Tool.load("hedra")
    create = Tool.load("create")
    thinksound = Tool.load("thinksound")

    prompt = args["prompt"]
    # n_samples = args.get("n_samples", 1)
    reference_images = args.get("reference_images", [])
    reference_video = args.get("reference_video", None)  # Extract reference_video
    start_image = reference_images[0] if len(reference_images) > 0 else None
    end_image = reference_images[1] if len(reference_images) > 1 else None
    seed = args.get("seed", None)
    lora_strength = args.get("lora_strength", 0.75)
    aspect_ratio = args.get("aspect_ratio", "auto")
    quality = args.get("quality", "standard")
    model_preference = args.get("model_preference", "seedance").lower()
    duration = args.get("duration", 5)
    extras = args.get("extras", [])
    talking_head = "talking_head" in extras
    audio = args.get("audio", None)
    sound_effects = args.get("sound_effects", None)

    intermediate_outputs = {}

    if end_image:
        assert start_image, "Must provide init_image if end_image is provided"

    # get loras
    loras = get_loras(args.get("lora"), args.get("lora2"))

    # Rules
    if reference_video:
        # Always use Runway Aleph for video-to-video style transfer
        video_tool = runway3
    elif talking_head and audio:
        video_tool = hedra
    elif quality == "standard":
        video_tool = {
            "kling": kling,
            "runway": runway,
            "seedance": seedance1,
            "veo": veo2,
        }.get(model_preference, veo2)
    elif quality == "pro":
        if veo3_enabled:
            if sound_effects:
                video_tool = veo3
            else:
                video_tool = {"kling": kling, "seedance": seedance1, "veo": veo2}.get(
                    model_preference, veo2
                )
        else:
            video_tool = {"kling": kling, "seedance": seedance1, "veo": veo2}.get(
                model_preference, veo2
            )

    print("Tool selected", video_tool.key)

    tool_calls = []

    # If there is no start image, generate one for any of the following reasons:
    # - Using Runway, because it requires one
    # - Lora is set, so we want to do img2vid with a lora-applied image instead of txt2vid
    # Otherwise, can just do txt2vid without a start image
    if not start_image:
        if video_tool in [runway, hedra] or loras:
            print("Generating start image with Lora")
            args = {"prompt": prompt}
            if loras:
                args.update(
                    {
                        "lora": str(loras[0].id),
                        "lora_strength": lora_strength,
                    }
                )
            try:
                result = await create.async_run(args, save_thumbnails=True)
                start_image = get_full_url(result["output"][0]["filename"])
                tool_calls.append({
                    "tool": create.key, 
                    "args": args, 
                    "output": start_image
                })
                intermediate_outputs["create_start_image"] = result["output"]
            except Exception as e:
                raise Exception(
                    "Error generating start image for img2vid. Try generating it yourself first with the 'create' tool, and then use it as start_image. Original error: {}".format(
                        e
                    )
                )

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
            aspect_ratio, "runway3" if end_image else "runway4", start_image_attributes
        )

        args = {
            "prompt_text": prompt,
            "start_image": start_image,  # Runway requires start image
            "model": "gen4_turbo",
            "duration": duration,
            "ratio": aspect_ratio,
        }

        if aspect_ratio != "auto":
            args["ratio"] = aspect_ratio

        # If ending image, must use gen3a_turbo
        if end_image:
            args.update(
                {
                    "end_image": end_image,
                    "model": "gen3a_turbo",
                }
            )

        if seed:
            args["seed"] = seed

        print("Running Runway", args)
        result = await runway.async_run(args, save_thumbnails=True)

    #########################################################
    # Kling
    elif video_tool == kling:
        # Kling can only produce 5 or 10s videos
        duration = 10 if duration > 7.5 else 5

        # Snap aspect ratio to closest Kling Pro preset
        aspect_ratio = snap_aspect_ratio_to_model(
            aspect_ratio, "kling", start_image_attributes
        )

        args = {"prompt": prompt, "duration": duration}

        if start_image:
            args.update(
                {
                    "start_image": start_image,
                }
            )

        # If an end image is requested, fall back to Kling 1.6 Pro which supports it
        if end_image:
            args.update(
                {
                    "end_image": end_image,
                    "quality": "medium",
                }
            )

        print(f"Running Kling {args.get('quality')}", args)

        if "start_image" in args:
            args.update(
                {
                    "mode": quality,
                }
            )
            result = await kling.async_run(args, save_thumbnails=True)
        else:
            args.update(
                {
                    "aspect_ratio": aspect_ratio,
                    "quality": "high",  # use Kling 2 optimistically
                }
            )
            result = await kling_pro.async_run(args, save_thumbnails=True)

    #########################################################
    # Seedance
    elif video_tool == seedance1:
        # Seedance can only produce 5 or 10s videos
        duration = 10 if duration > 7.5 else 5

        # Snap aspect ratio to closest Kling Pro preset
        aspect_ratio = snap_aspect_ratio_to_model(
            aspect_ratio, "seedance1", start_image_attributes
        )

        args = {
            "prompt": prompt,
            "duration": duration,
            "resolution": "1080p" if quality == "pro" else "480p",
        }

        if aspect_ratio != "auto":
            args["aspect_ratio"] = aspect_ratio

        if start_image:
            args.update(
                {
                    "image": start_image,
                }
            )

        if seed:
            args["seed"] = seed

        print("Running Seedance1", args)
        result = await seedance1.async_run(args, save_thumbnails=True)

    #########################################################
    # Veo-2
    elif video_tool == veo2:
        # Veo can only produce 5-8s videos
        duration = min(duration, 8)

        # Snap aspect ratio to closest Veo2 preset
        aspect_ratio = snap_aspect_ratio_to_model(
            aspect_ratio, "veo2", start_image_attributes
        )

        args = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        if start_image:
            args.update(
                {
                    "image": start_image,
                }
            )

        # if end_image:
        #     args.update({
        #         "end_image": end_image,
        #     })

        print("Running Veo2", args)
        result = await veo2.async_run(args, save_thumbnails=True)

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

        if start_image:
            args.update({
                "image": start_image,
            })

        if seed:
            args["seed"] = seed

        print("Running Veo3", args)
        result = await veo3.async_run(args, save_thumbnails=True)

    #########################################################
    # Hebra
    elif video_tool == hedra:
        # Snap aspect ratio to closest Hebra preset
        aspect_ratio = snap_aspect_ratio_to_model(
            aspect_ratio, "hedra", start_image_attributes
        )

        args = {
            "image": start_image,
            "prompt": prompt,
            "audio": audio,
            "aspectRatio": aspect_ratio,
        }

        print("Running Hebra", args)
        result = await hedra.async_run(args, save_thumbnails=True)

    #########################################################
    # Runway3 (Aleph)
    elif video_tool == runway3:
        # Snap aspect ratio to closest Runway preset
        aspect_ratio = snap_aspect_ratio_to_model(
            aspect_ratio, "runway4", None  # Use runway4 presets for Aleph
        )

        args = {
            "input_video": reference_video,  # The video to stylize
            "prompt_text": prompt,  # Style description
            "ratio": aspect_ratio,
        }

        # Add style reference images if provided
        if start_image:
            args["style_image"] = start_image  # Use first reference image as style image
        
        # Could also use a video as style reference (from second reference image if it's a video)
        # But for now we'll keep it simple
        
        if seed:
            args["seed"] = seed

        print("Running Runway3 (Aleph)", args)
        result = await runway3.async_run(args, save_thumbnails=True)

    else:
        raise Exception("Invalid video tool", video_tool)

    #########################################################
    # Final video is now generated
    print("result", result)
    final_video = get_full_url(result["output"][0]["filename"])
    print("final result", final_video)

    tool_calls.append({"tool": video_tool.key, "args": args, "output": final_video})

    # If sound effects are requested, try to add them
    if sound_effects and video_tool != veo3:
        print("----> creating sound effects")
        try:
            args = {
                "video": final_video,
                "caption": sound_effects,  # Use first 100 chars of prompt as caption
                # "cot": sound_effects,
                "cfg_scale": 5,
                "num_inference_steps": 24,
            }
            print("Running ThinkSound", args)
            sound_fx = await thinksound.async_run(args, save_thumbnails=True)
            final_video = get_full_url(sound_fx["output"][0]["filename"])
            print("Final result with sound effects", final_video)
            tool_calls.append(
                {"tool": thinksound.key, "args": args, "output": final_video}
            )

        except Exception as e:
            print(f"Error adding sound effects, just return video without it. Error: {e}")

    final_result = {
        "output": final_video, 
        "subtool_calls": tool_calls, 
    }
    if intermediate_outputs:
        final_result["intermediate_outputs"] = intermediate_outputs
    
    return final_result


def aspect_ratio_to_dimensions(aspect_ratio):
    if aspect_ratio == "auto":
        return {"size_from_input": True}
    elif aspect_ratio == "21:9":
        return {"width": 1536, "height": 640}
    elif aspect_ratio == "16:9":
        return {"width": 1344, "height": 768}
    elif aspect_ratio == "3:2":
        return {"width": 1216, "height": 832}
    elif aspect_ratio == "4:3":
        return {"width": 1152, "height": 896}
    elif aspect_ratio == "5:4":
        return {"width": 1200, "height": 960}
    elif aspect_ratio == "1:1":
        return {"width": 1024, "height": 1024}
    elif aspect_ratio == "4:5":
        return {"width": 960, "height": 1200}
    elif aspect_ratio == "3:4":
        return {"width": 896, "height": 1152}
    elif aspect_ratio == "2:3":
        return {"width": 832, "height": 1216}
    elif aspect_ratio == "9:16":
        return {"width": 768, "height": 1344}
    elif aspect_ratio == "9:21":
        return {"width": 640, "height": 1536}


def get_loras(lora1, lora2):
    loras = []
    for lora_id in [lora1, lora2]:
        if lora_id:
            if lora_id.lower() in ["null", "None"]:
                continue
            if not ObjectId.is_valid(str(lora_id)):
                continue
            lora = Model.from_mongo(lora_id)
            if not lora:
                raise Exception(f"Lora {lora_id} not found on {os.getenv('ENV')}")
            loras.append(lora)

    if len(loras) == 2 and "sdxl" in [lora.base_model for lora in loras]:
        print("Second Lora is not supported for SDXL")

    return loras


def get_closest_aspect_ratio_preset(aspect_ratio: float, presets: dict) -> str:
    """
    Get closest aspect ratio preset from a list of presets
    """
    closest_preset = None
    min_difference = float("inf")
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
        "runway3": {"16:9": 16 / 9, "9:16": 9 / 16},
        "runway4": {
            "21:9": 21 / 9,
            "16:9": 16 / 9,
            "4:3": 4 / 3,
            "1:1": 1 / 1,
            "3:4": 3 / 4,
            "9:16": 9 / 16,
        },
        "kling": {"16:9": 16 / 9, "1:1": 1 / 1, "9:16": 9 / 16},
        "veo2": {"16:9": 16 / 9, "9:16": 9 / 16},
        "hedra": {"16:9": 16 / 9, "1:1": 1 / 1, "9:16": 9 / 16},
        "seedance1": {
            "21:9": 21 / 9,
            "16:9": 16 / 9,
            "4:3": 4 / 3,
            "1:1": 1 / 1,
            "3:4": 3 / 4,
            "9:16": 9 / 16,
        },
    }[model_name]

    if aspect_ratio == "auto":
        # If there is a start image, snap to its aspect ratio
        if start_image_attributes:
            aspect_ratio = get_closest_aspect_ratio_preset(
                start_image_attributes["aspectRatio"], presets
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
            aspect_ratio = get_closest_aspect_ratio_preset(aspect_ratio_, presets)

    return aspect_ratio