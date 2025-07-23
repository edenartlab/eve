"""
Combined handler for both image and video creation.

TODO:
 x incorporate lora2
 x a bit too overeager on text precision?
 x when two loras and they are faces, use flux_double_character
 - face_swap, flux_inpainting, outpaint, remix_flux_schnell
 - deal with moderation errors for flux_kontext and openai tools, and errors in general
 - negative prompting
 - make init image strength a parameter
 - guidance, n_steps (low, medium, high) (low -> schnell)
 - txt2img has "style image" / ip adapter
 - check on n_samples
 - fix cost formula
 - video start_image_strength
 - enforce n_samples
 - vid2vid_sdxl, video_FX, texture_flow
 - no start image and aspect ratio == auto, predict good aspect ratio

MEDIA_EDITOR
- combine audio
- extract/remove/split tracks
- speed up/slow down?

"""

import os
from eve.s3 import get_full_url
from eve.tool import Tool
from eve.models import Model
from eve.user import User
from eve.eden_utils import get_media_attributes


async def handler(args: dict, user: str = None, agent: str = None):
    print("args", args)
    print("user", user)
    print("agent", agent)

    # Generate execution plan without executing anything
    execution_plan = _generate_execution_plan(args, user)
    print("***debug*** execution_plan", execution_plan)

    # Calculate costs for the execution plan
    execution_plan_with_costs = _calculate_plan_costs(execution_plan, user)
    print("***debug*** execution_plan_with_costs", execution_plan_with_costs)

    # Execute the plan step by step
    execution_result = await _execute_plan(execution_plan_with_costs["plan"])

    # Add cost information to the final result
    result = {
        **execution_result,
        "total_cost": execution_plan_with_costs["total_cost"],
        "cost_breakdown": execution_plan_with_costs["cost_breakdown"],
    }
    print("***debug*** result", result)

    return result


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


def _parse_image_args(args: dict) -> dict:
    """Parse and gather all image creation arguments"""
    prompt = args["prompt"]
    n_samples = args.get("n_samples", 1)
    init_image = args.get("init_image", None)
    extras = args.get("extras", [])
    text_precision = "text_precision" in extras
    double_character = "double_character" in extras
    controlnet = "controlnet" in extras
    seed = args.get("seed", None)
    aspect_ratio = args.get("aspect_ratio", "auto")
    model_preference = args.get("model_preference", "seedream").lower()

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

    return {
        "prompt": prompt,
        "n_samples": n_samples,
        "init_image": init_image,
        "extras": extras,
        "text_precision": text_precision,
        "double_character": double_character,
        "controlnet": controlnet,
        "seed": seed,
        "aspect_ratio": aspect_ratio,
        "model_preference": model_preference,
        "loras": loras,
        "lora_strength": lora_strength,
        "lora2_strength": lora2_strength,
        "two_faces": two_faces,
    }


def _determine_image_tool(parsed_args: dict) -> str:
    """Determine which image tool to use based on parsed arguments"""
    init_image = parsed_args["init_image"]
    text_precision = parsed_args["text_precision"]
    loras = parsed_args["loras"]
    controlnet = parsed_args["controlnet"]
    model_preference = parsed_args["model_preference"]
    two_faces = parsed_args["two_faces"]
    double_character = parsed_args["double_character"]

    # Determine tool
    if init_image:
        if text_precision:
            if loras:
                return "openai_image_edit"  # preceded by flux_dev_lora call
            else:
                return "openai_image_generate"
        else:
            if loras:
                if loras[0].base_model == "sdxl":
                    return "txt2img"
                else:
                    tool_name = "flux_dev_lora"
            elif controlnet:
                tool_name = (
                    "flux_dev"  # todo: controlnet vs instructions is kind of a hack
                )
            else:
                tool_name = {
                    "flux": "flux_kontext",
                    "seedream": "seedream3",
                    "openai": "openai_image_edit",
                    "sdxl": "txt2img",
                }.get(model_preference, "flux_kontext")
    else:
        if text_precision:
            if loras:
                return "openai_image_edit"
            else:
                return "openai_image_generate"
        else:
            if loras:
                if loras[0].base_model == "sdxl":
                    return "txt2img"
                else:
                    tool_name = "flux_dev_lora"
            else:
                tool_name = {
                    "flux": "flux_dev_lora",
                    "seedream": "seedream3",
                    "openai": "openai_image_generate",
                    "sdxl": "txt2img",
                }.get(model_preference, "seedream3")

    # Switch from Flux Dev Lora to Flux Dev if and only if 2 LoRAs or Controlnet
    if tool_name == "flux_dev_lora":
        if two_faces or double_character:
            return "flux_double_character"
        elif len(loras) > 1 or controlnet:
            return "flux_dev"

    return tool_name


def _build_image_tool_args(tool_name: str, parsed_args: dict, user: str = None) -> dict:
    """Build tool-specific arguments for the given tool"""
    if tool_name == "txt2img":
        return _build_txt2img_args(parsed_args)
    elif tool_name == "flux_schnell":
        return _build_flux_schnell_args(parsed_args)
    elif tool_name == "flux_dev_lora":
        return _build_flux_dev_lora_args(parsed_args)
    elif tool_name == "flux_dev":
        return _build_flux_dev_args(parsed_args)
    elif tool_name == "flux_double_character":
        return _build_flux_double_character_args(parsed_args)
    elif tool_name == "flux_kontext":
        return _build_flux_kontext_args(parsed_args)
    elif tool_name == "openai_image_generate":
        return _build_openai_image_generate_args(parsed_args, user)
    elif tool_name == "openai_image_edit":
        return _build_openai_image_edit_args(parsed_args, user)
    elif tool_name == "seedream3":
        return _build_seedream3_args(parsed_args)
    else:
        raise Exception(f"Unknown tool: {tool_name}")


def _build_txt2img_args(parsed_args: dict) -> dict:
    """Build arguments for txt2img tool"""
    args = {
        "prompt": parsed_args["prompt"],
        "n_samples": parsed_args["n_samples"],
        "enforce_SDXL_resolution": True,
    }

    if parsed_args["seed"]:
        args["seed"] = parsed_args["seed"]

    if parsed_args["loras"]:
        args.update(
            {
                "use_lora": True,
                "lora": str(parsed_args["loras"][0].id),
                "lora_strength": parsed_args["lora_strength"],
            }
        )

    if parsed_args["init_image"]:
        args.update(
            {
                "init_image": parsed_args["init_image"],
                "use_init_image": True,
                "denoise": 0.8,
            }
        )
        if parsed_args["controlnet"]:
            args.update(
                {
                    "use_controlnet": True,
                    "controlnet_strength": 0.6,
                }
            )

    args.update(aspect_ratio_to_dimensions(parsed_args["aspect_ratio"]))
    return args


def _build_flux_schnell_args(parsed_args: dict) -> dict:
    """Build arguments for flux_schnell tool"""
    aspect_ratio = parsed_args["aspect_ratio"]
    if aspect_ratio == "auto":
        aspect_ratio = "1:1"

    args = {
        "prompt": parsed_args["prompt"],
        "n_samples": parsed_args["n_samples"],
        "aspect_ratio": aspect_ratio,
    }

    if parsed_args["seed"]:
        args["seed"] = parsed_args["seed"]

    return args


def _build_flux_dev_lora_args(parsed_args: dict) -> dict:
    """Build arguments for flux_dev_lora tool"""
    args = {
        "prompt": parsed_args["prompt"],
        "n_samples": parsed_args["n_samples"],
    }

    if parsed_args["seed"]:
        args["seed"] = parsed_args["seed"]

    if parsed_args["init_image"]:
        args.update(
            {
                "init_image": parsed_args["init_image"],
                "prompt_strength": 0.8,
            }
        )
    else:
        if parsed_args["aspect_ratio"] != "auto":
            args["aspect_ratio"] = parsed_args["aspect_ratio"]

    if parsed_args["loras"]:
        args.update(
            {
                "lora": str(parsed_args["loras"][0].id),
                "lora_strength": parsed_args["lora_strength"],
            }
        )
    else:
        args.update({"lora_strength": 0.0})

    return args


def _build_flux_dev_args(parsed_args: dict) -> dict:
    """Build arguments for flux_dev tool"""
    args = {
        "prompt": parsed_args["prompt"],
        "denoise": 1.0 if parsed_args["init_image"] else 0.8,
        "n_samples": parsed_args["n_samples"],
        "speed_quality_tradeoff": 0.7,
    }

    if parsed_args["seed"]:
        args["seed"] = parsed_args["seed"]

    aspect_ratio = parsed_args["aspect_ratio"]
    if parsed_args["init_image"]:
        args.update(
            {
                "init_image": parsed_args["init_image"],
                "use_init_image": True,
                "denoise": 0.75,
            }
        )
    else:
        if aspect_ratio == "auto":
            aspect_ratio = "1:1"

    if parsed_args["controlnet"]:
        args.update(
            {
                "use_controlnet": True,
                "controlnet_strength": 0.6,
            }
        )

    if parsed_args["loras"]:
        args.update(
            {
                "use_lora": True,
                "lora": str(parsed_args["loras"][0].id),
                "lora_strength": parsed_args["lora_strength"],
            }
        )
    else:
        args.update({"lora_strength": 0.0})

    if parsed_args["loras"] and len(parsed_args["loras"]) > 1:
        args.update(
            {
                "use_lora2": True,
                "lora2": str(parsed_args["loras"][1].id),
                "lora2_strength": parsed_args["lora2_strength"],
            }
        )
    else:
        args.update({"lora2_strength": 0.0})

    args.update(aspect_ratio_to_dimensions(aspect_ratio))
    return args


def _build_flux_double_character_args(parsed_args: dict) -> dict:
    """Build arguments for flux_double_character tool"""
    loras = parsed_args["loras"]
    if len(loras) < 2:
        raise Exception("flux_double_character requires exactly 2 LoRAs")

    prompt = parsed_args["prompt"]
    print("HERE IS THE PROMPT", prompt)
    for idx, lora in enumerate(loras):
        prompt = prompt.replace(lora.name, f"subj_{idx+1}")
    print("HERE IS THE PROMPT 2", prompt)

    args = {
        "prompt": prompt,
        "n_samples": parsed_args["n_samples"],
        "speed_quality_slider": 0.4,
        "lora": str(loras[0].id),
        "lora2": str(loras[1].id),
    }
    args.update(aspect_ratio_to_dimensions(parsed_args["aspect_ratio"]))

    if parsed_args["seed"]:
        args["seed"] = parsed_args["seed"]

    return args


def _build_flux_kontext_args(parsed_args: dict) -> dict:
    """Build arguments for flux_kontext tool"""
    aspect_ratio = parsed_args["aspect_ratio"]
    if aspect_ratio == "auto":
        aspect_ratio = "match_input_image"

    args = {
        "prompt": parsed_args["prompt"],
        "init_image": parsed_args["init_image"],
        "n_samples": parsed_args["n_samples"],
        "aspect_ratio": aspect_ratio,
        "fast": False,
    }

    if parsed_args["seed"]:
        args["seed"] = parsed_args["seed"]

    return args


def _build_openai_image_generate_args(parsed_args: dict, user: str = None) -> dict:
    """Build arguments for openai_image_generate tool"""
    args = {
        "prompt": parsed_args["prompt"],
        "n_samples": parsed_args["n_samples"],
    }

    aspect_ratio = parsed_args["aspect_ratio"]
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

    return args


def _build_openai_image_edit_args(parsed_args: dict, user: str = None) -> dict:
    """Build arguments for openai_image_edit tool"""
    args = {
        "prompt": parsed_args["prompt"],
        "n_samples": parsed_args["n_samples"],
        "size": "auto",
    }

    if user:
        args["user"] = str(user)

    if parsed_args["init_image"]:
        args["image"] = [parsed_args["init_image"]]

    return args


def _build_seedream3_args(parsed_args: dict) -> dict:
    """Build arguments for seedream3 tool"""
    args = {
        "prompt": parsed_args["prompt"],
        "aspect_ratio": parsed_args["aspect_ratio"]
        if parsed_args["aspect_ratio"] != "auto"
        else "16:9",
        "size": "regular",
    }

    if parsed_args["init_image"]:
        args["image"] = parsed_args["init_image"]
        args.pop("aspect_ratio", None)

    if parsed_args["seed"]:
        args["seed"] = parsed_args["seed"]

    return args


def _parse_video_args(args: dict, user: str = None) -> dict:
    """Parse and gather all video creation arguments"""
    # veo3 is enabled by default
    # if a specific user is provided (e.g. from the website or api), check if they have access to veo3 and disable it if not
    veo3_enabled = True
    if user:
        user_obj = User.from_mongo(user)
        veo3_enabled = "tool_access_veo3" in user_obj.featureFlags

    prompt = args["prompt"]
    start_image = args.get(
        "init_image", None
    )  # Map init_image to start_image for video
    end_image = args.get("end_image", None)
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

    if end_image:
        assert start_image, "Must provide init_image if end_image is provided"

    # get loras
    loras = get_loras(args.get("lora"), args.get("lora2"))

    return {
        "veo3_enabled": veo3_enabled,
        "prompt": prompt,
        "start_image": start_image,
        "end_image": end_image,
        "seed": seed,
        "lora_strength": lora_strength,
        "aspect_ratio": aspect_ratio,
        "quality": quality,
        "model_preference": model_preference,
        "duration": duration,
        "extras": extras,
        "talking_head": talking_head,
        "audio": audio,
        "sound_effects": sound_effects,
        "loras": loras,
    }


def _determine_video_tool(parsed_args: dict) -> str:
    """Determine which video tool to use based on parsed arguments"""
    quality = parsed_args["quality"]
    talking_head = parsed_args["talking_head"]
    audio = parsed_args["audio"]
    model_preference = parsed_args["model_preference"]
    veo3_enabled = parsed_args["veo3_enabled"]
    sound_effects = parsed_args["sound_effects"]
    start_image = parsed_args["start_image"]

    # Rules
    if talking_head and audio:
        return "hedra"
    elif quality == "standard":
        return {
            "kling": "kling",
            "runway": "runway",
            "seedance": "seedance1",
            "veo": "veo2",
        }.get(model_preference, "veo2")
    elif quality == "pro":
        if veo3_enabled:
            if sound_effects and not start_image:
                return "veo3"
            else:
                return {"kling": "kling", "seedance": "seedance1", "veo": "veo2"}.get(
                    model_preference, "veo2"
                )
        else:
            return {"kling": "kling", "seedance": "seedance1", "veo": "veo2"}.get(
                model_preference, "veo2"
            )

    return "veo2"  # default fallback


def _build_video_tool_args(
    tool_name: str, parsed_args: dict, start_image_attributes: dict = None
) -> dict:
    """Build tool-specific arguments for the given video tool"""
    if tool_name == "runway":
        return _build_runway_args(parsed_args, start_image_attributes)
    elif tool_name == "kling":
        return _build_kling_args(parsed_args, start_image_attributes)
    elif tool_name == "seedance1":
        return _build_seedance1_args(parsed_args, start_image_attributes)
    elif tool_name == "veo2":
        return _build_veo2_args(parsed_args, start_image_attributes)
    elif tool_name == "veo3":
        return _build_veo3_args(parsed_args)
    elif tool_name == "hedra":
        return _build_hedra_args(parsed_args, start_image_attributes)
    else:
        raise Exception(f"Unknown video tool: {tool_name}")


def _build_runway_args(parsed_args: dict, start_image_attributes: dict = None) -> dict:
    """Build arguments for runway tool"""
    duration = parsed_args["duration"]
    end_image = parsed_args["end_image"]
    start_image = parsed_args["start_image"]
    prompt = parsed_args["prompt"]
    aspect_ratio = parsed_args["aspect_ratio"]
    seed = parsed_args["seed"]

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

    return args


def _build_kling_args(parsed_args: dict, start_image_attributes: dict = None) -> dict:
    """Build arguments for kling tool"""
    duration = parsed_args["duration"]
    prompt = parsed_args["prompt"]
    start_image = parsed_args["start_image"]
    end_image = parsed_args["end_image"]
    aspect_ratio = parsed_args["aspect_ratio"]
    quality = parsed_args["quality"]

    # Kling can only produce 5 or 10s videos
    duration = 10 if duration > 7.5 else 5

    # Snap aspect ratio to closest Kling Pro preset
    aspect_ratio = snap_aspect_ratio_to_model(
        aspect_ratio, "kling", start_image_attributes
    )

    args = {"prompt": prompt, "duration": duration}

    if start_image:
        args.update({"start_image": start_image})

    # If an end image is requested, fall back to Kling 1.6 Pro which supports it
    if end_image:
        args.update(
            {
                "end_image": end_image,
                "quality": "medium",
            }
        )

    if "start_image" in args:
        args.update({"mode": quality})
    else:
        args.update(
            {
                "aspect_ratio": aspect_ratio,
                "quality": "high",  # use Kling 2 optimistically
            }
        )

    return args


def _build_seedance1_args(
    parsed_args: dict, start_image_attributes: dict = None
) -> dict:
    """Build arguments for seedance1 tool"""
    prompt = parsed_args["prompt"]
    duration = parsed_args["duration"]
    quality = parsed_args["quality"]
    aspect_ratio = parsed_args["aspect_ratio"]
    start_image = parsed_args["start_image"]
    seed = parsed_args["seed"]

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
        args.update({"image": start_image})

    if seed:
        args["seed"] = seed

    return args


def _build_veo2_args(parsed_args: dict, start_image_attributes: dict = None) -> dict:
    """Build arguments for veo2 tool"""
    prompt = parsed_args["prompt"]
    duration = parsed_args["duration"]
    aspect_ratio = parsed_args["aspect_ratio"]
    start_image = parsed_args["start_image"]

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
        args.update({"image": start_image})

    return args


def _build_veo3_args(parsed_args: dict) -> dict:
    """Build arguments for veo3 tool"""
    prompt = parsed_args["prompt"]
    sound_effects = parsed_args["sound_effects"]
    duration = parsed_args["duration"]
    seed = parsed_args["seed"]

    # Veo can only produce 5-8s videos
    duration = min(duration, 8)

    args = {
        "prompt": f"{prompt}. {sound_effects}",
        "duration": duration,
    }

    if seed:
        args["seed"] = seed

    return args


def _build_hedra_args(parsed_args: dict, start_image_attributes: dict = None) -> dict:
    """Build arguments for hedra tool"""
    start_image = parsed_args["start_image"]
    prompt = parsed_args["prompt"]
    audio = parsed_args["audio"]
    aspect_ratio = parsed_args["aspect_ratio"]

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

    return args


def _generate_execution_plan(args: dict, user: str = None) -> list:
    """Generate a complete execution plan without executing anything"""
    output_type = args.get("output", "image")

    if output_type == "image":
        return _generate_image_plan(args, user)
    elif output_type == "video":
        return _generate_video_plan(args, user)
    else:
        raise Exception(f"Invalid output type: {output_type}")


def _generate_image_plan(args: dict, user: str = None) -> list:
    """Generate execution plan for image creation"""
    parsed_args = _parse_image_args(args)
    tool_name = _determine_image_tool(parsed_args)

    plan = []

    # Check if we need a preprocessing step for OpenAI with LoRA
    if tool_name == "openai_image_edit" and parsed_args["loras"]:
        # Step 1: Generate image with LoRA
        lora_tool_name = (
            "txt2img"
            if parsed_args["loras"][0].base_model == "sdxl"
            else "flux_dev_lora"
        )
        lora_args = _build_lora_preprocessing_args(parsed_args, lora_tool_name)

        plan.append(
            {
                "step": 1,
                "tool_name": lora_tool_name,
                "args": lora_args,
                "output_key": "lora_image",
                "is_intermediate": True,
            }
        )

        # Step 2: Use generated image with OpenAI
        # Update parsed_args to reference the lora_image output
        parsed_args["init_image"] = "{lora_image}"  # Reference to step 1 output
        parsed_args["prompt"] = (
            f"This was the prompt for the image you see here: {parsed_args['prompt']}. Regenerate this exact image in this exact style, as faithfully to the original image as possible, except completely redo any poorly rendered or illegible text rendered that doesn't match what's in the prompt."
        )

        openai_args = _build_image_tool_args(tool_name, parsed_args, user)
        plan.append(
            {
                "step": 2,
                "tool_name": tool_name,
                "args": openai_args,
                "output_key": "final_result",
                "is_intermediate": False,
            }
        )
    else:
        # Single step execution
        tool_args = _build_image_tool_args(tool_name, parsed_args, user)
        plan.append(
            {
                "step": 1,
                "tool_name": tool_name,
                "args": tool_args,
                "output_key": "final_result",
                "is_intermediate": False,
            }
        )

    return plan


def _generate_video_plan(args: dict, user: str = None) -> list:
    """Generate execution plan for video creation"""
    parsed_args = _parse_video_args(args, user)
    tool_name = _determine_video_tool(parsed_args)

    # Handle special case: Veo-3 doesn't support start images, so fall back to veo-2
    if parsed_args["start_image"] and tool_name == "veo3":
        tool_name = "veo2"

    plan = []

    # Check if we need to generate a start image
    if not parsed_args["start_image"] and (
        tool_name in ["runway", "hedra"] or parsed_args["loras"]
    ):
        # Step 1: Generate start image
        create_args = {"prompt": parsed_args["prompt"]}
        if parsed_args["loras"]:
            create_args.update(
                {
                    "lora": str(parsed_args["loras"][0].id),
                    "lora_strength": parsed_args["lora_strength"],
                }
            )

        plan.append(
            {
                "step": 1,
                "tool_name": "create",
                "args": create_args,
                "output_key": "start_image",
                "is_intermediate": True,
            }
        )

        # Update parsed_args to reference the generated start image
        parsed_args["start_image"] = "{start_image}"

    # Main video generation step
    step_num = len(plan) + 1

    # Get start image attributes if we have a start image
    start_image_attributes = None
    if parsed_args["start_image"] and not parsed_args["start_image"].startswith("{"):
        start_image_attributes, _ = get_media_attributes(parsed_args["start_image"])

    tool_args = _build_video_tool_args(tool_name, parsed_args, start_image_attributes)

    plan.append(
        {
            "step": step_num,
            "tool_name": tool_name,
            "args": tool_args,
            "output_key": "video_result",
            "is_intermediate": bool(
                parsed_args["sound_effects"] and tool_name != "veo3"
            ),
        }
    )

    # Check if we need sound effects step
    if parsed_args["sound_effects"] and tool_name != "veo3":
        step_num += 1
        sound_args = {
            "video": "{video_result}",
            "caption": parsed_args["sound_effects"],
            "cfg_scale": 5,
            "num_inference_steps": 24,
        }

        plan.append(
            {
                "step": step_num,
                "tool_name": "thinksound",
                "args": sound_args,
                "output_key": "final_result",
                "is_intermediate": False,
            }
        )
    else:
        # Update the previous step to be the final result
        plan[-1]["output_key"] = "final_result"
        plan[-1]["is_intermediate"] = False

    return plan


def _calculate_plan_costs(execution_plan: list, user: str = None) -> dict:
    """Calculate costs for each step in the execution plan and return total cost"""
    total_cost = 0.0
    plan_with_costs = []

    for step in execution_plan:
        step_with_cost = step.copy()
        tool_name = step["tool_name"]
        args = step["args"]

        try:
            # Load the tool to get cost calculation capabilities
            tool = Tool.load(tool_name)

            # Prepare args with defaults to handle ternary expressions in cost formulas
            prepared_args = tool.prepare_args(args, user=user)

            # Calculate cost using the prepared args
            step_cost = tool.calculate_cost(prepared_args)

            step_with_cost["cost"] = step_cost
            total_cost += step_cost

            print(f"Step {step['step']} ({tool_name}) cost: {step_cost}")

        except Exception as e:
            print(f"Warning: Could not calculate cost for {tool_name}: {e}")
            step_with_cost["cost"] = 0.0

        plan_with_costs.append(step_with_cost)

    return {
        "plan": plan_with_costs,
        "total_cost": total_cost,
        "cost_breakdown": [
            {
                "step": step["step"],
                "tool_name": step["tool_name"],
                "cost": step.get("cost", 0.0),
                "is_intermediate": step["is_intermediate"],
            }
            for step in plan_with_costs
        ],
    }


def _build_lora_preprocessing_args(parsed_args: dict, tool_name: str) -> dict:
    """Build arguments for LoRA preprocessing step"""
    args = {
        "prompt": parsed_args["prompt"],
        "n_samples": parsed_args["n_samples"],
        "lora": str(parsed_args["loras"][0].id),
        "lora_strength": parsed_args["lora_strength"],
    }

    if parsed_args["init_image"]:
        args.update(
            {
                "init_image": parsed_args["init_image"],
                "prompt_strength": 1.0,
            }
        )
    else:
        if parsed_args["aspect_ratio"] != "auto":
            args["aspect_ratio"] = parsed_args["aspect_ratio"]

    if tool_name == "txt2img":
        args.update(
            {
                "enforce_SDXL_resolution": True,
                "use_lora": True,
            }
        )
        if parsed_args["init_image"]:
            args.update({"use_init_image": True})
            if parsed_args["controlnet"]:
                args.update(
                    {
                        "use_controlnet": True,
                        "controlnet_strength": 0.6,
                    }
                )

    return args


async def _execute_plan(execution_plan: list) -> dict:
    """Execute the planned tool calls in sequence"""
    tool_calls = []
    step_outputs = {}  # Store outputs from each step

    for step in execution_plan:
        step_num = step["step"]
        tool_name = step["tool_name"]
        args = step["args"].copy()  # Don't modify the original plan
        output_key = step["output_key"]

        # Replace any references to previous step outputs
        args = _resolve_step_references(args, step_outputs)

        # Load and execute the tool
        tool = Tool.load(tool_name)
        print(f"Executing step {step_num}: {tool_name}", args)
        result = await tool.async_run(args)

        # Extract the output URL
        if tool_name == "kling" and "start_image" not in args:
            # Special case for kling_pro
            output_url = get_full_url(result["output"][0]["filename"])
        else:
            output_url = get_full_url(result["output"][0]["filename"])

        # Store the output for future steps
        step_outputs[output_key] = output_url

        # Add to tool_calls for the final result
        tool_calls.append({"tool": tool.key, "args": args, "output": output_url})

        print(f"Step {step_num} completed: {output_url}")

    # Get the final result
    final_output = step_outputs["final_result"]

    # Process URLs in tool calls (maintain compatibility with original format)
    for tool_call in tool_calls:
        for key, value in tool_call["args"].items():
            if key in ["init_image", "start_image", "end_image", "image", "video"]:
                if isinstance(value, dict) and "filename" in value:
                    tool_call["args"][key] = get_full_url(value["filename"])
                else:
                    tool_call["args"][key] = value

    return {"output": final_output, "subtool_calls": tool_calls}


def _resolve_step_references(args: dict, step_outputs: dict) -> dict:
    """Replace step references like {lora_image} with actual URLs"""
    resolved_args = {}

    for key, value in args.items():
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            # This is a reference to a previous step output
            reference_key = value[1:-1]  # Remove the braces
            if reference_key in step_outputs:
                resolved_args[key] = step_outputs[reference_key]
            else:
                raise Exception(
                    f"Step reference '{reference_key}' not found in previous outputs"
                )
        elif (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], str)
            and value[0].startswith("{")
        ):
            # Handle list format like ["image": ["{lora_image}"]]
            reference_key = value[0][1:-1]
            if reference_key in step_outputs:
                resolved_args[key] = [step_outputs[reference_key]]
            else:
                raise Exception(
                    f"Step reference '{reference_key}' not found in previous outputs"
                )
        else:
            resolved_args[key] = value

    return resolved_args
