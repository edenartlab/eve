"""

TODO:
 - when two loras and they are faces, use flux_double_character
 - when text_precision, lora, and no init_image, first use flux_dev_lora and then openai_image_edit or flux_kontext
 - face_swap, flux_inpainting, outpaint, remix_flux_schnell
 - deal with moderation errors for flux_kontext and openai tools
 - negative prompting
 - make init image strength a parameter
 - guidance, n_steps (low, medium, high) (low -> schnell)
 - txt2img has "style image" / ip adapter
 

VIDEO
- runway
- kling_pro
- veo2 + veo3
- hedra
- special (vid2vid_sdxl, video_FX, texture_flow, mmaudio?)

AUDIO
- elevenlabs
- stable_audio
- mmaudio
- ace_step_musicgen
- zonos
- transcription


text (prompt, vocals)
duration
type [vocals, music, sound effects]
voice (optional)
elevenlabs->speech


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

from eve.tool import Tool
from eve.models import Model

from eve.s3 import get_full_url




async def handler(args: dict, user: str = None, agent: str = None):
    flux_schnell = Tool.load("flux_schnell")
    flux_dev_lora = Tool.load("flux_dev_lora")
    flux_dev = Tool.load("flux_dev")
    flux_kontext = Tool.load("flux_kontext")
    txt2img = Tool.load("txt2img")
    openai_image_edit = Tool.load("openai_image_edit")
    openai_image_generate = Tool.load("openai_image_generate")

    args1 = {
        # "prompt": "Banny sits in a park. Next to them there is a sign that reads 'Created a custom BytesIOWithName class: This extends BytesIO and adds a name attribute that the OpenAI API can use to detect the file format'",
        "prompt": "Fix the text on the sign to say 'Created a custom BytesIOWithName class: This extends BytesIO and adds a name attribute that the OpenAI API can use to detect the file format'",
        # "text_precision": True,
        "init_image": "https://dtut5r9j4w7j4.cloudfront.net/bebda03b7d255adf8a60a5873e6506d361ad39f9824756e721e3da74b682b61d.png",
        "lora": "67fa5bd1eeb0f51f6e8f3c0c"  # sdxl
        # "lora": "6766760643808b38016c64ce",  # flux banny
        # "lora2": "681ec3abc31f6ec3d96630bf"
    }
    args1 = {
        "prompt": "Make an infographic about the history of the internet. Make it very detailed",
        "text_precision": True,
    }
    # args = args1


    prompt = args["prompt"]
    n_samples = args.get("n_samples", 1)
    init_image = args.get("init_image", None)
    text_precision = args.get("text_precision", False)
    seed = args.get("seed", None)
    lora_strength = args.get("lora_strength", 0.75)
    aspect_ratio = args.get("aspect_ratio", "match_input_image")
    if aspect_ratio == "auto":
        aspect_ratio = "match_input_image"
    controlnet = args.get("controlnet", False)

    # get loras
    loras = get_loras(args.get("lora"), args.get("lora2"))


    print("\n\n\n\n\n--------------------------------")
    print("INIT IMAGE", init_image)
    print("TEXT PRECISION", text_precision)
    print("LORAS", loras)
    print("CONTROLNET", controlnet)
    print("--------------------------------")

    # Determine tool
    if init_image:
        if text_precision:
            if loras:
                image_tool = openai_image_edit # preceded by flux_dev_lora call
            else:
                image_tool = openai_image_generate
        else:
            if loras:
                if loras[0].base_model == "sdxl":
                    image_tool = txt2img
                else:
                    image_tool = flux_dev_lora
            elif controlnet:
                image_tool = flux_dev # todo: controlnet vs instructions is kind of a hack
            else:
                image_tool = flux_kontext
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
                image_tool = flux_dev_lora # flux_schnell 

    # Switch from Flux Dev Lora to Flux Dev if and only if 2 LoRAs or Controlnet
    if image_tool == flux_dev_lora:
        if len(loras) > 1 or controlnet:
            image_tool = flux_dev

    print("\n\n\n\n\n--------------------------------")
    print("THE SELECTED IMAGE TOOL", image_tool.key)
    print("--------------------------------")


    # Run the tool
    if image_tool == txt2img:
        args = {
            "prompt": prompt,
            "n_samples": n_samples,
            "enforce_SDXL_resolution": True,
        }

        if seed:
            args["seed"] = seed

        if loras:
            args.update({
                "use_lora": True,
                "lora": str(loras[0].id),
                "lora_strength": lora_strength,
            })

        if init_image:
            args.update({
                "init_image": init_image,
                "use_init_image": True,
                "denoise": 0.8,
            })
            if controlnet:
                args.update({
                    "use_controlnet": True,
                    "controlnet_strength": 0.6,
                })

        args.update(aspect_ratio_to_dimensions(aspect_ratio))

        result = await txt2img.async_run(args)

    if image_tool == flux_schnell:
        if aspect_ratio == "match_input_image":
            aspect_ratio = "1:1"
        
        args = {
            "prompt": prompt,
            "n_samples": n_samples,
            "aspect_ratio": aspect_ratio,
        }
        
        if seed:
            args["seed"] = seed
        
        print("Running flux_schnell", args)
        result = await flux_schnell.async_run(args)
    
    elif image_tool == flux_dev_lora:
        args = {
            "prompt": prompt,
            "n_samples": n_samples,
        }

        if seed:
            args["seed"] = seed

        if init_image:
            args.update({
                "init_image": init_image,
                "prompt_strength": 0.8 if init_image else 1.0
            })
        else:
            if aspect_ratio != "match_input_image":
                args["aspect_ratio"] = aspect_ratio

        if loras:
            args.update({
                "lora": str(loras[0].id),
                "lora_strength": lora_strength,
            })
        else:
            args.update({
                "lora_strength": 0.0
            })

        print("Running flux_dev_lora", args)
        result = await flux_dev_lora.async_run(args)

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
            args.update({
                "init_image": init_image, 
                "use_init_image": True,
                "denoise": 0.75,
            })
        else:
            if aspect_ratio == "match_input_image":
                aspect_ratio = "1:1"

        if controlnet:
            args.update({
                "use_controlnet": True,
                "controlnet_strength": 0.6,
            })

        if loras:
            args.update({
                "use_lora": True,
                "lora": str(loras[0].id),
                "lora_strength": lora_strength,
            })
        else:
            args.update({
                "lora_strength": 0.0
            })

        if loras and len(loras) > 1:
            args.update({
                "use_lora2": True,
                "lora2": str(loras[1].id),
                "lora2_strength": lora_strength,
            })
        else:
            args.update({
                "lora2_strength": 0.0
            })

        args.update(aspect_ratio_to_dimensions(aspect_ratio))

        print("Running flux_dev", args)
        result = await flux_dev.async_run(args)
        # Todo: incorporate style_image / style_strength ?

    elif image_tool == flux_kontext:
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
        result = await flux_kontext.async_run(args)

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
        result = await openai_image_generate.async_run(args)

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
                    args_pre.update({
                        "init_image": init_image,
                        "prompt_strength": 1.0 if init_image else 0.8
                    })
                else:
                    if aspect_ratio != "match_input_image":
                        args_pre["aspect_ratio"] = aspect_ratio

                if loras[0].base_model == "sdxl":
                    args_pre.update({
                        "enforce_SDXL_resolution": True,
                        "use_lora": True,
                    })
                    if init_image:
                        args_pre.update({
                            "use_init_image": True,
                        })
                        if controlnet:
                            args_pre.update({
                                "use_controlnet": True,
                                "controlnet_strength": 0.6,
                            })
                    result = await txt2img.async_run(args_pre)
                else:
                    result = await flux_dev_lora.async_run(args_pre)

                init_image = get_full_url(result["output"][0]["filename"])

                prompt = f"This was the prompt for the image you see here: {prompt}. Regenerate this exact image in this exact style, as faithfully to the original image as possible, except completely redo any poorly rendered or illegible text rendered that doesn't match what's in the prompt."

                print("oae init_image", init_image)
                print("oae prompt", prompt)

            except Exception as e:
                print("Error in flux_dev_lora step, so just using openai_image_generate", e)
                raise e

        args = {
            "prompt": prompt,
            "n_samples": n_samples,
            "quality": "high",
            "size": "auto"
        }

        if user:
            args["user"] = str(user)

        if init_image:
            args["image"] = [init_image]
            print("Running openai_image_edit", args)
            result = await openai_image_edit.async_run(args)

        else:
            print("No init image, fall back on openai_image_generate", args)
            result = await openai_image_generate.async_run(args)

    else:
        raise Exception("Invalid args", args, image_tool)

    assert "output" in result, "No output from image tool"
    assert len(result["output"]) == 1, "Expected 1 output from image tool"
    assert "filename" in result["output"][0], "No filename in output from image tool"
    
    final_result = get_full_url(result["output"][0]["filename"])

    print("final result", final_result)
    return {
        "output": final_result
    }


def aspect_ratio_to_dimensions(aspect_ratio):
    if aspect_ratio == "match_input_image":
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
        raise Exception("Second Lora is not supported for SDXL")
        
    return loras
