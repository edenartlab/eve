"""

TODO:
 - when two loras and they are faces, use flux_double_character
 - when text_precision, lora, and no init_image, first use flux_dev_lora and then openai_image_edit or flux_kontext
 - face_swap, flux_inpainting, outpaint, remix_flux_schnell
 
"""

"""


flux_schnell
flux_dev + flux_dev_lora
txt2img

flux_kontext
openai_image_generate
openai_image_edit

flux_double_character

special
- face_swap
X flux_inpainting
X outpaint
X remix_flux_schnell

--------------------------------

runway
kling_pro
veo2 + veo3

hedra

special
-vid2vid_sdxl
-video_FX
-texture_flow

--------------------------------


media_editor


----


prompt
(negative prompt not supported in kling)
starting image
 - image strength (prompt strength)
 - use_controlnet (flux_dev_lora -> flux_dev)
aspect ratio (auto)
lora
 - lora strength
lora 2 (flux_dev_lora -> flux_dev)
n_samples
seed

guidance
n_steps (low, medium, high) (low -> schnell)



* txt2img has "style image" / ip adapter




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

from typing import Optional, List
from pydantic import BaseModel, Field
from jinja2 import Template
from bson import ObjectId
import os

from ....auth import get_my_eden_user
from ....user import User
from ....agent import Agent
from ....agent.session.models import Session, ChatMessage, PromptSessionContext, ChatMessageRequestInput
from ....agent.session.session import _run_prompt_session_internal
from ....tool_constants import BASE_TOOLS
from ....tool import Tool
from fastapi import BackgroundTasks

MODEL = "claude-sonnet-4-20250514"

prompt_template = Template("""{% if persona -%}
You are {{ agent_name }}, and you should stay in character and adhere to your persona unless instructed otherwise in the task parameters.

Your persona: {{ persona }}

{% else -%}
You are a creative AI assistant specializing in media generation.
{% endif -%}

You have access to tools for creating {{ output_type }} content.

Available {{ output_type }} creation tools:
{% for tool in filtered_tools -%}
- {{ tool }}
{% endfor %}

Tool Selection Guidelines for {{ output_type.title() }} Creation:
{% if output_type == "image" -%}
- DEFAULT: Use flux_dev_lora for most image generation tasks
- Use other image tools (flux_dev, flux_schnell, txt2img, etc.) ONLY if the instructions specifically request alternative models or approaches
- flux_inpainting: for editing existing images
- outpaint: for extending image boundaries
- remix_flux_schnell: for style variations
{% elif output_type == "video" -%}
- runway: Use for faster and cheaper video generation jobs
- kling_pro: Use for longer duration and high quality video requirements
- veo2: Use for the highest quality but most expensive video generation (premium option)
- Choose based on the quality vs speed/cost requirements mentioned in the instructions
{% elif output_type == "audio" -%}
- elevenlabs: Use for vocal generation and speech synthesis
- stable_audio: Use for general audio generation unless specifically requested otherwise
- mmaudio: For music generation if specifically requested
- ace_step_musicgen: Alternative music generation option
{% endif %}

Task Parameters:
- Output type: {{ output_type }}
- Instructions: {{ instructions }}
- Number of samples: {{ samples }}
- Aspect ratio: {{ aspect_ratio }}
{% if reference_media -%}
- Reference media: {{ reference_media | join(', ') }}
{% endif -%}
{% if additional_context -%}
- Additional context: {{ additional_context }}
{% endif %}

{% if persona -%}
Please create {{ output_type }} content that reflects your persona and character while fulfilling the user's request. {% endif -%}Analyze the request and use the appropriate {{ output_type }} creation tools to generate the requested content. Consider:
1. The specific {{ output_type }} requirements from the instructions
2. The style and content requirements
3. The specified aspect ratio and number of samples
4. Any reference media provided
{% if persona -%}
5. How to incorporate your unique perspective and persona into the creative process
{% endif %}
6. Tool selection based on the guidelines above - follow the DEFAULT recommendations unless instructions specify otherwise

IMPORTANT: Follow the tool selection guidelines above. {% if output_type == "image" -%}Start with flux_dev_lora unless told otherwise.{% elif output_type == "video" -%}Choose runway for speed/cost, kling_pro for quality, or veo2 for premium quality.{% elif output_type == "audio" -%}Use elevenlabs for vocals, stable_audio for general audio.{% endif %}

Use your tools strategically to create high-quality {{ output_type }} content that matches the user's vision. If multiple samples are requested, create variations that explore different interpretations of the prompt.</Template>""")


class CreateResults(BaseModel):
    """Represents the outcome of a media creation process."""

    results: Optional[List[str]] = Field(
        ...,
        description="A list of URLs referencing the output media files."
    )
    error: Optional[str] = Field(
        None,
        description="An error message if the media creation process fails, otherwise None."
    )

async def handler2(args: dict, user: str = None, agent: str = None):
    
    if not user:
        user = get_my_eden_user()
    else:
        user = User.from_mongo(user)

    # Define tool categories by output type
    output_type = args.get("output_type", "image")
    tool_categories = {
        "image": [
            "flux_schnell", "flux_dev_lora", "flux_dev", "flux_kontext", "txt2img",
            "flux_inpainting", "outpaint", "remix_flux_schnell", "flux_double_character",
            "openai_image_generate", "openai_image_edit"
        ],
        "video": [
            "runway", "kling_pro", "veo2", "hedra", "vid2vid_sdxl", "video_FX", 
            "texture_flow"
        ],
        "audio": [
            "ace_step_musicgen", "elevenlabs", "mmaudio", "stable_audio", "zonos",
            "transcription"
        ]
    }
    
    # Filter tools by output type
    relevant_tool_names = tool_categories.get(output_type, [])
    # Always include media_editor for post-processing
    relevant_tool_names.append("media_editor")
    
    # Load the filtered tools
    available_tools = []
    for tool_name in relevant_tool_names:
        if tool_name in BASE_TOOLS:
            try:
                tool = Tool.from_file(tool_name)
                available_tools.append(tool)
            except Exception as e:
                print(f"Warning: Could not load tool {tool_name}: {e}")

    # Handle agent selection and persona loading
    creator_agent_id = None
    agent_name = None
    agent_persona = None
    
    # Check if a specific agent was requested
    if args.get("agent"):
        try:
            requested_agent_id = ObjectId(args["agent"])
            # Try to load the specific agent
            from ....agent.agent import Agent as AgentModel
            agent_doc = AgentModel.get_collection().find_one({"_id": requested_agent_id})
            if agent_doc:
                creator_agent_id = requested_agent_id
                agent_name = agent_doc.get("name", agent_doc.get("username", "Agent"))
                agent_persona = agent_doc.get("persona", "")
                print(f"Using requested agent '{agent_name}' with ID: {creator_agent_id}")
                print(f"Agent persona: {agent_persona[:100]}..." if len(agent_persona) > 100 else f"Agent persona: {agent_persona}")
            else:
                print(f"Requested agent {requested_agent_id} not found, falling back to default")
        except Exception as e:
            print(f"Error loading requested agent {args.get('agent')}: {e}")
    
    # If no specific agent or agent not found, try to find a default agent
    if not creator_agent_id:
        try:
            # First try to find any valid agent ID from the database
            from ....agent.agent import Agent as AgentModel
            
            # Look for an existing agent (try common names)
            agent_names = ["eve", "sidekick", "media-editor"]
            
            for name in agent_names:
                try:
                    agent_doc = AgentModel.get_collection().find_one({"username": name})
                    if agent_doc and "_id" in agent_doc:
                        creator_agent_id = agent_doc["_id"]
                        agent_name = agent_doc.get("name", name)
                        agent_persona = agent_doc.get("persona", "")
                        print(f"Found default agent '{name}' with ID: {creator_agent_id}")
                        break
                except Exception as e:
                    print(f"Could not find agent '{name}': {e}")
                    continue
            
            # If no agent found, use user ID as fallback
            if not creator_agent_id:
                creator_agent_id = ObjectId(user.id)
                print(f"Using user ID as agent ID: {creator_agent_id}")
                
        except Exception as e:
            print(f"Error finding agent: {e}")
            creator_agent_id = ObjectId(user.id)
            print(f"Fallback: using user ID as agent ID: {creator_agent_id}")

    # Create session with the agent
    session = Session(
        owner=ObjectId(user.id),
        agents=[creator_agent_id],
        title=f"{output_type.title()} Creation Session",
        scenario=f"Creative {output_type} generation task"
    )
    session.save()
    
    print(f"Created session {session.id} for {output_type} creation with agent {creator_agent_id}")

    # Prepare the instruction prompt with filtered tools and persona
    instruction_prompt = prompt_template.render(
        output_type=output_type,
        filtered_tools=relevant_tool_names,
        instructions=args["instructions"],
        samples=args.get("samples", 1),
        aspect_ratio=args.get("aspect_ratio", "16:9"),
        reference_media=args.get("reference_media", []),
        additional_context=args.get("additional_context", ""),
        agent_name=agent_name,
        persona=agent_persona
    )

    # Create the initial message
    initial_message = ChatMessageRequestInput(
        content=instruction_prompt,
        attachments=args.get("reference_media", [])
    )

    # Create session context
    context = PromptSessionContext(
        session=session,
        initiating_user_id=str(user.id),
        message=initial_message
    )

    print("\n\n\n========= Creating media with session-based prompting ========")
    print(f"Output type: {output_type}")
    print(f"Agent: {agent_name} ({creator_agent_id})" if agent_name else f"Agent ID: {creator_agent_id}")
    print(f"Persona: {'Yes' if agent_persona else 'No'}")
    print(f"Instructions: {args['instructions']}")
    print(f"Samples: {args.get('samples', 1)}")
    print(f"Aspect ratio: {args.get('aspect_ratio', '16:9')}")
    print(f"Reference media: {args.get('reference_media', [])}")
    print(f"Filtered tools: {relevant_tool_names}")
    print("--------------------------------")

    # Create background tasks (required but not used)
    background_tasks = BackgroundTasks()
    
    # Run the session and collect all updates
    all_updates = []
    async for update_data in _run_prompt_session_internal(context, background_tasks, stream=False):
        all_updates.append(update_data)
        print(f"Update: {update_data.get('type', 'unknown')}")

    print(f"\n\n\n========= Session completed with {len(all_updates)} updates ========")

    # Collect all generated media URLs from tool results
    generated_media = []
    
    # Look through all messages in the session for tool results
    session.reload()  # Refresh session data
    for message_id in session.messages:
        try:
            message = ChatMessage.from_mongo(message_id)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.status == "completed" and tool_call.result:
                        # Extract URLs from tool results
                        for result in tool_call.result:
                            if "output" in result:
                                for output_item in result["output"]:
                                    if isinstance(output_item, dict) and "url" in output_item:
                                        generated_media.append(output_item["url"])
                                    elif isinstance(output_item, str) and output_item.startswith(("http", "https")):
                                        generated_media.append(output_item)
        except Exception as e:
            print(f"Error processing message {message_id}: {e}")

    print(f"Generated media URLs: {generated_media}")

    if not generated_media:
        # If no media was generated, check if there was an error
        error_messages = []
        for message_id in session.messages:
            try:
                message = ChatMessage.from_mongo(message_id)
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.status == "failed" and tool_call.error:
                            error_messages.append(tool_call.error)
            except Exception as e:
                print(f"Error checking for errors in message {message_id}: {e}")
        
        if error_messages:
            raise Exception(f"Media creation failed: {'; '.join(error_messages)}")
        else:
            raise Exception("No media was generated. The agent may not have used any creation tools.")

    return {
        "output": generated_media
    }


from eve.tool import Tool
from eve.models import Model

from eve.s3 import get_full_url


"""



if lora:
if sdxl:
    image_tool = txt2img
else:
    if lora2 or controlnet:
        image_tool = flux_dev
    else:
        image_tool = flux_dev_lora
else:
if init_image:
    if mode == "face_swap":
        image_tool = face_swap (face_image)
    elif mode == "outpaint":
        image_tool = outpaint (init_image)
    elif precise:
        image_tool = openai_image_edit
    else:
        image_tool = flux_kontext
else:
    if precise:
        image_tool = openai_image_generate
    else:
        if fast:
            image_tool = flux_schnell
        else:
            image_tool = flux_dev_lora

"""




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
    print("THE SELECTED IMAGE TOOL", image_tool)
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

        print("use oaedit 1")

        if loras:

            print("use oaedit 2")

            try:
                print("use oaedit 3")

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

                print("use oaedit 4")
                print("init_image", init_image)
                print("prompt", prompt)

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

        print("use oaedit 5")
        print("args", init_image)
        

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

    print("THIS IS THE FINAL OUTPUT!!", final_result)
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
