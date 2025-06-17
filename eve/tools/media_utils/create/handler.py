from typing import Optional, List
from pydantic import BaseModel, Field
from jinja2 import Template
from bson import ObjectId

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

async def handler(args: dict, user: str = None, agent: str = None):
    if user:
        user = User.from_mongo(ObjectId(user))
    else:
        user = get_my_eden_user()
    
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
            agent_id = args.get("agent")
            if not agent_id:
                agent = Agent.load("eve")
                agent_id = agent.id
            agent = Agent.from_mongo(agent_id)
            if agent:
                creator_agent_id = agent.id
                agent_name = agent.name
                agent_persona = agent.persona
                print(f"Using requested agent '{agent_name}' with ID: {creator_agent_id}")
                print(f"Agent persona: {agent_persona[:100]}..." if len(agent_persona) > 100 else f"Agent persona: {agent_persona}")
            else:
                print(f"Requested agent {agent_id} not found, falling back to default")
        except Exception as e:
            print(f"Error loading requested agent {args.get('agent')}: {e}")
    # Create session with the agent
    session = Session(
        owner=ObjectId(user.id),
        agents=[creator_agent_id],
        title=f"{output_type.title()} Creation Session",
        scenario=f"Creative {output_type} generation task",
        parent_session=args.get("parent_session", None)
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