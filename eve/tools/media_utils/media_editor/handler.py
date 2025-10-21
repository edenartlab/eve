from jinja2 import Template

from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent


init_message = """
You have been provided a set of attachments (URLs to media files), along with a request to perform certain media editing tasks. You have access to three specialized tools:

video_concat – Concatenate two or more videos into a longer video.

audio_video_combine – Overlay or mix an audio track onto a video track (the video may or may not already have audio).

ffmpeg_multitool – A general-purpose FFmpeg-based tool for any complex operations not handled by the first two tools.

Please follow these guidelines:

Whenever possible, prefer using video_concat and audio_video_combine.

Only use ffmpeg_multitool if the first two tools cannot achieve the desired result.

These tools accept the original file URLs as inputs (even though image thumbnails/previews might be provided for your reference).

You must adhere to the instructions given under the Task section below.

<Task>
Instructions from the user:
                           
{{ instructions }}

In your response, outline how you will solve the user’s request using the available tools, ensuring you respect the preferences and constraints described above.
</Task>
"""


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")

    session_post = Tool.load("session_post")

    instructions = args.get("instructions")
    media_files = args.get("media_files") or []

    user_message = Template(init_message).render(
        instructions=instructions,
    )

    result = await session_post.async_run({
        "role": "user",
        "agent_id": agent,
        "agent": "media-editor",
        "title": args.get("title") or "Media Editor Session",
        "content": user_message,
        "attachments": media_files,
        "pin": True,
        "prompt": True,
        "async": False,
        "extra_tools": ["video_concat", "audio_video_combine", "ffmpeg_multitool"],
    })

    if "error" in result:
        raise Exception(result["error"])
    
    
    print("media_editor result")
    print(result)

    #if "output" in result:
    output = result["output"]
    

    # session_id = result["output"][0]["session"]

    # return {"output": [{"session": session_id}]}

    print("---- HERE IS THE RESULT ----")
    print(result)
    print("---- HERE IS THE RESULT ----")
    return {"output": output}