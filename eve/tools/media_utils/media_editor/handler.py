from jinja2 import Template

from eve.tool import Tool, ToolContext


init_message = """
You have been provided a set of attachments (URLs to media files), along with a request to perform certain media editing tasks. You have access to three specialized tools:

video_concat – Concatenate two or more videos into a longer video.

audio_video_combine – Overlay or mix an audio track onto a video track (the video may or may not already have pre-existing audio).

ffmpeg_multitool – A general-purpose FFmpeg-based tool for any complex operations not handled by the first two tools.

Please follow these guidelines:
- Whenever possible, prefer using video_concat and audio_video_combine.
- When asked to mix audio to video, just use audio_video_combine -- you do not need to do fade-in or volume adjustments.
- Only use ffmpeg_multitool if the first two tools cannot achieve the desired result.
- These tools accept the original file URLs as inputs (even though image thumbnails/previews might be provided for your reference).
- You must adhere to the instructions given under the Task section below.

<Task>
Instructions from the user:
                           
{{ instructions }}

In your response, outline how you will solve the user’s request using the available tools, ensuring you respect the preferences and constraints described above.
</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    session_post = Tool.load("session_post")

    instructions = context.args.get("instructions")
    media_files = context.args.get("media_files") or []

    user_message = Template(init_message).render(
        instructions=instructions,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(context.agent),
        "agent": "eve",
        "title": context.args.get("title") or "Media Editor Session",
        "content": user_message,
        "attachments": media_files,
        "pin": True,
        "prompt": True,
        "async": False,
        "extra_tools": ["video_concat", "audio_video_combine", "ffmpeg_multitool"],
        "message_id": context.message,
        "tool_call_id": context.tool_call_id,
    }

    if context.args.get("resume_session"):
        args["session"] = context.args.get("resume_session")

    result = await session_post.async_run(args)

    # if "error" in result:
    #     raise Exception(result["error"])

    return result
