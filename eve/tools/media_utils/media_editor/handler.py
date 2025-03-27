from typing import Optional, List
from pydantic import BaseModel, Field
from jinja2 import Template

from ....auth import get_my_eden_user
from ....user import User
from ....agent import Agent
from ....agent.run_thread import async_prompt_thread
from ....agent.thread import UserMessage
from ....agent.llm import async_prompt

MODEL = "gpt-4o-mini"

prompt_template = Template("""You have been provided a set of attachments (URLs to media files), along with a request to perform certain media editing tasks. You have access to three specialized tools:

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
</Task>""")


class MediaResults(BaseModel):
    """Represents the outcome of a media editing process."""

    results: Optional[List[str]] = Field(
        ...,
        description="A list of URLs referencing the output media files."
    )
    error: Optional[str] = Field(
        None,
        description="An error message if the media editing process fails, otherwise None."
    )

async def handler(args: dict, user: str = None, agent: str = None):
    
    if not user:
        user = get_my_eden_user()
    else:
        user = User.from_mongo(user)

    
    agent = Agent.load("media-editor")
    tools = agent.get_tools(cache=True)
    thread = agent.request_thread()

    instruction_prompt = prompt_template.render(
        instructions=args["instructions"],
    )

    message = UserMessage(
        content=instruction_prompt,
        attachments=args["media_files"]
    )

    print("\n\n\n========= init message ========")
    print(message)
    print("--------------------------------")
    print(message.anthropic_schema(truncate_images=True))
    
    async for _ in async_prompt_thread(
        user, 
        agent, 
        thread, 
        message,
        tools, 
        force_reply=True, 
        use_thinking=False, 
        model="claude-3-7-sonnet-latest"
    ):
        pass


    prompt = "Based on the initial instructions and the editor's output, please provide one of the following:\n\n1. A JSON-compatible list of the resulting media file URLs, if successful.\n2. A string describing the error, if the process failed."

    print("\n\n\n=================")
    all_messages = thread.get_messages() + [UserMessage(content=prompt)]
    for m in all_messages:
        print(m)

    media_results = await async_prompt(
        messages=all_messages,
        system_message=agent.persona,
        model=MODEL,
        response_model=MediaResults,
    )

    print("======= media results")
    print(media_results)

    if media_results.error:
        raise Exception(media_results.error)
    else:
        return {
            "output": media_results.results
        }
