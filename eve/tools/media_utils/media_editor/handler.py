from typing import Optional, List
from pydantic import BaseModel, Field
from jinja2 import Template

from ....auth import get_my_eden_user
from ....user import User
from ....agent import Agent
from ....agent.run_thread import async_prompt_thread
from ....agent.thread import UserMessage
from ....agent.llm import async_prompt


prompt_template = Template("""<Rules>
You are given a set of attachments (URLs to media files), along with a request to do some media editing operations on those input files.
                           
You have these three tools:
1) video_concat - concatenate two or more videos together into a longer video
2) audio_video_combine - overlay an audio track onto a video track (which may or may not have pre-existing audio)
3) ffmpeg_multitool - a general purpose tool which uses ffmpeg to do more complex tasks that the above tools cannot or fail to do.

You should generally prefer to use the video_concat and audio_video_combine tools whenever possible, and only fall back to ffmpeg_multitool.
                           
Note, these tools all accept the original URLs as inputs. Images and frames from videos are provided to you merely for your convenience, but you should pass on the URLs to the tools.
</Rules>
<Task>
Instructions from the user:

{{ instructions }}
</Task>""")


class MediaResults(BaseModel):
    """A collection of media files resulting from the media editor's tools."""

    results: Optional[List[str]] = Field(
        ...,
        description="A list of urls to media files to return to the user.",
    )
    error: Optional[str] = Field(
        None,
        description="Return an error message if and only if the media editor failed to accomplish the task.",
    )
    

async def handler(args: dict, user: str = None, agent: str = None):
    print("THE USER IS", user)
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


    prompt = "Given the initial instructions and the subsequent results, output all the resulting media files as a list of urls if the task was successful, otherwise output a string explaining the error."

    print("\n\n\n=================")
    all_messages = thread.get_messages() + [UserMessage(content=prompt)]
    for m in all_messages:
        print(m)

    media_results = await async_prompt(
        messages=all_messages,
        system_message=agent.persona,
        model="gpt-4o-mini",
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
