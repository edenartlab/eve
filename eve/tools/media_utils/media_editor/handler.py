import anthropic
import json
import shutil
import subprocess
import os
import uuid
import shlex
from pathlib import Path
import asyncio
from jinja2 import Template
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from instructor.function_calls import openai_schema
from .... import eden_utils
from ....auth import get_my_eden_user


from ....agent import Agent
from ....agent.run_thread import async_prompt_thread
from ....agent.thread import UserMessage
from ....user import User
from ....agent.llm import async_prompt


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
    if not user:
        user = get_my_eden_user()

    user = User.from_mongo(user)
    agent = Agent.load("media-editor")
    tools = agent.get_tools(cache=True)
    thread = agent.request_thread()

    message = UserMessage(
        content=args["instructions"],
        attachments=args["media_files"]
    )

    print("\n\n\n========= init message ========")
    print(message)
    
    async for msg in async_prompt_thread(
        user, 
        agent, 
        thread, 
        message,
        tools, 
        force_reply=True, 
        use_thinking=False, 
        model="claude-3-7-sonnet-latest"
    ):
        print("\n\n===========")
        print(msg)


    prompt = "Given the initial instructions and the subsequent results, output all the resulting media files as a list of urls if the task was successful, otherwise output a string explaining the error."

    print("\n\n\n========= THE FINAL MESSAGES ========")
    for m in thread.get_messages() + [UserMessage(content=prompt)]:
        print()
        print(m)


    media_results = await async_prompt(
        thread.get_messages() + [UserMessage(content=prompt)],
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
