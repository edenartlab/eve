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



result_template = Template("""<Name>{{ name }}</Name>
<ChatLog>
Role: You are roleplaying as {{ name }} in a group chat. The following are the last {{ message_count }} messages. Note: "You" refers to your own messages.
---
{{ chat }}
---
</ChatLog>
<Task>
You will receive the next user message in this group chat. Note that the message may not be directed specifically to you. Use context to determine if it:
- Directly addresses you,
- References something you said,
- Is intended for another participant, or
- Is a general message.
Based on your analysis, generate a response containing:
- intention: Either "reply" or "ignore". Choose "reply" if the message is relevant or requests you; choose "ignore" if it is not.
- thought: A brief explanation of your reasoning regarding the message’s relevance and your decision.
- recall_knowledge: Whether to consult your background knowledge.
{{ reply_criteria }}
</Task>
{{ knowledge_description }}
<Message>
{{ message }}
</Message>
""")



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
    
    async for msg in async_prompt_thread(
        user, 
        agent, 
        thread, 
        UserMessage(
            content=args["instructions"],
            attachments=args["media_files"]
        ),
        tools, 
        force_reply=True, 
        use_thinking=False, 
        model="gpt-4o-mini"
    ):
        print(msg)


    prompt = "Given the initial instructions and the subsequent results, output all the resulting media files as a list of urls if the task was successful, otherwise output a string explaining the error."

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
