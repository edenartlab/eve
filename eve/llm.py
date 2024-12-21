import re
import sentry_sdk
from pprint import pprint
import traceback
import os
import asyncio
import openai
import anthropic
from enum import Enum
from typing import Optional, Dict, Any
from bson import ObjectId
from jinja2 import Template
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from instructor.function_calls import openai_schema
from typing import List, Optional, Dict, Any, Literal, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from . import sentry_sdk
from .tool import Tool
from .user import User
from .agent import Agent
from .thread import UserMessage, AssistantMessage, ToolCall, Thread


async def async_anthropic_prompt(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str] = "You are a helpful assistant.",
    model: str = "claude-3-5-sonnet-20241022",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = None,
    db: str = "STAGE",
):
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY env is not set")

    messages_json = [item for msg in messages for item in msg.anthropic_schema()]

    # print("--------------------------------")
    # pprint(messages_json)
    # print("--------------------------------")

    prompt = {
        "model": model,
        "max_tokens": 8192,
        "messages": messages_json,
        "system": system_message,
    }

    anthropic_client = anthropic.AsyncAnthropic()

    if tools or response_model:
        tools = [
            t.anthropic_schema(exclude_hidden=True) for t in (tools or {}).values()
        ]
        if response_model:
            tools.append(openai_schema(response_model).anthropic_schema)
            prompt["tool_choice"] = {"type": "tool", "name": response_model.__name__}
        prompt["tools"] = tools

    response = await anthropic_client.messages.create(**prompt)

    if response_model:
        return response_model(**response.content[0].input)

    else:
        content = ". ".join(
            [r.text for r in response.content if r.type == "text" and r.text]
        )
        tool_calls = [
            ToolCall.from_anthropic(r, db=db)
            for r in response.content
            if r.type == "tool_use"
        ]
        stop = response.stop_reason != "tool_use"
        return content, tool_calls, stop


async def async_openai_prompt(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str] = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",  # "gpt-4o-2024-08-06",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
    db: str = "STAGE",
):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY env is not set")

    messages_json = [item for msg in messages for item in msg.openai_schema()]
    if system_message:
        messages_json = [{"role": "system", "content": system_message}] + messages_json

    openai_client = openai.AsyncOpenAI()

    if response_model:
        response = await openai_client.beta.chat.completions.parse(
            model=model, messages=messages_json, response_format=response_model
        )
        return response.choices[0].message.parsed

    else:
        tools = (
            [t.openai_schema(exclude_hidden=True) for t in tools.values()]
            if tools
            else None
        )
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages_json, tools=tools
        )
        response = response.choices[0]
        content = response.message.content or ""
        tool_calls = [
            ToolCall.from_openai(t, db=db) for t in response.message.tool_calls or []
        ]
        stop = response.finish_reason != "tool_calls"

        return content, tool_calls, stop


@retry(
    retry=retry_if_exception(
        lambda e: isinstance(e, (openai.RateLimitError, anthropic.RateLimitError))
    ),
    wait=wait_exponential(multiplier=5, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
@retry(
    retry=retry_if_exception(
        lambda e: isinstance(
            e,
            (
                openai.APIConnectionError,
                openai.InternalServerError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
        )
    ),
    wait=wait_exponential(multiplier=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def async_prompt(
    messages: List[Union[UserMessage, AssistantMessage]],
    system_message: Optional[str] = "You are a helpful assistant.",
    model: str = "claude-3-5-sonnet-20241022",
    response_model: Optional[type[BaseModel]] = None,
    tools: Dict[str, Tool] = {},
    db: str = "STAGE",
):
    if model.startswith("claude"):
        return await async_anthropic_prompt(
            messages, system_message, model, response_model, tools, db
        )
    else:
        return await async_openai_prompt(
            messages, system_message, model, response_model, tools, db
        )


def anthropic_prompt(messages, system_message, model, response_model=None, tools=None):
    return asyncio.run(
        async_anthropic_prompt(messages, system_message, model, response_model, tools)
    )


def openai_prompt(messages, system_message, model, response_model=None, tools=None):
    return asyncio.run(
        async_openai_prompt(messages, system_message, model, response_model, tools)
    )


def prompt(messages, system_message, model, response_model=None, tools=None):
    return asyncio.run(
        async_prompt(messages, system_message, model, response_model, tools)
    )


class UpdateType(str, Enum):
    START_PROMPT = "start_prompt"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_COMPLETE = "tool_complete"
    ERROR = "error"
    UPDATE_COMPLETE = "update_complete"


models = ["claude-3-5-sonnet-20241022", "gpt-4o-mini", "gpt-4o-2024-08-06"]


# todo: `msg.error` not `msg.message.error`
class ThreadUpdate(BaseModel):
    type: UpdateType
    message: Optional[AssistantMessage] = None
    tool_name: Optional[str] = None
    tool_index: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


system_instructions = """In addition to the instructions above, follow these additional guidelines:
* In your response, do not include anything besides for your chat message. Do not include pretext, stage directions, or anything other than what you are saying.
* Do not apologize.
* Try to be concise. Do not be verbose.
"""

template = """<Summary>You are roleplaying as {{ name }}.</Summary>
<Description>
This is a description of {{ name }}.

{{ description }}
</Description>
<Instructions>
{{ instructions }}
</Instructions>
<System Instructions>
{{ system_instructions }}
</System Instructions>"""


async def async_think():
    # - think (gpt3)
    # - choose tools
    # - choose knowledge
    # - which tools to make available
    # - decide to reply
    # - make intentions
    pass


async def async_prompt_thread(
    db: str,
    user: User,
    agent: Agent,
    thread: Thread,
    user_messages: Union[UserMessage, List[UserMessage]],
    tools: Dict[str, Tool],
    force_reply: bool = True,
    model: Literal[tuple(models)] = "claude-3-5-sonnet-20241022",
):
    print("================================================")
    print(user_messages)
    print("================================================")

    print(agent)

    user_messages = (
        user_messages if isinstance(user_messages, List) else [user_messages]
    )
    user_message_id = user_messages[-1].id

    system_message = Template(template).render(
        name=agent.name,
        description=agent.description,
        instructions=agent.instructions,
        system_instructions=system_instructions,
    )

    pushes = {"messages": user_messages}

    agent_mentioned = any(
        re.search(rf"\b{re.escape(agent.name.lower())}\b", (msg.content or "").lower())
        for msg in user_messages
    )

    if agent_mentioned or force_reply:
        pushes["active"] = user_message_id
        thread.push(pushes)
    else:
        thread.push(pushes)
        return

    # think = True
    # if think:
    #     thought = await async_think(thread.messages, tools)
    #     if not speak, pop active

    yield ThreadUpdate(type=UpdateType.START_PROMPT)

    while True:
        try:
            # for error tracing
            sentry_sdk.add_breadcrumb(
                category="prompt",
                data={
                    "messages": thread.get_messages(),
                    "system_message": system_message,
                    "model": model,
                    "tools": tools.keys(),
                },
            )

            # main call to LLM
            content, tool_calls, stop = await async_prompt(
                thread.get_messages(),
                system_message=system_message,
                model=model,
                tools=tools,
            )

            # for error tracing
            sentry_sdk.add_breadcrumb(
                category="prompt",
                data={"content": content, "tool_calls": tool_calls, "stop": stop},
            )

            # create assistant message
            assistant_message = AssistantMessage(
                content=content or "",
                tool_calls=tool_calls,
                reply_to=user_messages[-1].id,
            )

            # push assistant message to thread and pop user message from actives array
            pushes = {"messages": assistant_message}
            pops = {"active": user_message_id} if stop else {}
            thread.push(pushes, pops)
            assistant_message = thread.messages[-1]

            # yield update
            yield ThreadUpdate(
                type=UpdateType.ASSISTANT_MESSAGE, message=assistant_message
            )

        except Exception as e:
            # capture error
            sentry_sdk.capture_exception(e)
            traceback.print_exc()

            # create assistant message
            assistant_message = AssistantMessage(
                content="I'm sorry, but something went wrong internally. Please try again later.",
                reply_to=user_messages[-1].id,
            )

            # push assistant message to thread and pop user message from actives array
            pushes = {"messages": assistant_message}
            pops = {"active": user_message_id}
            thread.push(pushes, pops)

            # yield update
            yield ThreadUpdate(
                type=UpdateType.ERROR, message=assistant_message, error=str(e)
            )

            # stop thread
            stop = True
            break

        # handle tool calls
        for t, tool_call in enumerate(assistant_message.tool_calls):
            try:
                # get tool
                tool = tools.get(tool_call.tool)
                if not tool:
                    raise Exception(f"Tool {tool_call.tool} not found.")

                # start task
                task = await tool.async_start_task(
                    user.id, agent.id, tool_call.args, db=db
                )

                # update tool call with task id and status
                thread.update_tool_call(
                    assistant_message.id,
                    t,
                    {"task": ObjectId(task.id), "status": "pending"},
                )

                # wait for task to complete
                result = await tool.async_wait(task)
                thread.update_tool_call(assistant_message.id, t, result)

                # yield update
                if result["status"] == "completed":
                    yield ThreadUpdate(
                        type=UpdateType.TOOL_COMPLETE,
                        tool_name=tool_call.tool,
                        tool_index=t,
                        result=result,
                    )
                else:
                    yield ThreadUpdate(
                        type=UpdateType.ERROR,
                        tool_name=tool_call.tool,
                        tool_index=t,
                        error=result.get("error"),
                    )

            except Exception as e:
                # capture error
                sentry_sdk.capture_exception(e)
                traceback.print_exc()

                # update tool call with status and error
                thread.update_tool_call(
                    assistant_message.id, t, {"status": "failed", "error": str(e)}
                )

                # yield update
                yield ThreadUpdate(
                    type=UpdateType.ERROR,
                    tool_name=tool_call.tool,
                    tool_index=t,
                    error=str(e),
                )

        if stop:
            print("Stopping prompt thread")
            yield ThreadUpdate(type=UpdateType.UPDATE_COMPLETE)
            break


def prompt_thread(
    db: str,
    user: User,
    agent: Agent,
    thread: Thread,
    user_messages: Union[UserMessage, List[UserMessage]],
    tools: Dict[str, Tool],
    force_reply: bool = False,
    model: Literal[tuple(models)] = "claude-3-5-sonnet-20241022",
):
    async_gen = async_prompt_thread(
        db, user, agent, thread, user_messages, tools, force_reply, model
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while True:
            try:
                yield loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


async def async_title_thread(thread: Thread, *extra_messages: UserMessage):
    """
    Generate a title for a thread
    """

    class TitleResponse(BaseModel):
        """A title for a thread of chat messages. It must entice a user to click on the thread when they are interested in the subject."""

        title: str = Field(
            description="a phrase of 2-5 words (or up to 30 characters) that conveys the subject of the chat thread. It should be concise and terse, and not include any special characters or punctuation."
        )

    system_message = "You are an expert at creating concise titles for chat threads."
    messages = thread.get_messages()
    messages.extend(extra_messages)
    messages.append(UserMessage(content="Come up with a title for this thread."))

    try:
        result = await async_prompt(
            messages,
            system_message=system_message,
            model="gpt-4o-mini",
            response_model=TitleResponse,
        )
        thread.title = result.title
        thread.save()

    except Exception as e:
        sentry_sdk.capture_exception(e)
        traceback.print_exc()
        return
