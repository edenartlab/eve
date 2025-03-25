import re
import os
import asyncio
import traceback
import functools
from bson import ObjectId
from typing import Optional, Dict, Any, List, Union, Literal
from pydantic import BaseModel
from pydantic.config import ConfigDict
from sentry_sdk import trace, start_transaction, add_breadcrumb, capture_exception

from ..eden_utils import load_template
from ..mongo import get_collection
from ..models import Model
from ..tool import Tool
from ..user import User
from .agent import Agent
from .thread import UserMessage, AssistantMessage, ToolCall, Thread
from .llm import async_prompt, async_prompt_stream, UpdateType, MODELS, DEFAULT_MODEL
from .think import async_think


system_template = load_template("system")
knowledge_reply_template = load_template("knowledge_reply")
models_instructions_template = load_template("models_instructions")
model_template = load_template("model_doc")

USE_THINKING = False  # os.getenv("USE_THINKING", "false").lower() == "true"

# _models_cache: Dict[str, Dict[str, Model]] = {} # todo


# todo: `msg.error` not `msg.message.error`
class ThreadUpdate(BaseModel):
    type: UpdateType
    message: Optional[AssistantMessage] = None
    tool_name: Optional[str] = None
    tool_index: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    text: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


def sentry_transaction(op: str, name: str):
    def decorator(func):
        @trace
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            transaction = start_transaction(op=op, name=name)
            try:
                async for item in func(*args, **kwargs):
                    yield item
            finally:
                transaction.finish()

        return wrapper

    return decorator


async def process_tool_call(
    thread: Thread,
    assistant_message: AssistantMessage,
    tool_call_index: int,
    tool_call: ToolCall,
    tools: Dict[str, Tool],
    user_id: str,
    agent_id: str,
    is_client_platform: bool = False,
) -> ThreadUpdate:
    """Process a single tool call and return the appropriate ThreadUpdate"""

    try:
        # Get tool
        tool = tools.get(tool_call.tool)
        print("TOOL IS !!", tool)
        if not tool:
            raise Exception(f"Tool {tool_call.tool} not found.")

        # Start task
        task = await tool.async_start_task(
            user_id, agent_id, tool_call.args, is_client_platform
        )

        # Update tool call with task id and status
        thread.update_tool_call(
            assistant_message.id,
            tool_call_index,
            {"task": ObjectId(task.id), "status": "pending"},
        )

        # Wait for task to complete
        print("TYOPE OF TOIOL", type(tool))
        result = await tool.async_wait(task)

        print("RESULT OF THE TASK IS 555", result)
        result = {"status": "completed", "output": result}

        thread.update_tool_call(
            assistant_message.id, 
            tool_call_index, 
            result
        )

        # Task completed
        if result["status"] == "completed":
            # Yield update
            return ThreadUpdate(
                type=UpdateType.TOOL_COMPLETE,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                result=result,
            )
        else:
            # Yield error
            return ThreadUpdate(
                type=UpdateType.ERROR,
                tool_name=tool_call.tool,
                tool_index=tool_call_index,
                error=result.get("error"),
            )

    except Exception as e:
        # Capture error
        capture_exception(e)
        traceback.print_exc()

        # Update tool call with status and error
        thread.update_tool_call(
            assistant_message.id,
            tool_call_index,
            {"status": "failed", "error": str(e)},
        )

        # Yield update
        return ThreadUpdate(
            type=UpdateType.ERROR,
            tool_name=tool_call.tool,
            tool_index=tool_call_index,
            error=str(e),
        )


@sentry_transaction(op="llm.prompt", name="async_prompt_thread")
async def async_prompt_thread(
    user: User,
    agent: Agent,
    thread: Thread,
    user_messages: Union[UserMessage, List[UserMessage]],
    tools: Dict[str, Tool],
    force_reply: bool = False,
    use_thinking: bool = True,
    model: Literal[tuple(MODELS)] = DEFAULT_MODEL,
    user_is_bot: bool = False,
    stream: bool = False,
    is_client_platform: bool = False,
):
    model = model or DEFAULT_MODEL
    user_messages = (
        user_messages if isinstance(user_messages, list) else [user_messages]
    )
    user_message_id = user_messages[-1].id

    # Apply bot-specific limits
    if user_is_bot:
        print("Bot message, stopping")
        return

    # Thinking step
    use_thinking = False
    if use_thinking:
        print("Thinking...")

        # A thought contains intention and tool pre-selection
        thought = await async_think(
            agent=agent,
            thread=thread,
            user_message=user_messages[-1],
            force_reply=force_reply,
        )
        thought = thought.model_dump()

    else:
        print("Skipping thinking, default to classic behavior")

        # Check mentions
        agent_mentioned = any(
            re.search(
                rf"\b{re.escape(agent.name.lower())}\b", (msg.content or "").lower()
            )
            for msg in user_messages
        )
        print("Agent mentioned", agent_mentioned)

        # When there's no thinking, reply if mentioned or forced, and include all tools
        thought = {
            "thought": "none",
            "intention": "reply" if agent_mentioned or force_reply else "ignore",
            "recall_knowledge": False,
        }

    # for error tracing
    add_breadcrumb(
        category="prompt_thought",
        data={
            "user_message": user_messages[-1],
            "model": model,
            "thought": thought,
        },
    )

    # reply only if intention is "reply"
    should_reply = thought["intention"] == "reply"

    if should_reply:
        # Update thread and continue
        thread.push({"messages": user_messages, "active": user_message_id})
    else:
        # Update thread and stop
        thread.push({"messages": user_messages})
        return

    # Get text describing models
    if agent.models or agent.model:
        models_collection = get_collection(Model.collection_name)
        models = agent.models or [
            {"lora": agent.model, "use_when": "This is the default Lora model"}
        ]
        models = {m["lora"]: m for m in models}
        model_docs = models_collection.find(
            {"_id": {"$in": list(models.keys())}, "deleted": {"$ne": True}}
        )
        model_docs = list(model_docs or [])
        for doc in model_docs:
            doc["use_when"] = (
                f'\n<use_when>{models[ObjectId(doc["_id"])].get("use_when", "This is the default Lora model")}</use_when>'
            )
        models_list = "\n".join(model_template.render(doc) for doc in model_docs)
        models_instructions = models_instructions_template.render(models=models_list)
    else:
        models_instructions = ""

    # Yield start signal
    yield ThreadUpdate(type=UpdateType.START_PROMPT)

    while True:
        try:
            messages = thread.get_messages(25)

            # If knowledge requested, prepend with full knowledge text
            if thought.get("recall_knowledge") and agent.knowledge:
                knowledge = knowledge_reply_template.render(knowledge=agent.knowledge)
            else:
                knowledge = ""

            system_message = system_template.render(
                name=agent.name,
                persona=agent.persona,
                knowledge=knowledge,
                models_instructions=models_instructions,
            )

            # For error tracing
            add_breadcrumb(
                category="prompt_in",
                data={
                    "messages": messages,
                    "model": model,
                    "tools": (tools or {}).keys(),
                },
            )

            # Main call to LLM, streaming
            if stream:
                content_chunks = []
                tool_calls = []
                stop = True

                async for update_type, content in async_prompt_stream(
                    messages,
                    system_message=system_message,
                    model=model,
                    tools=tools,
                ):
                    # Stream an individual token
                    if update_type == UpdateType.ASSISTANT_TOKEN:
                        if not content:  # Skip empty content
                            continue
                        content_chunks.append(content)
                        yield ThreadUpdate(
                            type=UpdateType.ASSISTANT_TOKEN, text=content
                        )

                    # Tool call
                    elif update_type == UpdateType.TOOL_CALL:
                        tool_calls.append(content)

                    # Detect stop call
                    elif update_type == UpdateType.ASSISTANT_STOP:
                        stop = content == "end_turn" or content == "stop"

                # Create assistant message from accumulated content
                content = "".join(content_chunks)

            # Main call to LLM, non-streaming
            else:
                content, tool_calls, stop = await async_prompt(
                    messages,
                    system_message=system_message,
                    model=model,
                    tools=tools,
                )

            # For error tracing
            add_breadcrumb(
                category="prompt_out",
                data={"content": content, "tool_calls": tool_calls, "stop": stop},
            )

            # Create assistant message
            assistant_message = AssistantMessage(
                content=content or "",
                tool_calls=tool_calls,
                reply_to=user_messages[-1].id,
            )

            # Todo: Save thought to just first assistant message
            # if use_thinking:
            #     assistant_message.thought = copy.deepcopy(thought)
            #     thought = None

            # Push assistant message to thread and pop user message from actives array
            pushes = {"messages": assistant_message}
            pops = {"active": user_message_id} if stop else {}
            thread.push(pushes, pops)
            assistant_message = thread.messages[-1]

            # Yield update
            if not agent.mute:
                yield ThreadUpdate(
                    type=UpdateType.ASSISTANT_MESSAGE, message=assistant_message
                )

        except Exception as e:
            # Capture error
            capture_exception(e)
            traceback.print_exc()

            # Create assistant message
            assistant_message = AssistantMessage(
                content="I'm sorry, but something went wrong internally. Please try again later.",
                reply_to=user_messages[-1].id,
            )

            # Push assistant message to thread and pop user message from actives array
            pushes = {"messages": assistant_message}
            pops = {"active": user_message_id}
            thread.push(pushes, pops)

            # Yield error message
            yield ThreadUpdate(
                type=UpdateType.ERROR, message=assistant_message, error=str(e)
            )

            # Stop thread
            stop = True
            break

        # Handle tool calls in batches of 4
        tool_calls = assistant_message.tool_calls or []
        for b in range(0, len(tool_calls), 4):
            batch = enumerate(tool_calls[b : b + 4])
            tasks = [
                process_tool_call(
                    thread,
                    assistant_message,
                    b + idx,
                    tool_call,
                    tools,
                    user.id,
                    agent.id,
                    is_client_platform,
                )
                for idx, tool_call in batch
            ]

            # Wait for batch to complete and yield each result
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for result in results:
                yield result

        # If stop called, break out of loop
        if stop:
            break

    # Yield end signal
    yield ThreadUpdate(type=UpdateType.END_PROMPT)


def prompt_thread(
    user: User,
    agent: Agent,
    thread: Thread,
    user_messages: Union[UserMessage, List[UserMessage]],
    tools: Dict[str, Tool],
    force_reply: bool = False,
    use_thinking: bool = USE_THINKING,
    model: Literal[tuple(MODELS)] = DEFAULT_MODEL,
    user_is_bot: bool = False,
    is_client_platform: bool = False,
):
    async_gen = async_prompt_thread(
        user,
        agent,
        thread,
        user_messages,
        tools,
        force_reply,
        use_thinking,
        model,
        user_is_bot,
        is_client_platform=is_client_platform,
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
