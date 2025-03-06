import asyncio
from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional

from .agent import Agent, refresh_agent
from .thread import UserMessage, Thread
from ..tool import TOOL_CATEGORIES
from ..eden_utils import dump_json, load_template
from .llm import async_prompt

knowledge_think_template = load_template("knowledge_think")
thought_template = load_template("thought")
tools_template = load_template("tools")


async def async_think(
    agent: Agent,
    thread: Thread,
    user_message: UserMessage,
    force_reply: bool = True,
    metadata: Optional[Dict] = None,
):
    # intention_description = "Response class to the last user message. Ignore if irrelevant, reply if relevant and you intend to say something."

    # if agent.reply_criteria:
    #     intention_description += (
    #         f"\nAdditional criteria for replying spontaneously: {agent.reply_criteria}"
    #     )

    class ChatThought(BaseModel):
        """A response to a chat message."""

        intention: Literal["ignore", "reply"] = Field(
            ...,
            description="Ignore if last message is irrelevant, reply if relevant or criteria met.",
        )
        thought: str = Field(
            ...,
            description="A very brief thought about what relevance, if any, the last user message has to you, and a justification of your intention.",
        )
        tools: Optional[Literal[tuple(TOOL_CATEGORIES.keys())]] = Field(
            ...,
            description=f"Which tools to include in reply context",
        )
        recall_knowledge: bool = Field(
            ...,
        )

    # generate text blob of chat history
    chat = ""
    messages = thread.get_messages(25)
    for msg in messages:
        content = msg.content
        if msg.role == "user":
            if msg.attachments:
                content += f" (attachments: {msg.attachments})"
            name = "You" if msg.name == agent.name else msg.name or "User"
        elif msg.role == "assistant":
            name = agent.name
            for tc in msg.tool_calls:
                args = ", ".join([f"{k}={v}" for k, v in tc.args.items()])
                tc_result = dump_json(tc.result, exclude="blurhash")
                content += f"\n -> {tc.tool}({args}) -> {tc_result}"
        time_str = msg.createdAt.strftime("%H:%M")
        chat += f"<{name} {time_str}> {content}\n"

    # user message text
    content = user_message.content
    if user_message.attachments:
        content += f" (attachments: {user_message.attachments})"
    time_str = user_message.createdAt.strftime("%H:%M")
    message = f"<{user_message.name} {time_str}> {content}"

    if agent.knowledge:
        # if knowledge is requested but no knowledge description, create it now
        if not agent.knowledge_description:
            await refresh_agent(agent)
            agent.reload()

        knowledge_description = f"Summary: {agent.knowledge_description.summary}. Recall if: {agent.knowledge_description.retrieval_criteria}"
        knowledge_description = knowledge_think_template.render(
            knowledge_description=knowledge_description
        )
    else:
        knowledge_description = ""

    if agent.reply_criteria:
        reply_criteria = f"Note: You should additionally set reply to true if any of the follorwing criteria are met: {agent.reply_criteria}"
    else:
        reply_criteria = ""

    tool_descriptions = "\n".join([f"{k}: {v}" for k, v in TOOL_CATEGORIES.items()])
    tools_description = tools_template.render(tool_categories=tool_descriptions)

    prompt = thought_template.render(
        name=agent.name,
        chat=chat,
        tools_description=tools_description,
        knowledge_description=knowledge_description,
        message=message,
        reply_criteria=reply_criteria,
    )

    thought = await async_prompt(
        [UserMessage(content=prompt)],
        system_message=f"You analyze the chat on behalf of {agent.name} and generate a thought.",
        model="gpt-4o-mini",
        response_model=ChatThought,
    )

    if force_reply:
        thought.intention = "reply"

    return thought


def think(
    agent: Agent, thread: Thread, user_message: UserMessage, force_reply: bool = True
):
    return asyncio.run(async_think(agent, thread, user_message, force_reply))
