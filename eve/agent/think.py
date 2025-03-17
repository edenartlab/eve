import asyncio
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import Literal

from ..eden_utils import dump_json
from .llm import async_prompt
from .agent import Agent, refresh_agent
from .thread import UserMessage, Thread


knowledge_think_template = Template("""<Knowledge>
The following summarizes your background knowledge and the circumstances for which you may need to consult or refer to it. If you need to consult your knowledge base, set "recall_knowledge" to true.

{{ knowledge_description }}
</Knowledge>""")


thought_template = Template("""<Name>{{ name }}</Name>
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


async def async_think(
    agent: Agent,
    thread: Thread,
    user_message: UserMessage,
    force_reply: bool = True,
):

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
        recall_knowledge: bool = Field(
            ...,
            description="Whether to look up relevant knowledge from the knowledge base.",
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

    prompt = thought_template.render(
        name=agent.name,
        message_count=len(messages),
        chat=chat,
        knowledge_description=knowledge_description,
        message=message,
        reply_criteria=reply_criteria,
    )

    thought = await async_prompt(
        [UserMessage(content=prompt)],
        system_message=f"You analyze the chat on behalf of {agent.name} and generate a thought.",
        # model="gpt-4o-mini",
        model="claude-3-5-haiku-latest",
        response_model=ChatThought,
    )

    if force_reply:
        thought.intention = "reply"

    return thought


def think(
    agent: Agent, 
    thread: Thread, 
    user_message: UserMessage, 
    force_reply: bool = True
):
    return asyncio.run(
        async_think(agent, thread, user_message, force_reply)
    )