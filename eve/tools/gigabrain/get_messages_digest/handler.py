import json
from datetime import datetime
from typing import List

import pytz
from bson import ObjectId
from pydantic import BaseModel, Field

from eve.agent.agent import Agent
from eve.agent.llm.llm import async_prompt
from eve.agent.llm.prompts.system_template import system_template
from eve.agent.session.models import ChatMessage, LLMConfig, LLMContext
from eve.tool import ToolContext

# Import the get_messages functionality
from eve.tools.eden_utils.get_messages.handler import handler as get_messages_handler


class MessagesDigest(BaseModel):
    """Structured output for messages digest"""

    summary: str = Field(
        description="A comprehensive summary of the messages that are relevant to the user's instructions"
    )
    attachments: List[str] = Field(
        description="A curated list of up to 4 attachment URLs that are most relevant to the user's instructions",
        max_length=4,
    )


DIGEST_PROMPT_TEMPLATE = """You have been given a set of chat messages from various sessions and a set of attachments (media URLs) from those messages.

<Instructions>
{instructions}
</Instructions>

<Messages>
{messages}
</Messages>

Based on the user's instructions above, provide:
1. A comprehensive summary of anything in the messages that is relevant to the instructions. Be thorough and include all relevant details, quotes, and context.
2. A curated selection of up to 4 attachments (URLs) that are most relevant to the instructions. Only include attachments that directly relate to what the user is looking for. If no attachments are relevant, return an empty list.

IMPORTANT: When referencing or quoting specific messages in your summary, always cite the original message URL (shown in parentheses after the message content, e.g. https://discord.com/channels/... or https://x.com/... or https://farcaster.xyz/...). This allows readers to navigate directly to the source."""


async def handler(context: ToolContext):
    agent_id = context.agent
    session_ids = context.args.get("session_ids", [])
    hours = context.args.get("hours", 24)
    instructions = context.args.get("instructions", "")

    if not agent_id:
        return {"output": {"error": "agent is required"}}

    if not session_ids:
        return {"output": {"error": "session_ids is required"}}

    if not instructions:
        return {"output": {"error": "instructions is required"}}

    # Load the agent
    if isinstance(agent_id, str):
        agent_id = ObjectId(agent_id)
    agent = Agent.from_mongo(agent_id)
    if not agent:
        return {"output": {"error": f"Agent not found: {agent_id}"}}

    # Call get_messages to get the raw messages and attachments
    # Session filtering by agent membership is handled by get_messages via context.agent
    get_messages_context = ToolContext(
        args={"session_ids": session_ids, "hours": hours},
        agent=str(agent_id),
    )
    messages_result = await get_messages_handler(get_messages_context)
    messages_data = messages_result.get("output", {})

    messages_text = messages_data.get("messages", "")
    all_attachments = messages_data.get("attachments", [])

    if not messages_text or messages_text == "No valid session IDs provided.":
        return {
            "output": {
                "summary": "No messages found in the specified sessions and time window.",
                "attachments": [],
            }
        }

    # Format attachments as a numbered list for the LLM
    attachments_text = (
        "\n".join(f"{i + 1}. {url}" for i, url in enumerate(all_attachments))
        if all_attachments
        else "(No attachments available)"
    )

    # Build the prompt
    user_prompt = DIGEST_PROMPT_TEMPLATE.format(
        instructions=instructions,
        messages=messages_text,
        attachments=attachments_text,
    )

    # Build system message using the agent's persona
    system_message = system_template.render(
        name=agent.name,
        current_date=datetime.now(pytz.utc).strftime("%Y-%m-%d"),
        persona=agent.persona,
        tools=None,
    )

    # Build LLM context with structured output
    llm_context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=user_prompt),
        ],
        config=LLMConfig(model="gpt-4o-mini", response_format=MessagesDigest),
    )

    # Get structured response from LLM
    response = await async_prompt(llm_context)

    # Parse the structured output
    digest = MessagesDigest(**json.loads(response.content))

    # Validate that returned attachments are from the available set
    valid_attachments = [url for url in digest.attachments if url in all_attachments]

    return {
        "output": {
            "summary": digest.summary,
            "attachments": valid_attachments[:4],  # Ensure max 4
        }
    }
