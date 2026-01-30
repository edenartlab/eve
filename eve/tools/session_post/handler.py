import asyncio
import json
from enum import Enum
from typing import List, Optional

import modal
from bson import ObjectId
from fastapi import BackgroundTasks
from loguru import logger
from pydantic import BaseModel, Field

from eve import db
from eve.agent import Agent
from eve.agent.llm.llm import async_prompt
from eve.agent.session.context import add_chat_message
from eve.agent.session.models import (
    ChatMessage,
    ChatMessageRequestInput,
    LLMConfig,
    LLMContext,
    PromptSessionContext,
    Session,
)
from eve.agent.session.service import create_prompt_session_handle
from eve.api.api import remote_prompt_session
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.tool import Tool, ToolContext
from eve.user import User, increment_message_count


class ResponseType(str, Enum):
    """Types of structured output responses for session_post"""

    MEDIA = "media"  # Default: looks for media outputs (URLs)
    SEED = "seed"  # Looks for seed IDs from verdelis_seed or similar tools


# Response models for different output types


class MediaSessionResponse(BaseModel):
    """Response for media-producing sessions (default)"""

    outputs: List[str] = Field(
        description="A list of all requested successful media outputs, given the original request. Do not include intermediate results here -- only the desired output given the task. IMPORTANT: Always extract and return whatever outputs were successfully generated, even if the count doesn't perfectly match what was requested. Prefer returning partial results over returning nothing."
    )
    intermediate_outputs: Optional[List[str]] = Field(
        description="An optional list of all important **intermediate media urls** that were generated during the session, leading up to the final result. This does not include the original attachments provided by the user, or the final output."
    )
    error: Optional[str] = Field(
        description="A human-readable error message explaining any issues encountered. Can be set alongside outputs if there were partial failures. Only set this if there was a genuine error or significant issue - minor discrepancies in counts should not be treated as errors."
    )


class SeedSessionResponse(BaseModel):
    """Response for artifact-creating sessions (verdelis_seed, verdelis_storyboard, etc.)"""

    artifact_ids: List[str] = Field(
        description="A list of all artifact IDs that were successfully created during this session. Look for tool calls to verdelis_seed, verdelis_storyboard, or similar artifact-creating tools and extract the artifact_id from their results."
    )
    error: Optional[str] = Field(
        description="A human-readable error message that explains why the session failed to create any artifacts. Mutually exclusive with artifact_ids -- **ONLY** set this if there was an error or no artifacts were successfully created."
    )


# System prompts for each response type

MEDIA_SYSTEM_PROMPT = """
You are a helpful assistant that extracts results from a remote session prompt.

Given the original request and the subsequent response, extract all successfully generated outputs.

CRITICAL RULES:
1. **Always return outputs if they exist** - If media was successfully generated, include it in outputs even if the session had issues or the count doesn't match what was requested.
2. **Coerce to expected format** - If the session generated more items than expected, select the most relevant ones. If it generated fewer, return what exists.
3. **Prefer partial success over failure** - Only set error WITHOUT outputs if genuinely nothing was produced. If some outputs exist alongside problems, return both the outputs AND an error note.
4. **Extract structured data** - Look for JSON blocks (like STORYBOARD_TRANSCRIPT, VIDEO_TRANSCRIPT) and URLs in the session. These are the outputs.
5. **Don't be pedantic about counts** - If 24 images were generated instead of 20, that's still a success. Extract what's there.

Examples:
- Session generated 24 keyframes instead of 20 requested → Return all 24 (or the final set) as outputs, no error
- Session generated images but forgot the final JSON block → Return the image URLs as outputs, optionally note the missing JSON
- Session had tool errors but eventually produced results → Return the results, optionally note there were retries
- Session produced nothing at all → Return empty outputs with error explaining why
"""

SEED_SYSTEM_PROMPT = """
You are a helpful assistant that extracts artifact IDs from a session.

Look through the session messages for any tool calls to artifact-creating tools (like verdelis_seed or verdelis_storyboard).
For each successful artifact creation, extract the artifact_id from the tool result.

If artifacts were successfully created, list all their IDs in the artifact_ids field.
If no artifacts were created or there was an error, provide a human-readable error message explaining what went wrong.
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    if not context.user:
        raise Exception("User is required")

    # note: if agent is provided in context.args, it is a subagent being called by the originating agent (which is just "agent" in the handler context.args)
    if context.args.get("agent"):
        agent = context.args.get("agent")
        agent = Agent.load(agent)
    else:
        agent = Agent.from_mongo(context.agent)

    user = User.from_mongo(context.user)

    # Note: session in handler args refers to the parent session (if there is one), **not** the session that is being posted to. The session to post to is context.args.get("session")
    session_id = context.args.get("session")

    # if no session is provided, create a new session
    if session_id is None:
        title = context.args.get("title")
        if not title:
            title = (
                f"Session spawned from {context.session}"
                if context.session
                else "New Session"
            )
        request = PromptSessionRequest(
            user_id=str(user.id),
            creation_args=SessionCreationArgs(
                owner_id=str(user.id),
                agents=[str(agent.id)],
                title=title,
                parent_session=context.session,
                extras={"is_public": context.args.get("public", False)},
                visible=context.args.get("visible"),
            ),
        )
        placeholder_request = request.model_copy()
        placeholder_request.message = ChatMessageRequestInput(
            role="system", content="Initializing session"
        )
        handle = create_prompt_session_handle(placeholder_request, BackgroundTasks())
        new_session = handle.session
        session_id = str(new_session.id)

    if context.message and context.tool_call_id:
        message = ChatMessage.from_mongo(context.message)
        tool_call = [tc for tc in message.tool_calls if tc.id == context.tool_call_id]
        if tool_call:
            tool_call[0].child_session = ObjectId(session_id)
            message.save()

    # make a new set of drafts
    session = Session.from_mongo(session_id)

    # Update visible flag if explicitly provided
    if context.args.get("visible") is not None:
        session.visible = context.args.get("visible")
        session.save()

    if context.args.get("role") == "assistant":
        attachments = context.args.get("attachments") or []
        if attachments:
            from eve.s3 import upload_attachments_to_eden

            attachments = await upload_attachments_to_eden(attachments)

        new_message = ChatMessage(
            role="assistant",
            sender=agent.id,
            session=[session.id],
            content=context.args.get("content"),
            attachments=attachments,
        )
        prompt_context = PromptSessionContext(
            session=session,
            initiating_user_id=str(user.id),
            message=new_message,
            llm_config=LLMConfig(model="claude-sonnet-4-5-20250929"),
        )
        new_message.save()

        # Increment message count for the agent (sender)
        increment_message_count(agent.id)

    elif context.args.get("role") in ["system", "user"]:
        # If we're going to prompt, run session prompt routine (it handles message addition)
        if context.args.get("prompt"):
            # Get response type (default to media)
            response_type_str = context.args.get("response_type", "media")
            response_type = ResponseType(response_type_str)

            result = await run_session_prompt(
                session_id=session_id,
                agent_id=str(agent.id),
                user_id=str(user.id),
                content=context.args.get("content"),
                attachments=context.args.get("attachments") or [],
                extra_tools=context.args.get("extra_tools") or [],
                async_mode=context.args.get("async"),
                response_type=response_type,
                selection_limit=context.args.get("selection_limit"),
            )
            return result

        else:
            new_message = ChatMessageRequestInput(
                role=context.args.get("role"),
                content=context.args.get("content"),
                attachments=context.args.get("attachments") or [],
            )

            prompt_context = PromptSessionContext(
                session=session,
                initiating_user_id=str(user.id),
                message=new_message,
                llm_config=LLMConfig(model="claude-sonnet-4-5-20250929"),
            )

            if context.args.get("extra_tools"):
                prompt_context.extra_tools = {
                    k: Tool.load(k) for k in context.args.get("extra_tools")
                }

            await add_chat_message(session, prompt_context)

    return {"output": [{"session": session_id}]}


async def run_session_prompt(
    session_id: str,
    agent_id: str,
    user_id: str,
    content: str,
    attachments: Optional[List[str]] = [],
    extra_tools: Optional[List[str]] = [],
    async_mode: bool = False,
    response_type: ResponseType = ResponseType.MEDIA,
    selection_limit: Optional[int] = None,
):
    # If async_mode, spawn session prompt and return immediately
    if async_mode:
        remote_fn = modal.Function.from_name(
            f"api-{db.lower()}", "remote_prompt_session_fn"
        )
        remote_fn.spawn(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            content=content,
            attachments=attachments,
            extra_tools=extra_tools,
            selection_limit=selection_limit,
        )

        return {"output": {"session_id": session_id}}

    # Otherwise, run session prompt and return to parent session
    try:
        await remote_prompt_session(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            content=content,
            attachments=attachments,
            extra_tools=extra_tools,
            selection_limit=selection_limit,
        )
    except asyncio.CancelledError:
        # Session was cancelled during execution
        return {
            "output": [],
            "status": "cancelled",
            "error": "The tool call was cancelled by the user",
            "session_id": session_id,
        }

    # Check if session was cancelled (indicated by "Response cancelled by user" message)
    last_messages = ChatMessage.find(
        {"session": ObjectId(session_id)},
        sort="createdAt",
        desc=True,
        limit=1,
    )
    if last_messages and last_messages[0].content == "Response cancelled by user":
        return {
            "output": [],
            "status": "cancelled",
            "error": "The tool call was cancelled by the user",
            "session_id": session_id,
        }

    # Get structured output based on response type
    if response_type == ResponseType.SEED:
        return await _extract_seed_response(session_id)
    else:
        return await _extract_media_response(session_id)


async def _extract_media_response(session_id: str) -> dict:
    """Extract media outputs from session (default behavior)"""
    messages = ChatMessage.find({"session": ObjectId(session_id)})

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=MEDIA_SYSTEM_PROMPT),
            *messages,
            ChatMessage(
                role="user",
                content="Extract all successfully generated outputs from the session. Remember: always return outputs if any media was generated, even if there were issues or count mismatches.",
            ),
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5",
            response_format=MediaSessionResponse,
        ),
    )

    response = await async_prompt(context)
    output = MediaSessionResponse(**json.loads(response.content))

    logger.info(f"========== Media response for session {session_id}: {output}")

    # Always return outputs if they exist, even alongside errors (partial success)
    result = {
        "output": output.outputs if output.outputs else [],
        "intermediate_outputs": output.intermediate_outputs,
        "session_id": session_id,
    }

    # Only include error in result if there's an error AND no outputs
    # (If there are outputs, the error is just informational and shouldn't block)
    if output.error and not output.outputs:
        result["error"] = output.error

    return result


async def _extract_seed_response(session_id: str) -> dict:
    """Extract artifact IDs from session"""
    messages = ChatMessage.find({"session": ObjectId(session_id)})

    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=SEED_SYSTEM_PROMPT),
            *messages,
            ChatMessage(
                role="user",
                content="Extract all artifact IDs that were created in this session.",
            ),
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5",
            response_format=SeedSessionResponse,
        ),
    )

    response = await async_prompt(context)

    output = SeedSessionResponse(**json.loads(response.content))

    if output.error:
        return {
            "output": [],
            "artifact_ids": [],
            "error": output.error,
            "session_id": session_id,
        }
    else:
        return {
            "output": output.artifact_ids,
            "artifact_ids": output.artifact_ids,
            "session_id": session_id,
        }
