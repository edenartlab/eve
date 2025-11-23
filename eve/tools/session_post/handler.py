import json
from typing import List, Optional

import modal
from bson import ObjectId
from fastapi import BackgroundTasks
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
from eve.user import User


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

    if context.args.get("role") == "assistant":
        new_message = ChatMessage(
            role="assistant",
            sender=agent.id,
            session=session.id,
            content=context.args.get("content"),
            attachments=context.args.get("attachments") or [],
        )
        prompt_context = PromptSessionContext(
            session=session,
            initiating_user_id=str(user.id),
            message=new_message,
            llm_config=LLMConfig(model="claude-sonnet-4-5-20250929"),
        )
        new_message.save()

        # stats = agent.stats
        # stats["messageCount"] += 1
        # agent.update(stats=stats.model_dump())

    elif context.args.get("role") in ["system", "user"]:
        # If we're going to prompt, run session prompt routine (it handles message addition)
        if context.args.get("prompt"):
            result = await run_session_prompt(
                session_id=session_id,
                agent_id=str(agent.id),
                user_id=str(user.id),
                content=context.args.get("content"),
                attachments=context.args.get("attachments") or [],
                extra_tools=context.args.get("extra_tools") or [],
                async_mode=context.args.get("async"),
            )
            return result

        else:
            # Otherwise, just add the message manually, but don't prompt
            new_message = ChatMessage(
                role=context.args.get("role"),
                sender=user.id,
                session=session.id,
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
        )

        return {"output": {"session_id": session_id}}

    # Otherwise, run session prompt and return to parent session
    await remote_prompt_session(
        session_id=session_id,
        agent_id=agent_id,
        user_id=user_id,
        content=content,
        attachments=attachments,
        extra_tools=extra_tools,
    )

    # Get structured output from the remote session prompt
    class RemoteSessionResponse(BaseModel):
        """All relevant results (or error report) from the remote session prompt"""

        outputs: List[str] = Field(
            description="A list of all requested successful media outputs, given the original request. Do not include intermediate results here -- only the desired output given the task."
        )
        intermediate_outputs: Optional[List[str]] = Field(
            description="An optional list of all important **intermediate media urls** that were generated during the session, leading up to the final result. This does not include the original attachments provided by the user, or the final output."
        )
        error: Optional[str] = Field(
            description="A human-readable error message that explains why the session failed to produce the requested outputs. Mutually exclusive with outputs -- **ONLY** set this if there was an error or the requested output was not successfully generated."
        )

    system_message = """
    You are a helpful assistant that summarizes the results of a remote session prompt.

    Given the original request and the subsequent response to it, identify if the session was successful or not.
    If it was successful, list all the successful outputs. If there were any important intermediate results, list them in the intermediate_outputs field.
    If it was not successful, provide a human-readable error message that explains why the session failed to produce the requested outputs.
    """

    messages = ChatMessage.find({"session": ObjectId(session_id)})

    # Build LLM context with custom tools
    context = LLMContext(
        messages=[
            ChatMessage(role="system", content=system_message),
            *messages,
            ChatMessage(role="user", content="Summarize the results of the session."),
        ],
        config=LLMConfig(
            model="claude-sonnet-4-5",
            # model="gpt-4o-mini",
            response_format=RemoteSessionResponse,
        ),
    )

    response = await async_prompt(context)

    output = RemoteSessionResponse(**json.loads(response.content))

    # if output.error:
    #     raise Exception(output.error)

    if output.error:
        result = {
            "output": [],
            "intermediate_outputs": output.intermediate_outputs,
            "error": output.error,
            "session_id": session_id,
        }
    else:
        result = {
            "output": output.outputs,
            "intermediate_outputs": output.intermediate_outputs,
            "session_id": session_id,
        }

    return result
