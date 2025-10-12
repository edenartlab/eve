import os
import json
import requests
import uuid

from eve.agent import Agent
from eve.user import User
from eve.tool import Tool
from eve.api.handlers import setup_session
from eve.api.api import run_session_prompt
from eve.api.api_requests import (
    PromptSessionRequest,
    SessionCreationArgs
)
from eve.agent.session.models import (
    PromptSessionContext,
    ChatMessageRequestInput,
    LLMConfig,
    Session,
    ChatMessage
)
from eve.agent.session.session import (
    add_chat_message,
    build_llm_context,
    async_prompt_session
)


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")
    if not user:
        raise Exception("User is required")

    agent = Agent.from_mongo(agent)
    user = User.from_mongo(user)

    # note: session in handler args refers to the originating session (if there is one), not the session that is being posted to. session to post to is args.get("session")
    session_id = args.get("session")

    # create genesis session if new
    request = None
    if session_id is None:
        title = args.get("title")
        if not title:
            if session:
                title = f"Session spawned from {session}"
            else:
                title = f"New Session"

        request = PromptSessionRequest(
            user_id=str(user.id),
            creation_args=SessionCreationArgs(
                owner_id=str(user.id),
                agents=[str(agent.id)],
                title=title
            )
        )

        new_session = setup_session(
            None,
            request.session_id,
            request.user_id,
            request
        )

        session_id = str(new_session.id)

    # make a new set of drafts
    session = Session.from_mongo(session_id)

    # Use user.id as initiating_user_id if request is not available
    initiating_user_id = request.user_id if request else str(user.id)

    if args.get("role") == "assistant":

        new_message = ChatMessage(
            role="assistant",
            sender=agent.id,
            session=session.id,
            content=args.get("content"),
            attachments=args.get("attachments") or [],
        )

        context = PromptSessionContext(
            session=session,
            initiating_user_id=initiating_user_id,
            message=new_message,
            llm_config=LLMConfig(model="claude-sonnet-4-5-20250929")
        )

        new_message.save()

    elif args.get("role") in ["system", "user"]:
        new_message = ChatMessage(
            role=args.get("role"),
            sender=user.id,
            session=session.id,
            content=args.get("content"),
            attachments=args.get("attachments") or [],
        )

        context = PromptSessionContext(
            session=session,
            initiating_user_id=initiating_user_id,
            message=new_message,
            llm_config=LLMConfig(model="claude-sonnet-4-5-20250929"),
        )

        if args.get("extra_tools"):
            context.extra_tools = {
                k: Tool.load(k) for k in args.get("extra_tools")
            }

        await add_chat_message(session, context)

    if args.get("prompt") and args.get("role") in ["system", "user"]:
        try:
            # Spawn the session prompt in a remote Modal container
            run_session_prompt.spawn(
                session_id=session_id,
                agent_id=str(agent.id),
                extra_tools=args.get("extra_tools"),
            )
        except Exception as e:
            # Fallback for local testing without Modal
            print(f"Modal spawn failed ({e}), running session prompt inline for local testing")
            context = await build_llm_context(
                session,
                agent,
                context,
                trace_id=trace_id,
            )
            async for m in async_prompt_session(session, context, agent):
                pass
    
    return {
        "output": [{"session": session_id}]
    }