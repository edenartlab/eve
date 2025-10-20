import os
import json
import requests
import uuid
import modal

from eve.agent import Agent
from eve.user import User
from eve.tool import Tool
from eve.api.api import remote_prompt_session
from eve.api.handlers import setup_session
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
from eve import db


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")
    if not user:
        raise Exception("User is required")

    # note: if agent is provided in args, it is a subagent being called by the originating agent (which is just "agent" in the handler args)
    if args.get("agent"):
        agent = args.get("agent")
        agent = Agent.load(agent)
    else:
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
        
        # If we're going to prompt, let the remote function handle message addition
        if args.get("prompt"):

            # Run asynchronously
            if args.get("async"):
                app_name = f"api-{db.lower()}"
                remote_fn = modal.Function.from_name(app_name, "remote_prompt_session_fn")
                remote_fn.spawn(
                    session_id=session_id,
                    agent_id=str(agent.id),
                    user_id=str(user.id),
                    content=args.get("content"),
                    attachments=args.get("attachments") or [],
                    extra_tools=args.get("extra_tools") or [],
                )

            # Run and wait for the result
            else:
                result = await remote_prompt_session(
                    session_id=session_id,
                    agent_id=str(agent.id),
                    user_id=str(user.id),
                    content=args.get("content"),
                    attachments=args.get("attachments") or [],
                    extra_tools=args.get("extra_tools") or [],
                )
                session_id = result["session_id"]

        else:
            # Just add the message without prompting
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
    
    return {
        "output": [{"session": session_id}]
    }