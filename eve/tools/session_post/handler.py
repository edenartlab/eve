import os
import json
import requests
import uuid

from eve.agent import Agent
from eve.user import User
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
            initiating_user_id=request.user_id,
            message=new_message,
            llm_config=LLMConfig(model="claude-sonnet-4-5")
        )

        new_message.save()

    elif args.get("role") in ["system", "user"]:

        from eve.tool import Tool
        custom_tool = Tool.load("farcaster_cast")
        
        new_message = ChatMessage(
            role=args.get("role"),
            sender=user.id,
            session=session.id,
            content=args.get("content"),
            attachments=args.get("attachments") or [],
        )

        context = PromptSessionContext(
            session=session,
            initiating_user_id=request.user_id,
            message=new_message,
            llm_config=LLMConfig(model="claude-sonnet-4-5"),
            extra_tools={custom_tool.key: custom_tool},
        )

        if args.get("extra_tools"):
            context.extra_tools = {
                k: Tool.load(k) for k in args.get("extra_tools")
            }

        await add_chat_message(session, context)
    
    if args.get("prompt") and args.get("role") in ["system", "user"]:
        context = await build_llm_context(
            session, 
            agent, 
            context, 
            trace_id=str(uuid.uuid4()), 
        )
        
        # set this up in a remote container
        async for m in async_prompt_session(session, context, agent):
            pass
    
    return {
        "session": session_id
    }

