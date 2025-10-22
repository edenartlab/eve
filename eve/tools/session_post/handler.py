import modal

from eve.agent import Agent
from eve.user import User
from eve.tool import Tool, ToolContext
from eve.api.api import remote_prompt_session
from eve.api.handlers import setup_session
from eve.api.api_requests import PromptSessionRequest, SessionCreationArgs
from eve.agent.session.models import (
    PromptSessionContext,
    LLMConfig,
    Session,
    ChatMessage,
)
from eve.agent.session.session import add_chat_message
from eve import db


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

    # note: session in handler args refers to the originating session (if there is one), not the session that is being posted to. session to post to is context.args.get("session")
    session_id = context.args.get("session")

    # create genesis session if new
    request = None
    if session_id is None:
        title = context.args.get("title")
        if not title:
            if session:
                title = f"Session spawned from {session}"
            else:
                title = f"New Session"

        request = PromptSessionRequest(
            user_id=str(user.id),
            creation_args=SessionCreationArgs(
                owner_id=str(user.id), agents=[str(agent.id)], title=title
            ),
        )

        new_session = setup_session(None, request.session_id, request.user_id, request)

        session_id = str(new_session.id)

        # Update parent tool call with child session ID (if called via tool execution)
        from eve.agent.session.tool_context import get_current_tool_call

        context = get_current_tool_call()
        if context:
            tool_call, assistant_message, tool_call_index = context
            tool_call.child_session = new_session.id
            if assistant_message.tool_calls and tool_call_index < len(
                assistant_message.tool_calls
            ):
                assistant_message.tool_calls[
                    tool_call_index
                ].child_session = new_session.id
                assistant_message.save()

    # make a new set of drafts
    session = Session.from_mongo(session_id)

    # Use user.id as initiating_user_id if request is not available
    initiating_user_id = request.user_id if request else str(user.id)

    if context.args.get("role") == "assistant":
        new_message = ChatMessage(
            role="assistant",
            sender=agent.id,
            session=session.id,
            content=context.args.get("content"),
            attachments=context.args.get("attachments") or [],
        )

        context = PromptSessionContext(
            session=session,
            initiating_user_id=initiating_user_id,
            message=new_message,
            llm_config=LLMConfig(model="claude-sonnet-4-5-20250929"),
        )

        new_message.save()

    elif context.args.get("role") in ["system", "user"]:
        # If we're going to prompt, let the remote function handle message addition
        if context.args.get("prompt"):
            # Run asynchronously
            if context.args.get("async"):
                app_name = f"api-{db.lower()}"
                remote_fn = modal.Function.from_name(
                    app_name, "remote_prompt_session_fn"
                )
                remote_fn.spawn(
                    session_id=session_id,
                    agent_id=str(agent.id),
                    user_id=str(user.id),
                    content=context.args.get("content"),
                    attachments=context.args.get("attachments") or [],
                    extra_tools=context.args.get("extra_tools") or [],
                )

            # Run and wait for the result
            else:
                result = await remote_prompt_session(
                    session_id=session_id,
                    agent_id=str(agent.id),
                    user_id=str(user.id),
                    content=context.args.get("content"),
                    attachments=context.args.get("attachments") or [],
                    extra_tools=context.args.get("extra_tools") or [],
                )

                # session_id = result["session_id"]

                return {"output": result}
                # return result

        else:
            # Just add the message without prompting
            new_message = ChatMessage(
                role=context.args.get("role"),
                sender=user.id,
                session=session.id,
                content=context.args.get("content"),
                attachments=context.args.get("attachments") or [],
            )

            context = PromptSessionContext(
                session=session,
                initiating_user_id=initiating_user_id,
                message=new_message,
                llm_config=LLMConfig(model="claude-sonnet-4-5-20250929"),
            )

            if context.args.get("extra_tools"):
                context.extra_tools = {
                    k: Tool.load(k) for k in context.args.get("extra_tools")
                }

            await add_chat_message(session, context)

    return {"output": [{"session": session_id}]}
