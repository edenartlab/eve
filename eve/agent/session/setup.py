from typing import List, Optional

from bson import ObjectId
from fastapi import BackgroundTasks

from eve.agent import Agent
from eve.agent.llm.util import is_test_mode_prompt
from eve.agent.session.functions import async_title_session
from eve.agent.session.models import (
    ChatMessage,
    EdenMessageAgentData,
    EdenMessageData,
    EdenMessageType,
    Session,
)
from eve.api.api_requests import PromptSessionRequest
from eve.api.errors import APIError
from eve.trigger import Trigger


def _is_test_prompt_request(request: PromptSessionRequest) -> bool:
    if not request or not request.message:
        return False
    content = getattr(request.message, "content", None)
    return bool(content and is_test_mode_prompt(content))


def create_eden_message(
    session_id: ObjectId, message_type: EdenMessageType, agents: List[Agent]
) -> ChatMessage:
    """Create an eden message for agent operations"""
    eden_message = ChatMessage(
        session=session_id,
        sender=ObjectId("000000000000000000000000"),  # System sender
        role="eden",
        content="",
        eden_message_data=EdenMessageData(
            message_type=message_type,
            agents=[
                EdenMessageAgentData(
                    id=agent.id,
                    name=agent.name or agent.username,
                    avatar=agent.userImage,
                )
                for agent in agents
            ],
        ),
    )
    eden_message.save()
    return eden_message


def generate_session_title(
    session: Session, request: PromptSessionRequest, background_tasks: BackgroundTasks
):
    if session.title:
        return

    if request.creation_args and request.creation_args.title:
        return

    if _is_test_prompt_request(request):
        if session.title != "test thread":
            session.update(title="test thread")
            session.title = "test thread"
        return

    if background_tasks:
        background_tasks.add_task(async_title_session, session, request.message.content)


def setup_session(
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request: PromptSessionRequest = None,
):
    if session_id:
        session = Session.from_mongo(ObjectId(session_id))
        if not session:
            raise APIError(f"Session not found: {session_id}", status_code=404)

        # TODO: titling
        if background_tasks:
            generate_session_title(session, request, background_tasks)
        return session

    if not request.creation_args:
        raise APIError(
            "Session creation requires additional parameters", status_code=400
        )

    # Create new session
    agent_object_ids = [ObjectId(agent_id) for agent_id in request.creation_args.agents]
    session_kwargs = {
        "owner": ObjectId(request.creation_args.owner_id or user_id),
        "agents": agent_object_ids,
        "title": request.creation_args.title,
        "session_key": request.creation_args.session_key,
        "session_type": request.creation_args.session_type,
        "platform": request.creation_args.platform,
        "status": "active",
        "trigger": ObjectId(request.creation_args.trigger)
        if request.creation_args.trigger
        else None,
    }

    # Only include budget if it's not None, so default factory can work
    if request.creation_args.budget is not None:
        session_kwargs["budget"] = request.creation_args.budget

    if request.creation_args.parent_session:
        session_kwargs["parent_session"] = ObjectId(
            request.creation_args.parent_session
        )

    if request.creation_args.extras:
        session_kwargs["extras"] = request.creation_args.extras

    session = Session(**session_kwargs)

    if _is_test_prompt_request(request):
        session.title = "test thread"

    session.save()

    # Update trigger with session ID
    if request.creation_args.trigger:
        trigger = Trigger.from_mongo(ObjectId(request.creation_args.trigger))
        if trigger and not trigger.deleted:
            trigger.session = session.id
            trigger.save()

    # Create eden message for initial agent additions
    agents = [Agent.from_mongo(agent_id) for agent_id in agent_object_ids]
    agents = [agent for agent in agents if agent]  # Filter out None values
    if agents:
        create_eden_message(session.id, EdenMessageType.AGENT_ADD, agents)

    # Generate title for new sessions if no title provided and we have background tasks
    # TODO: titling
    if background_tasks:
        generate_session_title(session, request, background_tasks)

    return session
