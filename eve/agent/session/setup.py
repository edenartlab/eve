from typing import List, Optional

from bson import ObjectId
from fastapi import BackgroundTasks
from loguru import logger

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

logger.info("[SETUP MODULE] setup.py loaded - agent_sessions support ENABLED")


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
        session=[session_id],
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


def create_agent_sessions(
    parent_session: Session, agents: List[ObjectId]
) -> dict[str, ObjectId]:
    """Create private agent_sessions for all agents in a multi-agent session.

    Each agent_session:
    - Has the single agent as THE agent
    - Inherits users from parent
    - Has parent_session pointing to the parent
    - Uses session_type="passive" (orchestration controls the flow)

    Args:
        parent_session: The parent multi-agent session
        agents: List of agent ObjectIds to create sessions for

    Returns:
        Dict mapping agent_id (str) to agent_session ObjectId
    """
    agent_sessions = {}

    for agent_id in agents:
        agent = Agent.from_mongo(agent_id)
        if not agent:
            continue

        # Build context explaining the private workspace purpose
        workspace_context = (
            f"This is your private workspace for the multi-agent chatroom '{parent_session.title or 'Untitled'}'. "
            f"You receive messages from other participants as CHAT MESSAGE NOTIFICATIONS. "
            f"Each notification includes the sender's name and a Message ID that references "
            f"the original message in the shared chatroom. "
            f"When you're ready to contribute to the conversation, use the post_to_chatroom tool "
            f"to send your message to all participants in the chatroom. "
            f"Remember, this space is private. The other participants cannot see anything you write or make here -- all they see is the messages you post via the post_to_chatroom tool."
        )

        agent_session = Session(
            owner=parent_session.owner,
            users=parent_session.users.copy() if parent_session.users else [],
            agents=[agent_id],  # Single agent
            parent_session=parent_session.id,
            session_type="passive",  # Not automatic - orchestration controls it
            status="active",
            title=f"Workspace: {parent_session.title}"
            if parent_session.title
            else f"Workspace: {agent.username}",
            context=workspace_context,  # Explain the private workspace purpose
        )
        agent_session.save()

        agent_sessions[str(agent_id)] = agent_session.id

    return agent_sessions


def setup_session(
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request: PromptSessionRequest = None,
):
    logger.info(
        f"[SETUP] setup_session called: session_id={session_id}, has_creation_args={bool(request and request.creation_args)}"
    )

    if session_id:
        session = Session.from_mongo(ObjectId(session_id))
        if not session:
            raise APIError(f"Session not found: {session_id}", status_code=404)

        logger.info(
            f"[SETUP] Returning existing session {session_id}, agent_sessions={session.agent_sessions}"
        )

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

    # Automatic sessions start paused - user must explicitly activate them
    initial_status = (
        "paused" if request.creation_args.session_type == "automatic" else "active"
    )

    session_kwargs = {
        "owner": ObjectId(request.creation_args.owner_id or user_id),
        "agents": agent_object_ids,
        "title": request.creation_args.title,
        "session_key": request.creation_args.session_key,
        "session_type": request.creation_args.session_type,
        "platform": request.creation_args.platform,
        "status": initial_status,
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

    if request.creation_args.context:
        session_kwargs["context"] = request.creation_args.context

    if request.creation_args.settings:
        from eve.agent.session.models import SessionSettings

        session_kwargs["settings"] = SessionSettings(**request.creation_args.settings)

    session = Session(**session_kwargs)

    if _is_test_prompt_request(request):
        session.title = "test thread"

    session.save()

    # Create agent_sessions for multi-agent sessions
    # Each agent gets their own private workspace for orchestration
    logger.info(
        f"[SETUP] Session {session.id} created with {len(agent_object_ids)} agents"
    )
    if len(agent_object_ids) > 1:
        logger.info(
            f"[SETUP] Creating agent_sessions for multi-agent session {session.id}"
        )
        agent_sessions = create_agent_sessions(session, agent_object_ids)
        session.update(agent_sessions=agent_sessions)
        session.agent_sessions = agent_sessions
        logger.info(f"[SETUP] Created agent_sessions: {agent_sessions}")

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
