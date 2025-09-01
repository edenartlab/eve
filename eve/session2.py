import json
import logging
import os
import time
import uuid
import aiohttp
from bson import ObjectId
from typing import Dict, List, Optional
from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
import aiohttp

from eve.agent.deployments.farcaster import FarcasterClient
from eve.agent.session.models import (
    PromptSessionContext,
    Session,
    ChatMessage,
    EdenMessageType,
    EdenMessageData,
    EdenMessageAgentData,
    Trigger,
    Deployment,
    DeploymentConfig,
    Notification,
    NotificationChannel,
    NotificationConfig,
)
from eve.deploy import (
    Deployment as DeploymentV1,
)
# from eve.agent.session.session import run_prompt_session, run_prompt_session_stream
# from eve.agent.session.triggers import create_trigger_fn, stop_trigger
from eve.api.errors import handle_errors, APIError
from eve.api.api_requests import (
    CancelRequest,
    CancelSessionRequest,
    ChatRequest,
    CreateDeploymentRequestV2,
    CreateTriggerRequest,
    DeleteDeploymentRequestV2,
    DeleteTriggerRequest,
    DeploymentEmissionRequest,
    DeploymentInteractRequest,
    PromptSessionRequest,
    TaskRequest,
    PlatformUpdateRequest,
    UpdateConfig,
    AgentToolsUpdateRequest,
    AgentToolsDeleteRequest,
    UpdateDeploymentRequestV2,
    CreateNotificationRequest,
    RunTriggerRequest
)
# from eve.api.helpers import (
#     emit_update,
#     get_platform_client,
#     # setup_chat,
#     create_telegram_chat_request,
#     update_busy_state,
# )
# from eve.utils import prepare_result, dumps_json, serialize_json
# from eve.tools.replicate_tool import replicate_update_task
# from eve.agent.llm import UpdateType
# from eve.agent.run_thread import async_prompt_thread
# from eve.mongo import get_collection
# from eve.task import Task
# from eve.tool import Tool
from eve.agent import Agent
# from eve.user import User
# from eve.agent.thread import Thread, UserMessage
# from eve.tools.twitter import X
# from eve.api.helpers import get_eden_creation_url

logger = logging.getLogger(__name__)
db = os.getenv("DB", "STAGE").upper()



def generate_session_title(
    session: Session, request: PromptSessionRequest, background_tasks: BackgroundTasks
):
    from eve.agent.session.session import async_title_session

    if session.title:
        return

    if request.creation_args and request.creation_args.title:
        return

    background_tasks.add_task(async_title_session, session, request.message.content)



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

def setup_session(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request: PromptSessionRequest = None,
):
    if session_id:
        session = Session.from_mongo(ObjectId(session_id))
        if not session:
            raise APIError(f"Session not found: {session_id}", status_code=404)
        
        # TODO: This is broken without background tasks
        # generate_session_title(session, request, background_tasks)
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
        "scenario": request.creation_args.scenario,
        "session_key": request.creation_args.session_key,
        "platform": request.creation_args.platform,
        "status": "active",
        "trigger": ObjectId(request.creation_args.trigger)
        if request.creation_args.trigger
        else None,
    }

    if request.creation_args.session_id:
        session_kwargs["_id"] = ObjectId(request.creation_args.session_id)

    # Only include budget if it's not None, so default factory can work
    if request.creation_args.budget is not None:
        session_kwargs["budget"] = request.creation_args.budget

    session = Session(**session_kwargs)
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
        eden_message = create_eden_message(
            session.id, EdenMessageType.AGENT_ADD, agents
        )
        session.messages.append(eden_message.id)
        session.save()

    # Generate title for new sessions if no title provided and we have background tasks
    # TODO: This is broken without background tasks
    # generate_session_title(session, request, background_tasks)

    return session
