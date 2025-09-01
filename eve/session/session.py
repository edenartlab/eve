from enum import Enum
from bson import ObjectId
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal
from pydantic import ConfigDict, Field, BaseModel, field_serializer

from eve.agent import Agent
from eve.mongo import Collection, Document
from eve.session.channel import Channel
from eve.session.message import ChatMessage, EdenMessageData, EdenMessageAgentData, EdenMessageType
from eve.agent.session.models import PromptSessionContext

from eve.api.errors import APIError
from eve.api.api_requests import PromptSessionRequest


class ActorSelectionMethod(Enum):
    RANDOM = "random"
    RANDOM_EXCLUDE_LAST = "random_exclude_last"


class SessionAutonomySettings(BaseModel):
    auto_reply: bool = False
    reply_interval: int = 0
    actor_selection_method: ActorSelectionMethod = ActorSelectionMethod.RANDOM

    @field_serializer("actor_selection_method")
    def serialize_actor_selection_method(self, value: ActorSelectionMethod) -> str:
        return value.value


class SessionBudget(BaseModel):
    token_budget: Optional[int] = None
    manna_budget: Optional[float] = None
    turn_budget: Optional[int] = None
    tokens_spent: Optional[int] = 0
    manna_spent: Optional[float] = 0
    turns_spent: Optional[int] = 0


class SessionMemoryContext(BaseModel):
    """Session-level memory context and caching"""
    # Core memory caching
    cached_memory_context: Optional[str] = None
    memory_context_timestamp: Optional[datetime] = None
    
    # Session activity tracking
    last_activity: Optional[datetime] = None
    last_memory_message_id: Optional[ObjectId] = None
    messages_since_memory_formation: int = 0
    
    # Memory freshness timestamps
    agent_memory_timestamp: Optional[datetime] = None
    user_memory_timestamp: Optional[datetime] = None
    
    # Episode memory caching
    cached_episode_memories: Optional[List[Dict[str, Any]]] = None
    episode_memories_timestamp: Optional[datetime] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("sessions")
class Session(Document):
    owner: ObjectId
    users: Optional[List[ObjectId]] = None  # List of allowed users (defaults to null)
    session_key: Optional[str] = None
    channel: Optional[Channel] = None
    parent_session: Optional[ObjectId] = None
    agents: List[ObjectId] = Field(default_factory=list)
    status: Literal["active", "archived"] = "active"
    messages: List[ObjectId] = Field(default_factory=list)
    memory_context: Optional[SessionMemoryContext] = Field(default_factory=SessionMemoryContext)
    title: Optional[str] = None
    scenario: Optional[str] = None
    autonomy_settings: Optional[SessionAutonomySettings] = None
    last_actor_id: Optional[ObjectId] = None
    budget: SessionBudget = SessionBudget()
    platform: Optional[Literal["discord", "telegram", "twitter", "farcaster"]] = None
    trigger: Optional[ObjectId] = None
    active_requests: Optional[List[str]] = []

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist for optimal query performance"""
        collection = cls.get_collection()
        collection.create_index([
            ("memory_context.last_activity", 1),
            ("memory_context.messages_since_memory_formation", 1), 
            ("status", 1)
        ], name="cold_sessions_idx", background=True)





def add_user_message(
    session: Session, 
    context: PromptSessionContext,
    pin: bool = False
):
    new_message = ChatMessage(
        session=session.id,
        sender=ObjectId(context.initiating_user_id),
        role="user",
        content=context.message.content,
        attachments=context.message.attachments or [],
    )

    if pin:
        new_message.pinned = True
    
    new_message.save()

    session.messages.append(new_message.id)

    session.memory_context.last_activity = datetime.now(timezone.utc)
    session.memory_context.messages_since_memory_formation += 1

    session.save()

    return new_message







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
    agent_ids = [
        ObjectId(agent_id) for agent_id in request.creation_args.agents
    ]

    session_kwargs = {
        "owner": ObjectId(request.creation_args.owner_id or user_id),
        "agents": agent_ids,
        "title": request.creation_args.title,
        "scenario": request.creation_args.scenario,
        "session_key": request.creation_args.session_key,
        "platform": request.creation_args.platform,
        "status": "active",
    }

    if request.creation_args.trigger:
        session_kwargs["trigger"] = ObjectId(request.creation_args.trigger)

    # Set specific session ID if provided
    if request.creation_args.session_id:
        session_kwargs["_id"] = ObjectId(request.creation_args.session_id)

    # Include budget if set
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
    agents = [Agent.from_mongo(agent_id) for agent_id in agent_ids]
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



def create_eden_message(
    session_id: ObjectId, 
    message_type: EdenMessageType, 
    agents: List[Agent]
) -> ChatMessage:
    """Create an eden message for agent operations"""

    message_data = EdenMessageData(
        message_type=message_type,
        agents=[
            EdenMessageAgentData(
                id=agent.id,
                name=agent.name or agent.username,
                avatar=agent.userImage,
            )
            for agent in agents
        ],
    )

    eden_message = ChatMessage(
        session=session_id,
        role="eden",
        eden_message_data=message_data
    )
    
    eden_message.save()
    
    return eden_message







"""

def generate_session_title(
    session: Session, request: PromptSessionRequest, background_tasks: BackgroundTasks
):
    from eve.agent.session.session import async_title_session

    if session.title:
        return

    if request.creation_args and request.creation_args.title:
        return

    background_tasks.add_task(async_title_session, session, request.message.content)

"""