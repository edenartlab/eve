from enum import Enum
from bson import ObjectId
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal
from pydantic import ConfigDict, Field, BaseModel, field_serializer

from eve.mongo import Collection, Document
from eve.session.channel import Channel
from eve.session.message import ChatMessage
from eve.agent.session.models import PromptSessionContext


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
