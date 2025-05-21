from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

from eve.mongo import Collection, Document
from eve.tool import Tool


class UpdateType(Enum):
    START_PROMPT = "start_prompt"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_COMPLETE = "tool_complete"
    ERROR = "error"
    END_PROMPT = "end_prompt"


class ToolCall(BaseModel):
    id: str
    tool: str
    args: Dict[str, Any]
    task: Optional[ObjectId] = None
    status: Optional[
        Literal["pending", "running", "completed", "failed", "cancelled"]
    ] = None
    result: Optional[List[Dict[str, Any]]] = None
    reactions: Optional[Dict[str, List[ObjectId]]] = None
    error: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("channels")
class Channel(Document):
    type: Literal["eden", "discord", "telegram", "twitter"]
    key: str


@Collection("messages")
class ChatMessage(Document):
    session: ObjectId
    channel: Optional[Channel] = None
    reply_to: Optional[ObjectId] = None
    sender: ObjectId = None
    sender_name: Optional[str] = None
    role: Literal["user", "assistant", "system"]
    reactions: Optional[Dict[str, List[ObjectId]]] = {}
    content: Optional[str] = None
    attachments: Optional[List[str]] = []
    tool_calls: Optional[List[ToolCall]] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_content(self, schema, truncate_images=False):
        return self.content

    def openai_schema(self, truncate_images=False):
        return {
            "role": self.role,
            "content": self._get_content("openai", truncate_images=truncate_images),
            **({"name": self.sender_name} if self.sender_name else {}),
        }


@dataclass
class LLMTraceMetadata:
    session_id: str = None
    initiating_user_id: Optional[str] = None
    actor_agent_id: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMContextMetadata:
    trace_name: Optional[str] = None
    trace_id: Optional[str] = None
    generation_name: Optional[str] = None
    generation_id: Optional[ObjectId] = None
    trace_metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"


@dataclass
class LLMContext:
    messages: List[ChatMessage]
    tools: Optional[List[Tool]] = None
    metadata: LLMContextMetadata = Field(default_factory=LLMContextMetadata)
    config: Optional[LLMConfig] = None


@Collection("sessions")
class Session(Document):
    owner: ObjectId
    channel: Optional[Channel] = None
    title: str
    agents: List[ObjectId] = Field(default_factory=list)
    scenario: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    budget: Optional[float] = None
    spent: Optional[float] = 0
    status: Optional[Literal["active", "archived"]] = "active"


class UpdateConfig(BaseModel):
    sub_channel_name: Optional[str] = None
    update_endpoint: Optional[str] = None
    deployment_id: Optional[str] = None
    discord_channel_id: Optional[str] = None
    discord_message_id: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_message_id: Optional[str] = None
    telegram_thread_id: Optional[str] = None
    farcaster_hash: Optional[str] = None
    farcaster_author_fid: Optional[int] = None
    farcaster_message_id: Optional[str] = None
    twitter_tweet_to_reply_id: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_is_bot: Optional[bool] = False


@dataclass
class PromptSessionContext:
    session: Session
    initiating_user_id: Optional[ObjectId] = None
    actor_agent_id: Optional[ObjectId] = None
    message: Optional[str] = None
    update_config: Optional[UpdateConfig] = None
