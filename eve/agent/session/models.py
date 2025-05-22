from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from bson import ObjectId
from pydantic import ConfigDict, Field, BaseModel
from dataclasses import dataclass, field

from eve.mongo import Collection, Document
from eve.tool import Tool


class UpdateType(Enum):
    START_PROMPT = "start_prompt"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_COMPLETE = "tool_complete"
    ERROR = "error"
    END_PROMPT = "end_prompt"


@dataclass
class ToolCall:
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
class ChatMessageRequestInput:
    content: str
    attachments: Optional[List[str]] = None
    sender_name: Optional[str] = None


@dataclass
class SessionUpdate:
    type: UpdateType
    message: Optional[ChatMessage] = None
    tool_name: Optional[str] = None
    tool_index: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    text: Optional[str] = None


class LLMTraceMetadata(BaseModel):
    session_id: str = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMContextMetadata(BaseModel):
    trace_name: Optional[str] = None
    trace_id: Optional[str] = None
    generation_name: Optional[str] = None
    generation_id: Optional[str] = None
    trace_metadata: Optional[LLMTraceMetadata] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class LLMConfig:
    model: Optional[str] = "gpt-4o-mini"


@dataclass
class LLMContext:
    messages: List[ChatMessage]
    config: LLMConfig = field(default_factory=LLMConfig)
    tools: Optional[List[Tool]] = None
    metadata: LLMContextMetadata = None
    trace_metadata: LLMTraceMetadata = None


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


@dataclass
class UpdateConfig:
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
    user_is_bot: Optional[bool] = False


@dataclass
class PromptSessionContext:
    session: Session
    initiating_user_id: str
    message: ChatMessageRequestInput
    update_config: Optional[UpdateConfig] = None
    actor_agent_id: Optional[str] = None
    llm_config: Optional[LLMConfig] = None


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    stop: Optional[str] = None
