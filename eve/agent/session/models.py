from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

from eve.mongo import Collection, Document
from eve.tool import Tool


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
class LLMContext:
    messages: List[ChatMessage]
    tools: Optional[List[Tool]] = None
    session_id: Optional[ObjectId] = None
    initiating_user_id: Optional[ObjectId] = None


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"


@dataclass
class PromptSessionContext:
    session_id: ObjectId
    initiating_user_id: Optional[ObjectId] = None
    message: Optional[str] = None


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
