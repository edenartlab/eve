import json
from bson import ObjectId
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from pydantic import ConfigDict, BaseModel, field_serializer

from eve.utils import prepare_result, dumps_json
from eve.mongo import Collection, Document
from eve.session.channel import Channel


class ToolCallStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EdenMessageType(Enum):
    AGENT_ADD = "agent_add"
    AGENT_REMOVE = "agent_remove"

class ChatMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    EDEN = "eden"



class ToolCall(BaseModel):
    id: str
    tool: str
    args: Dict[str, Any]
    task: Optional[ObjectId] = None
    cost: Optional[float] = None
    status: Optional[ToolCallStatus] = None
    result: Optional[List[Dict[str, Any]]] = None
    reactions: Optional[Dict[str, List[str]]] = None
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def from_openai(tool_call):
        return ToolCall(
            id=tool_call.id,
            tool=tool_call.function.name,
            args=json.loads(tool_call.function.arguments),
        )

    @staticmethod
    def from_anthropic(tool_call):
        return ToolCall(
            id=tool_call.id, 
            tool=tool_call.name, 
            args=tool_call.input
        )


class EdenMessageAgentData(BaseModel):
    id: ObjectId
    name: str
    avatar: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class EdenMessageData(BaseModel):
    message_type: EdenMessageType
    agents: Optional[List[EdenMessageAgentData]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("message_type")
    def serialize_message_type(self, value: EdenMessageType) -> str:
        return value.value


class ChatMessageObservability(BaseModel):
    provider: Literal["langfuse"] = "langfuse"
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    generation_id: Optional[str] = None
    tokens_spent: Optional[int] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("messages")
class ChatMessage(Document):
    role: ChatMessageRole
    name: Optional[str] = None

    channel: Optional[Channel] = None
    session: Optional[ObjectId] = None
    sender: Optional[ObjectId] = None
    eden_message_data: Optional[EdenMessageData] = None
    reply_to: Optional[ObjectId] = None
    pinned: Optional[bool] = False

    content: str = ""
    reactions: Optional[Dict[str, List[str]]] = {}

    attachments: Optional[List[str]] = []
    tool_calls: Optional[List[ToolCall]] = []
    cost: Optional[float] = None  # todo: add cost

    observability: Optional[ChatMessageObservability] = None
    finish_reason: Optional[str] = None
    thought: Optional[List[Dict[str, Any]]] = None
    llm_config: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def react(self, user: ObjectId, reaction: str):
        if reaction not in self.reactions:
            self.reactions[reaction] = []
        self.reactions[reaction].append(user)

    def as_user_message(self):
        if self.role == "user":
            return self

        attachments = self.attachments.copy()
        for tc in self.tool_calls or []:
            result = prepare_result(tc.result) or []
            urls = [
                item["url"] for r in result for item in r["output"] if item.get("url")
            ]
            attachments.extend([r for r in urls if r])

        content = self.content
        if self.tool_calls:
            tool_calls = "\n".join(
                [dumps_json(tc.model_dump()) for tc in self.tool_calls]
            )
            content += f"\n\n<Tool calls>\n{tool_calls}\n</Tool calls>"

        return self.model_copy(
            update={"role": "user", "content": content, "attachments": attachments}
        )

    def as_assistant_message(self):
        if self.role == "assistant":
            return self

        return self.model_copy(update={"role": "assistant"})

    def as_system_message(self):
        if self.role == "system":
            return self

        return self.model_copy(update={"role": "system"})

    def filter_cancelled_tool_calls(self):
        """Return a copy of the message with cancelled tool calls filtered out"""
        if not self.tool_calls:
            return self

        filtered_tool_calls = [tc for tc in self.tool_calls if tc.status != "cancelled"]

        return self.model_copy(update={"tool_calls": filtered_tool_calls})

