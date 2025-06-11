from enum import Enum
import json
import os
from typing import List, Optional, Dict, Any, Literal
from bson import ObjectId
import magic
from pydantic import ConfigDict, Field, BaseModel, field_serializer
from dataclasses import dataclass, field

from eve.eden_utils import download_file, image_to_base64
from eve.mongo import Collection, Document
from eve.tool import Tool


def serialize_for_json(obj):
    """Recursively serialize objects for JSON, handling ObjectId and other special types"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj


class UpdateType(Enum):
    START_PROMPT = "start_prompt"
    ASSISTANT_TOKEN = "assistant_token"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_COMPLETE = "tool_complete"
    ERROR = "error"
    END_PROMPT = "end_prompt"


class ToolCall(BaseModel):
    id: str
    tool: str
    args: Dict[str, Any]
    task: Optional[ObjectId] = None
    cost: Optional[float] = None
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


class EdenMessageType(Enum):
    AGENT_ADD = "agent_add"
    AGENT_REMOVE = "agent_remove"


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


@Collection("messages")
class ChatMessage(Document):
    role: Literal[
        "user",
        "assistant",
        "system",
        "tool",
        "eden",
    ]
    content: str = ""
    session: Optional[ObjectId] = None
    sender: Optional[ObjectId] = None
    eden_message_data: Optional[EdenMessageData] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    task: Optional[ObjectId] = None
    cost: Optional[float] = None
    channel: Optional[Channel] = None
    reply_to: Optional[ObjectId] = None
    sender_name: Optional[str] = None
    reactions: Optional[Dict[str, List[str]]] = field(default_factory=dict)
    attachments: Optional[List[str]] = []
    tool_calls: Optional[List[ToolCall]] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_content(self, schema, truncate_images=False):
        """Assemble user message content block"""

        # start with original message content
        content = self.content or ""

        # let claude see names
        if self.name and schema == "anthropic":
            content = f"<User>{self.name}</User>\n\n{content}"

        if self.attachments:
            # append attachments info (url and type) to content
            attachment_lines = []
            attachment_files = []
            attachment_errors = []
            for attachment in self.attachments:
                try:
                    attachment_file = download_file(
                        attachment,
                        os.path.join(
                            "/tmp/eden_file_cache/", attachment.split("/")[-1]
                        ),
                        overwrite=False,
                    )
                    mime_type = magic.from_file(attachment_file, mime=True)
                    if "video" in mime_type:
                        attachment_lines.append(
                            f"* {attachment} (The asset is a video, the corresponding image attachment is its first frame.)"
                        )
                        attachment_files.append(attachment_file)
                    elif "image" in mime_type:
                        attachment_lines.append(f"* {attachment}")
                        attachment_files.append(attachment_file)
                    else:
                        attachment_lines.append(
                            f"* {attachment}: (Mime type: {mime_type})"
                        )
                except Exception as e:
                    attachment_errors.append(f"* {attachment}: {str(e)}")

            attachments = ""
            if attachment_lines:
                attachments += "The attached images correspond to the following urls:\n"
                attachments += "\n".join(attachment_lines)
            if attachment_errors:
                attachments += "The following files failed to attach:\n"
                attachments += "\n".join(attachment_errors)
            attachments = f"<attachments>\n{attachments}\n</attachments>"
            content += f"\n{attachments}"

            # add image blocks
            if schema == "anthropic":
                block = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_to_base64(
                                file_path,
                                max_size=512,
                                quality=95,
                                truncate=truncate_images,
                            ),
                        },
                    }
                    for file_path in attachment_files
                ]
            elif schema == "openai":
                block = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"""data:image/jpeg;base64,{image_to_base64(
                            file_path, 
                            max_size=512, 
                            quality=95, 
                            truncate=truncate_images
                            
                        )}"""
                        },
                    }
                    for file_path in attachment_files
                ]

            if content:
                block.extend([{"type": "text", "text": content.strip()}])

            content = block

        return content

    def openai_schema(self, truncate_images=False):
        base_schema = {
            "role": self.role,
            "content": self._get_content("openai", truncate_images=truncate_images),
            **({"attachments": self.attachments} if self.attachments else {}),
            **({"name": self.sender_name} if self.sender_name else {}),
            **({"tool_call_id": self.tool_call_id} if self.tool_call_id else {}),
        }
        if self.tool_calls:
            base_schema["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.tool,
                        "arguments": json.dumps(serialize_for_json(tool_call.args)),
                    },
                }
                for tool_call in self.tool_calls
            ]
        return base_schema


@dataclass
class ChatMessageRequestInput:
    content: str
    attachments: Optional[List[str]] = None
    sender_name: Optional[str] = None


class SessionUpdate(BaseModel):
    type: UpdateType
    message: Optional[ChatMessage] = None
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_index: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    update_config: Optional[Dict[str, Any]] = None
    agent: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMTraceMetadata(BaseModel):
    session_id: str = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LLMContextMetadata(BaseModel):
    session_id: Optional[str] = None
    trace_name: Optional[str] = None
    generation_name: Optional[str] = None
    trace_metadata: Optional[LLMTraceMetadata] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class LLMConfig:
    model: Optional[str] = "gpt-4o-mini"


@dataclass
class LLMContext:
    messages: List[Any]
    config: LLMConfig = field(default_factory=LLMConfig)
    tools: Optional[List[Tool]] = None
    metadata: LLMContextMetadata = None


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


@Collection("triggers2")
class Trigger(Document):
    trigger_id: str
    user: ObjectId
    schedule: Dict[str, Any]
    instruction: str
    agent: Optional[ObjectId] = None
    session: Optional[ObjectId] = None
    update_config: Optional[Dict[str, Any]] = None
    status: Optional[Literal["active", "paused", "finished"]] = "active"


class SessionContext(BaseModel):
    memories: Optional[List[ObjectId]] = []
    memory_updated: Optional[ObjectId] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


@Collection("sessions")
class Session(Document):
    owner: ObjectId
    channel: Optional[Channel] = None
    agents: List[ObjectId] = Field(default_factory=list)
    status: Literal["active", "archived"] = "active"
    messages: List[ObjectId] = Field(default_factory=list)
    context: Optional[SessionContext] = SessionContext()
    title: Optional[str] = None
    scenario: Optional[str] = None
    autonomy_settings: Optional[SessionAutonomySettings] = None
    last_actor_id: Optional[ObjectId] = None
    budget: SessionBudget = SessionBudget()
    platform: Optional[Literal["discord", "telegram", "twitter", "farcaster"]] = None
    trigger: Optional[ObjectId] = None


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
    tokens_spent: Optional[int] = None
