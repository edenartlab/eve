from typing import Optional
from pydantic import BaseModel, ConfigDict

from eve.thread import UserMessage


class TaskRequest(BaseModel):
    tool: str
    args: dict
    user_id: str


class CancelRequest(BaseModel):
    task_id: str
    user_id: str


class UpdateConfig(BaseModel):
    sub_channel_name: Optional[str] = None
    update_endpoint: Optional[str] = None
    discord_channel_id: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    cast_hash: Optional[str] = None
    author_fid: Optional[int] = None
    message_id: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChatRequest(BaseModel):
    user_id: str
    agent_id: str
    user_message: UserMessage
    thread_id: Optional[str] = None
    update_config: Optional[UpdateConfig] = None
    force_reply: bool = False


class ScheduleRequest(BaseModel):
    agent_id: str
    user_id: str
    instruction: str
