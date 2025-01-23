from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

from eve.models import ClientType
from eve.thread import UserMessage


class TaskRequest(BaseModel):
    tool: str
    args: dict
    user_id: str


class CancelRequest(BaseModel):
    taskId: str
    user: str


class UpdateConfig(BaseModel):
    sub_channel_name: Optional[str] = None
    update_endpoint: Optional[str] = None
    discord_channel_id: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_message_id: Optional[str] = None
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
    dont_reply: bool = False
    model: Optional[str] = None


class CronSchedule(BaseModel):
    year: Optional[int | str] = Field(None, description="4-digit year")
    month: Optional[int | str] = Field(None, description="month (1-12)")
    day: Optional[int | str] = Field(None, description="day of month (1-31)")
    week: Optional[int | str] = Field(None, description="ISO week (1-53)")
    day_of_week: Optional[int | str] = Field(
        None,
        description="number or name of weekday (0-6 or mon,tue,wed,thu,fri,sat,sun)",
    )
    hour: Optional[int | str] = Field(None, description="hour (0-23)")
    minute: Optional[int | str] = Field(None, description="minute (0-59)")
    second: Optional[int | str] = Field(None, description="second (0-59)")
    start_date: Optional[datetime] = Field(
        None, description="earliest possible date/time to trigger on (inclusive)"
    )
    end_date: Optional[datetime] = Field(
        None, description="latest possible date/time to trigger on (inclusive)"
    )
    timezone: Optional[str] = Field(
        None, description="time zone to use for the date/time calculations"
    )

    def to_cron_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class CreateTriggerRequest(BaseModel):
    agent_id: str
    user_id: str
    message: str
    schedule: CronSchedule
    update_config: Optional[UpdateConfig] = None


class DeleteTriggerRequest(BaseModel):
    id: str


class CreateDeploymentRequest(BaseModel):
    agent_key: str
    platform: ClientType
    credentials: Optional[Dict[str, str]] = None


class DeleteDeploymentRequest(BaseModel):
    agent_key: str
    platform: ClientType
