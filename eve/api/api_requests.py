from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

from eve.agent.thread import UserMessage
from eve.agent.llm import UpdateType
from eve.agent.session.models import (
    ChatMessageRequestInput,
    LLMConfig,
    SessionUpdateConfig,
    DeploymentSecrets,
    DeploymentConfig,
    ClientType,
)

class TaskRequest(BaseModel):
    tool: str
    args: dict
    user_id: str
    public: bool = False


class CancelRequest(BaseModel):
    taskId: str
    user: str


class CancelSessionRequest(BaseModel):
    session_id: str
    user_id: str


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


class PlatformUpdateRequest(BaseModel):
    type: UpdateType
    content: Optional[str] = None
    tool: Optional[str] = None
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    update_config: Optional[UpdateConfig] = None


class ChatRequest(BaseModel):
    user_id: str
    agent_id: str
    user_message: UserMessage
    thread_id: Optional[str] = None
    update_config: Optional[UpdateConfig] = None
    force_reply: bool = False
    use_thinking: bool = True
    model: Optional[str] = None
    user_is_bot: Optional[bool] = False


class CronSchedule(BaseModel):
    year: Optional[int | str] = Field(None, description="4-digit year")
    month: Optional[int | str] = Field(None, description="month (1-12)")
    day: Optional[int | str] = Field(None, description="day of month (1-31)")
    day_of_month: Optional[int | str] = Field(None, description="day of month (1-31)")
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


class DeleteTriggerRequest(BaseModel):
    id: str


class AllowedChannel(BaseModel):
    id: str
    note: str


class PostingInstructions(BaseModel):
    session_id: Optional[str] = None
    post_to: Optional[Literal['same', 'another', 'discord', 'telegram', 'x', 'farcaster']] = None
    channel_id: Optional[str] = None
    custom_instructions: Optional[str] = None


class CreateTriggerRequest(BaseModel):
    agent: str
    user: str
    instruction: str
    posting_instructions: Optional[PostingInstructions] = None
    schedule: CronSchedule
    update_config: Optional[UpdateConfig] = None
    session_type: Literal['new', 'another'] = 'new'
    session: Optional[str] = None


class AgentDeploymentConfig(BaseModel):
    discord_channel_allowlist: Optional[List[AllowedChannel]] = None
    telegram_topic_allowlist: Optional[List[AllowedChannel]] = None


class AgentToolsUpdateRequest(BaseModel):
    agent_id: str
    tools: Dict[str, Dict]


class AgentToolsDeleteRequest(BaseModel):
    agent_id: str
    tools: List[str]


class SessionCreationArgs(BaseModel):
    owner_id: Optional[str] = None
    agents: List[str]
    title: Optional[str] = None
    scenario: Optional[str] = None
    budget: Optional[float] = None
    trigger: Optional[str] = None
    session_key: Optional[str] = None
    platform: Optional[str] = None


class PromptSessionRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[ChatMessageRequestInput] = None
    user_id: Optional[str] = None
    actor_agent_id: Optional[str] = None
    actor_agent_ids: Optional[List[str]] = None
    update_config: Optional[SessionUpdateConfig] = None
    llm_config: Optional[LLMConfig] = None
    stream: bool = False

    # Session creation fields (used when session_id is not provided)
    creation_args: Optional[SessionCreationArgs] = None


class CreateDeploymentRequestV2(BaseModel):
    agent: str
    user: str
    platform: ClientType
    secrets: Optional[DeploymentSecrets] = None
    config: Optional[DeploymentConfig] = None


class UpdateDeploymentRequestV2(BaseModel):
    deployment_id: str
    config: Optional[DeploymentConfig] = None


class DeleteDeploymentRequestV2(BaseModel):
    deployment_id: str


class DeploymentInteractRequest(BaseModel):
    deployment_id: str
    interaction: PromptSessionRequest


class DeploymentEmissionRequest(BaseModel):
    type: UpdateType
    update_config: SessionUpdateConfig
    content: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TranscribeRequest(BaseModel):
    user_id: str
    model: Optional[str] = "gpt-4o-mini-transcribe"
