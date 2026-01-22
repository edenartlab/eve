from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from eve.agent.session.models import (
    ChatMessageRequestInput,
    ClientType,
    DeploymentConfig,
    DeploymentSecrets,
    LLMConfig,
    NotificationChannel,
    NotificationPriority,
    NotificationType,
    SessionUpdateConfig,
    UpdateType,
)


class TaskRequest(BaseModel):
    tool: str
    args: dict
    user_id: str
    public: bool = False
    metadata: Optional[Dict[str, Any]] = None


class RealtimeToolRequest(BaseModel):
    """Request for realtime tool execution from ElevenLabs client tools."""

    tool_name: str  # "create", "display", "create_async"
    args: Dict[str, Any]
    user_id: str
    session_id: str
    wait_for_response: bool = True  # True for blocking, False for fire-and-forget
    timeout_seconds: int = 120  # Max wait time for blocking calls


class CancelRequest(BaseModel):
    taskId: str
    user: str


class CancelSessionRequest(BaseModel):
    session_id: str
    user_id: str
    trace_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_call_index: Optional[int] = None


class UpdateSessionStatusRequest(BaseModel):
    session_id: str
    status: Literal["active", "paused", "stopped", "archived"]


class UpdateSessionFieldsRequest(BaseModel):
    session_id: str
    context: Optional[str] = Field(
        default=None, description="Session context - use empty string to clear"
    )
    title: Optional[str] = None
    # Add other updateable fields as needed

    model_config = ConfigDict(extra="forbid")  # Prevent arbitrary fields


class UpdateConfig(BaseModel):
    sub_channel_name: Optional[str] = None
    update_endpoint: Optional[str] = None
    deployment_id: Optional[str] = None
    discord_channel_id: Optional[str] = None
    discord_message_id: Optional[str] = None
    discord_user_id: Optional[str] = None
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


class AllowedChannel(BaseModel):
    id: str
    note: str


class CreateConceptRequest(BaseModel):
    concept: str


class UpdateConceptRequest(BaseModel):
    concept: str


class AgentDeploymentConfig(BaseModel):
    discord_channel_allowlist: Optional[List[AllowedChannel]] = None
    telegram_topic_allowlist: Optional[List[AllowedChannel]] = None


class AgentToolsUpdateRequest(BaseModel):
    agent_id: str
    tools: Dict[str, Dict]


class AgentToolsDeleteRequest(BaseModel):
    agent_id: str
    tools: List[str]


class RunTriggerRequest(BaseModel):
    trigger_id: str


class SessionCreationArgs(BaseModel):
    owner_id: Optional[str] = None
    agents: List[str]
    title: Optional[str] = None
    budget: Optional[float] = None
    trigger: Optional[str] = None
    session_key: Optional[str] = None
    session_type: Literal["passive", "natural", "automatic"] = "passive"
    parent_session: Optional[str] = None
    platform: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None
    context: Optional[str] = None  # Scenario/premise for automatic multi-agent sessions
    settings: Optional[Dict[str, Any]] = None  # SessionSettings (e.g., delay_interval)
    visible: Optional[bool] = None  # Visibility flag for public listing


class PromptSessionRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[ChatMessageRequestInput] = None
    user_id: Optional[str] = None  # The user who owns/initiates the session
    acting_user_id: Optional[str] = (
        None  # The user whose permissions are used for tool authorization (defaults to user_id if not provided)
    )
    actor_agent_ids: Optional[List[str]] = None
    update_config: Optional[SessionUpdateConfig] = None
    llm_config: Optional[LLMConfig] = None
    stream: bool = False
    notification_config: Optional[Dict[str, Any]] = None
    thinking: Optional[bool] = None  # Override agent's thinking policy per-message
    api_key_id: Optional[str] = None  # API key ID to attach to messages
    trigger: Optional[str] = (
        None  # Mark message as coming from a trigger (for memory skip)
    )

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
    secrets: Optional[DeploymentSecrets] = None


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


# Notification API requests
class CreateNotificationRequest(BaseModel):
    user_id: str
    type: NotificationType
    title: str
    message: str
    priority: Optional[NotificationPriority] = NotificationPriority.NORMAL
    channels: Optional[List[NotificationChannel]] = None
    trigger_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    action_url: Optional[str] = None
    expires_at: Optional[datetime] = None


class EmbedSearchRequest(BaseModel):
    query: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    tool: Optional[str] = None
    limit: Optional[int] = 20


class LLMMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None


class AsyncLLMCallRequest(BaseModel):
    user_id: str
    messages: List[LLMMessage]
    system_message: Optional[str] = None
    model: Optional[str] = None
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    response_model: Optional[str] = None  # Not implemented for now


class AgentPromptsExtractionRequest(BaseModel):
    user_id: str
    session_id: str
    agent_name: Optional[str] = None


class ReactionRequest(BaseModel):
    message_id: str
    tool_call_id: Optional[str] = None  # If reacting to a specific tool call
    reaction: str  # The reaction emoji or key
    user_id: Optional[str] = None  # The user who reacted (optional)


# Discord channel management requests
class GetDiscordChannelsRequest(BaseModel):
    deployment_id: str
    user_id: str  # For ownership verification


class DiscordChannelInfo(BaseModel):
    id: str
    name: str
    type: int
    type_name: str
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    position: int = 0


class DiscordGuildInfo(BaseModel):
    id: str
    name: str
    icon: Optional[str] = None
    member_count: Optional[int] = None
    channels: List[DiscordChannelInfo]


class GetDiscordChannelsResponse(BaseModel):
    guilds: List[DiscordGuildInfo]
    last_refreshed_at: Optional[str] = None


class RefreshDiscordChannelsRequest(BaseModel):
    deployment_id: str
    user_id: str


class RefreshDiscordChannelsResponse(BaseModel):
    success: bool
    guilds_count: int
    channels_count: int
    guilds: List[Dict[str, Any]]


class SyncDiscordChannelsRequest(BaseModel):
    deployment_id: str
    user_id: str


class SyncDiscordChannelsResponse(BaseModel):
    success: bool
    total_channels: int
    channels_with_sessions: Dict[str, str]  # channel_id -> session_id
    channels_without_sessions: List[str]
    sessions_created: List[Dict[str, Any]]  # [{session_id, channel_id, title}]
    backfill_results: Dict[str, int]  # channel_id -> messages_backfilled
