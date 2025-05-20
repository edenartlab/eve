from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Dict, Any, Optional, List
from bson import ObjectId
from dataclasses import dataclass
from eve.agent.session.session_llm import LLMConfig, LLMContext, async_prompt
from eve.mongo import Collection, Document


@dataclass
class PromptSessionContext:
    session_id: ObjectId
    initiating_user_id: Optional[ObjectId] = None
    message: Optional[str] = None


@Collection("channels")
class Channel(Document):
    type: Literal["eden", "discord", "telegram", "twitter"]
    key: str


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


@Collection("messages")
class ChatMessage(Document):
    session: ObjectId
    channel: Optional[Channel] = None
    reply_to: Optional[ObjectId] = None
    sender: ObjectId = None
    role: Literal["user", "assistant", "system"]
    reactions: Optional[Dict[str, List[ObjectId]]] = {}
    content: Optional[str] = None
    attachments: Optional[List[str]] = []
    tool_calls: Optional[List[ToolCall]] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)


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


def create_session(
    owner: ObjectId,
    title: str,
    agents: List[ObjectId],
    scenario: Optional[str] = None,
    budget: Optional[float] = None,
):
    session = Session(
        owner=owner, title=title, agents=agents, scenario=scenario, budget=budget
    )
    session.save()
    return session


def list_sessions(owner: ObjectId):
    return Session.find_many(Session.owner == owner)


def get_session(session_id: ObjectId):
    return Session.from_mongo(session_id)


def archive_session(session_id: ObjectId):
    session = Session.from_mongo(session_id)
    session.status = "archived"
    session.save()
    return session


async def prompt_session(
    context: PromptSessionContext, config: Optional[LLMConfig] = None
):
    # TODO: Determine who should act, what messages are available in context, and what tools are available
    messages = [ChatMessage(role="user", content=context.message)]
    tools = []
    context = LLMContext(
        session_id=context.session_id,
        initiating_user_id=context.initiating_user_id,
        messages=messages,
        tools=tools,
    )
    await async_prompt(context, config)
