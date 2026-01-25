from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionMessageInput(BaseModel):
    content: str
    attachments: List[str] = Field(default_factory=list)


class SessionCreationInput(BaseModel):
    agent_ids: List[str]


class SessionPromptRequest(BaseModel):
    session_run_id: Optional[str] = None
    session_id: Optional[str] = None
    actor_agent_ids: Optional[List[str]] = None
    thinking: Optional[bool] = None
    creation: Optional[SessionCreationInput] = None
    message: SessionMessageInput
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None


class SessionPromptResponse(BaseModel):
    session_id: str
    session_run_id: str
    message_id: Optional[str] = None


class SessionStreamEvent(BaseModel):
    event: str
    data: Dict[str, Any]
