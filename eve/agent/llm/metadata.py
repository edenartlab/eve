"""Helper utilities for constructing LLM metadata payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from eve.agent.session.models import LLMContextMetadata, LLMTraceMetadata


@dataclass
class ToolMetadataBuilder:
    """Factory for consistent tool metadata envelopes."""

    tool_name: str
    litellm_session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    def __call__(self) -> LLMContextMetadata:
        return LLMContextMetadata(
            session_id=self.litellm_session_id,
            trace_name=f"TOOL_{self.tool_name}",
            generation_name=f"TOOL_{self.tool_name}",
            trace_metadata=LLMTraceMetadata(
                user_id=str(self.user_id) if self.user_id else None,
                agent_id=str(self.agent_id) if self.agent_id else None,
                session_id=str(self.session_id) if self.session_id else None,
            ),
        )


__all__ = ["ToolMetadataBuilder"]
