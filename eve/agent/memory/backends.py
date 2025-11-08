"""Memory backend abstractions and concrete implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from bson import ObjectId

if TYPE_CHECKING:  # pragma: no cover
    from eve.agent import Agent
    from eve.agent.session.models import Session
    from eve.user import User


logger = logging.getLogger(__name__)


class MemoryBackend(ABC):
    """Minimal interface for memory backends."""

    @abstractmethod
    async def assemble_memory_context(
        self,
        session: "Session",
        agent: "Agent",
        user: "User",
        *,
        force_refresh: bool = False,
        reason: str = "unknown",
        skip_save: bool = False,
    ) -> str:
        """Return the memory context string for the given session."""

    @abstractmethod
    async def maybe_form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
    ) -> bool:
        """Conditionally form memories for the supplied session."""

    @abstractmethod
    async def form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
        *,
        conversation_text: Optional[str] = None,
        char_counts_by_source: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Force memory formation for the supplied session."""


class MongoMemoryBackend(MemoryBackend):
    """Adapter around the existing Mongo-backed implementation."""

    async def assemble_memory_context(
        self,
        session: "Session",
        agent: "Agent",
        user: "User",
        *,
        force_refresh: bool = False,
        reason: str = "unknown",
        skip_save: bool = False,
    ) -> str:
        from .memory_assemble_context import assemble_memory_context as _assemble

        return await _assemble(
            session,
            agent,
            user,
            force_refresh=force_refresh,
            reason=reason,
            skip_save=skip_save,
        )

    async def maybe_form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
    ) -> bool:
        from .memory import maybe_form_memories as _maybe_form_memories

        return await _maybe_form_memories(agent_id, session, agent)

    async def form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
        *,
        conversation_text: Optional[str] = None,
        char_counts_by_source: Optional[Dict[str, Any]] = None,
    ) -> bool:
        from .memory import form_memories as _form_memories

        return await _form_memories(
            agent_id,
            session,
            agent,
            conversation_text=conversation_text,
            char_counts_by_source=char_counts_by_source,
        )


class GraphitiMemoryBackend(MemoryBackend):
    """Graphiti-backed implementation placeholder."""

    warning_logged = False

    async def assemble_memory_context(
        self,
        session: "Session",
        agent: "Agent",
        user: "User",
        *,
        force_refresh: bool = False,
        reason: str = "unknown",
        skip_save: bool = False,
    ) -> str:
        self._warn()
        raise NotImplementedError(
            "GraphitiMemoryBackend.assemble_memory_context is not implemented yet."
        )

    async def maybe_form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
    ) -> bool:
        self._warn()
        raise NotImplementedError(
            "GraphitiMemoryBackend.maybe_form_memories is not implemented yet."
        )

    async def form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
        *,
        conversation_text: Optional[str] = None,
        char_counts_by_source: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self._warn()
        raise NotImplementedError(
            "GraphitiMemoryBackend.form_memories is not implemented yet."
        )

    def _warn(self) -> None:
        if not type(self).warning_logged:
            logger.warning(
                "GraphitiMemoryBackend is a stub. Configure the backend once the "
                "Graphiti integration is implemented."
            )
            type(self).warning_logged = True
