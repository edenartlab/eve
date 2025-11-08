"""High level access to the configured memory backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any

from bson import ObjectId

from .backends import MemoryBackend, MongoMemoryBackend

if TYPE_CHECKING:  # pragma: no cover
    from eve.agent import Agent
    from eve.agent.session.models import Session
    from eve.user import User


class MemoryService:
    """Facade coordinating access to a concrete memory backend."""

    def __init__(self, backend: Optional[MemoryBackend] = None) -> None:
        self._backend: MemoryBackend = backend or MongoMemoryBackend()

    @property
    def backend(self) -> MemoryBackend:
        return self._backend

    def configure(self, backend: MemoryBackend) -> None:
        """Swap the underlying backend implementation."""
        self._backend = backend

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
        return await self._backend.assemble_memory_context(
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
        return await self._backend.maybe_form_memories(agent_id, session, agent)

    async def form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
        *,
        conversation_text: Optional[str] = None,
        char_counts_by_source: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return await self._backend.form_memories(
            agent_id,
            session,
            agent,
            conversation_text=conversation_text,
            char_counts_by_source=char_counts_by_source,
        )


memory_service = MemoryService()
