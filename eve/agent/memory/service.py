"""High level access to the configured memory backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from bson import ObjectId

from .backends import MemoryBackend, MongoMemoryBackend

# =============================================================================
# MEMORY SYSTEM TOGGLE
# =============================================================================
# Toggle between memory systems:
#   - True  = memory v2 (eve/agent/memory2) - new reflection/facts based system
#   - False = memory v1 (eve/agent/memory)  - original memory system
#
# The session module (eve/agent/session) imports `memory_service` from here,
# so this single toggle controls the entire application's memory backend.
# =============================================================================

USE_MEMORY_V2 = False

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
        instrumentation=None,
    ) -> str:
        return await self._backend.assemble_memory_context(
            session,
            agent,
            user,
            force_refresh=force_refresh,
            reason=reason,
            skip_save=skip_save,
            instrumentation=instrumentation,
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


def _create_backend() -> MemoryBackend:
    """Create the appropriate backend based on USE_MEMORY_V2 toggle above."""
    if USE_MEMORY_V2:
        from eve.agent.memory2.backend import Memory2Backend

        return Memory2Backend()
    return MongoMemoryBackend()


# Global singleton - this is imported by eve/agent/session modules
memory_service = MemoryService(backend=_create_backend())
