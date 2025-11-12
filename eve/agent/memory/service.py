"""High level access to the configured memory backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from bson import ObjectId

from .backends import MemoryBackend, MongoMemoryBackend

if TYPE_CHECKING:  # pragma: no cover
    from eve.agent import Agent
    from eve.agent.session.models import Session
    from eve.user import User


logger = logging.getLogger(__name__)


class MemoryService:
    """Facade coordinating access to a concrete memory backend."""

    def __init__(self, backend: Optional[MemoryBackend] = None) -> None:
        self._backend: MemoryBackend = backend or MongoMemoryBackend()
        self._backend_factories: Dict[str, Callable[[], MemoryBackend]] = {}
        self._backends: Dict[str, MemoryBackend] = {}

    @property
    def backend(self) -> MemoryBackend:
        return self._backend

    def configure(self, backend: MemoryBackend) -> None:
        """Swap the default backend implementation."""
        self._backend = backend

    def register_backend(self, name: str, factory: Callable[[], MemoryBackend]) -> None:
        """Register an additional backend that can be selected per agent."""
        self._backend_factories[name] = factory

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
        backend = self._resolve_backend(agent=agent)
        return await backend.assemble_memory_context(
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
        backend = self._resolve_backend(agent=agent, agent_id=agent_id)
        return await backend.maybe_form_memories(agent_id, session, agent)

    async def form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
        *,
        conversation_text: Optional[str] = None,
        char_counts_by_source: Optional[Dict[str, Any]] = None,
    ) -> bool:
        backend = self._resolve_backend(agent=agent, agent_id=agent_id)
        return await backend.form_memories(
            agent_id,
            session,
            agent,
            conversation_text=conversation_text,
            char_counts_by_source=char_counts_by_source,
        )

    def _resolve_backend(
        self,
        *,
        agent: Optional["Agent"],
        agent_id: Optional[ObjectId] = None,
    ) -> MemoryBackend:
        backend_name = self._get_backend_name(agent)

        if backend_name is None and agent is None and agent_id is not None:
            agent = self._load_agent(agent_id)
            backend_name = self._get_backend_name(agent)

        if backend_name is None:
            return self._backend

        return self._get_named_backend(backend_name)

    def _get_backend_name(self, agent: Optional["Agent"]) -> Optional[str]:
        if not agent:
            return None
        extras = getattr(agent, "agent_extras", None)
        if not extras:
            return None
        return getattr(extras, "experimental_memory_backend", None)

    def _get_named_backend(self, name: str) -> MemoryBackend:
        backend = self._backends.get(name)
        if backend:
            return backend

        factory = self._backend_factories.get(name)
        if not factory:
            logger.warning(
                "Memory backend '%s' is not registered; falling back to default.", name
            )
            return self._backend

        try:
            backend = factory()
        except Exception:
            logger.exception(
                "Failed to instantiate memory backend '%s'; using default.", name
            )
            return self._backend

        self._backends[name] = backend
        return backend

    @staticmethod
    def _load_agent(agent_id: ObjectId) -> Optional["Agent"]:
        try:
            from eve.agent import Agent

            return Agent.from_mongo(agent_id)
        except Exception:
            logger.warning(
                "Unable to load agent %s while resolving memory backend override.",
                agent_id,
                exc_info=True,
            )
            return None


memory_service = MemoryService()

try:  # pragma: no cover - optional dependency
    from .backends import GraphitiMemoryBackend
except Exception:  # pragma: no cover - optional dependency
    GraphitiMemoryBackend = None  # type: ignore[assignment]

if GraphitiMemoryBackend is not None:  # pragma: no cover - optional dependency
    memory_service.register_backend("graphiti", lambda: GraphitiMemoryBackend())
