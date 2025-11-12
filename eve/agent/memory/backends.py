"""Memory backend abstractions and concrete implementations."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict
from uuid import uuid4

from bson import ObjectId

from .graphiti import init_graphiti
from .memory import (
    _extract_related_users,
    safe_update_memory_context,
    should_form_memories,
    _extract_all_memories,
    _save_all_memories,
)
from .memory_constants import (
    LOCAL_DEV,
    MAX_N_EPISODES_TO_REMEMBER,
    NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES,
    SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES,
)
from .memory_models import (
    AgentMemory,
    _get_recent_messages,
    messages_to_text,
    select_messages,
)
from .memory_assemble_context import (
    _build_memory_xml,
    check_memory_freshness,
    _assemble_user_memory as _mongo_assemble_user_memory,
    _assemble_agent_memories as _mongo_assemble_agent_memories,
)

try:  # pragma: no cover - optional dependency
    from graphiti_core.nodes import EpisodeType
except ImportError:  # pragma: no cover - optional dependency
    EpisodeType = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from graphiti_core import Graphiti as GraphitiClient
    from eve.agent import Agent
    from eve.agent.session.models import Session
    from eve.user import User
else:  # pragma: no cover
    GraphitiClient = Any


logger = logging.getLogger(__name__)


class _GraphMemoryRecord(TypedDict, total=False):
    uuid: str
    memory_type: str
    content: str
    metadata: Dict[str, Any]
    created_at: Optional[datetime]


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
    """Graphiti-backed implementation with namespaced episodic storage."""

    def __init__(
        self,
        graphiti: Optional["GraphitiClient"] = None,
        *,
        episode_limit: int = MAX_N_EPISODES_TO_REMEMBER,
        episode_type: Optional["EpisodeType"] = None,
    ) -> None:
        if EpisodeType is None:
            raise ImportError(
                "graphiti_core is not available. Install it to enable Graphiti-backed memory."
            )

        self._graphiti: "GraphitiClient" = graphiti or init_graphiti()
        self._episode_limit = episode_limit
        self._episode_type: EpisodeType = episode_type or EpisodeType.message

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
        safe_update_memory_context(session, {})

        cached_context = session.memory_context.cached_memory_context
        if (
            not force_refresh
            and cached_context
            and await self._is_cache_fresh(session, agent, user)
        ):
            if LOCAL_DEV:
                logger.debug(
                    "Graphiti memory: using cached context for session %s", session.id
                )
            return cached_context

        user_memory_content = await self._build_user_memory_from_graph(agent, user)
        logger.debug(f"========== User memory: {user_memory_content}")
        if not user_memory_content:
            logger.debug("========== No user memory found, skipping")
            user_memory_content = ""

        agent_collective_memories = await self._build_agent_memories_from_graph(agent)
        logger.debug(f"========== Agent memory: {agent_collective_memories}")
        if not agent_collective_memories:
            logger.debug("========== No agent collective memories found, skipping")
            agent_collective_memories = []

        episode_memories = await self._load_graphiti_episode_memories(session)

        memory_context = _build_memory_xml(
            user_memory_content, agent_collective_memories, episode_memories
        )

        now = datetime.now(timezone.utc)
        safe_update_memory_context(
            session,
            {
                "cached_memory_context": memory_context,
                "cached_episode_memories": episode_memories,
                "memory_context_timestamp": now,
                "agent_memory_timestamp": now,
                "user_memory_timestamp": now,
            },
            skip_save=skip_save,
        )

        if not skip_save:
            session.save()

        return memory_context

    async def maybe_form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
    ) -> bool:
        should_run, conversation_text, char_counts = should_form_memories(
            agent_id, session
        )
        if not should_run:
            return False
        return await self.form_memories(
            agent_id,
            session,
            agent,
            conversation_text=conversation_text,
            char_counts_by_source=char_counts,
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
        session_messages = select_messages(session)
        if not session_messages:
            return False

        last_message_id = session_messages[-1].id

        safe_update_memory_context(session, {})
        old_last_memory_message_id = session.memory_context.last_memory_message_id
        recent_messages = _get_recent_messages(
            session_messages, old_last_memory_message_id
        )

        if (
            not recent_messages
            or len(recent_messages) < NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES
        ):
            return False

        safe_update_memory_context(
            session,
            {
                "last_memory_message_id": last_message_id,
                "last_activity": datetime.now(timezone.utc),
                "messages_since_memory_formation": 0,
            },
            skip_save=False,
        )

        should_reset_counters = True
        reset_applied = False

        try:
            if conversation_text is None:
                conversation_text, char_counts_by_source = messages_to_text(
                    recent_messages
                )

            extracted_data, memory_to_shard_map = await _extract_all_memories(
                agent_id,
                conversation_text,
                session,
                agent=agent,
                char_counts_by_source=char_counts_by_source or {},
            )

            await self._persist_graphiti_memories(
                agent_id=agent_id,
                session=session,
                agent=agent,
                recent_messages=recent_messages,
                extracted_data=extracted_data,
                memory_to_shard_map=memory_to_shard_map,
            )

            await _save_all_memories(
                agent_id,
                extracted_data,
                recent_messages,
                session,
                memory_to_shard_map,
            )

            related_users = _extract_related_users(
                session_messages, agent_id, user_only=True
            )
            last_speaker = related_users[-1] if related_users else None

            await self.assemble_memory_context(
                session,
                agent,
                last_speaker,  # type: ignore[arg-type]
                force_refresh=True,
                reason="form_memories",
                skip_save=True,
            )

            reset_applied = True
            return True
        except Exception:
            logger.exception(
                "Graphiti memory ingestion failed for session %s (agent %s)",
                session.id,
                agent_id,
            )
            return False
        finally:
            if should_reset_counters and not reset_applied and last_message_id:
                safe_update_memory_context(
                    session,
                    {
                        "last_activity": datetime.now(timezone.utc),
                        "last_memory_message_id": last_message_id,
                        "messages_since_memory_formation": 0,
                    },
                )

    async def _persist_graphiti_memories(
        self,
        *,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"],
        recent_messages: List[Any],
        extracted_data: Dict[str, List[str]],
        memory_to_shard_map: Dict[str, ObjectId],
    ) -> None:
        if not extracted_data:
            return

        timestamp = datetime.now(timezone.utc)
        message_ids = [str(msg.id) for msg in recent_messages]
        related_users = _extract_related_users(
            recent_messages, agent_id, user_only=True
        )
        tasks = []

        for idx, content in enumerate(extracted_data.get("episode", [])):
            metadata = {
                "session_id": str(session.id),
                "agent_id": str(agent_id),
                "message_ids": message_ids,
                "sequence": idx,
            }
            tasks.append(
                self._graphiti.add_episode(
                    name=f"session:{session.id}:episode:{timestamp.isoformat()}:{idx}",
                    episode_body=content,
                    source_description=json.dumps(metadata),
                    reference_time=timestamp,
                    source=EpisodeType.message,
                    group_id=self._session_namespace(session),
                )
            )

        for idx, content in enumerate(extracted_data.get("directive", [])):
            for user_id in related_users or []:
                payload = {
                    "memory_type": "directive",
                    "content": content,
                    "metadata": {
                        "agent_id": str(agent_id),
                        "user_id": str(user_id),
                        "session_id": str(session.id),
                        "message_ids": message_ids,
                        "sequence": idx,
                    },
                }
                tasks.append(
                    self._graphiti.add_episode(
                        name=f"user:{agent_id}:{user_id}:directive:{uuid4()}",
                        episode_body=json.dumps(payload),
                        source_description="eden.user.directive",
                        reference_time=timestamp,
                        source=EpisodeType.json,
                        group_id=self._user_namespace(agent_id, user_id),
                    )
                )

        for memory_type in ("fact", "suggestion"):
            for idx, content in enumerate(extracted_data.get(memory_type, [])):
                shard_id = memory_to_shard_map.get(f"{memory_type}_{idx}")
                if not shard_id:
                    continue
                payload = {
                    "memory_type": memory_type,
                    "content": content,
                    "metadata": {
                        "agent_id": str(agent_id),
                        "shard_id": str(shard_id),
                        "session_id": str(session.id),
                        "message_ids": message_ids,
                        "sequence": idx,
                    },
                }
                tasks.append(
                    self._graphiti.add_episode(
                        name=f"shard:{agent_id}:{shard_id}:{memory_type}:{uuid4()}",
                        episode_body=json.dumps(payload),
                        source_description=f"eden.shard.{memory_type}",
                        reference_time=timestamp,
                        source=EpisodeType.json,
                        group_id=self._shard_namespace(agent_id, shard_id),
                    )
                )

        if tasks:
            await asyncio.gather(*tasks)

    def _session_namespace(self, session: "Session") -> str:
        return self._normalize_namespace(f"session_{session.id}")

    def _user_namespace(self, agent_id: ObjectId, user_id: ObjectId) -> str:
        return self._normalize_namespace(f"user_{agent_id}_{user_id}")

    def _shard_namespace(self, agent_id: ObjectId, shard_id: ObjectId) -> str:
        return self._normalize_namespace(f"shard_{agent_id}_{shard_id}")

    @staticmethod
    def _normalize_namespace(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)

    async def _is_cache_fresh(
        self,
        session: "Session",
        agent: "Agent",
        user: "User",
    ) -> bool:
        cached_timestamp = session.memory_context.memory_context_timestamp
        if not cached_timestamp:
            return False

        if cached_timestamp.tzinfo is None:
            cached_timestamp = cached_timestamp.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - cached_timestamp
        if age.total_seconds() >= SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES * 60:
            return False

        return await check_memory_freshness(session, agent, user)

    async def _load_graphiti_episode_memories(
        self, session: "Session"
    ) -> list[Dict[str, Any]]:
        reference_time = datetime.now(timezone.utc)
        group_id = self._session_namespace(session)

        episodes = await self._graphiti.retrieve_episodes(
            reference_time=reference_time,
            last_n=self._episode_limit,
            group_ids=[group_id],
            source=self._episode_type,
        )

        formatted: list[Dict[str, Any]] = []
        for episode in episodes:
            payload = self._decode_episode_payload(episode.content)
            summary = payload.get("content") or self._summarize_episode_content(
                episode.content
                if isinstance(episode.content, str)
                else json.dumps(episode.content),
            )
            formatted.append(
                {
                    "id": episode.uuid,
                    "content": summary,
                    "created_at": episode.valid_at.isoformat()
                    if episode.valid_at
                    else None,
                }
            )
        return formatted

    async def _fetch_structured_memories(
        self,
        *,
        group_id: str,
        limit: int = 25,
    ) -> List[_GraphMemoryRecord]:
        episodes = await self._graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            last_n=limit,
            group_ids=[group_id],
            source=None,
        )

        records: List[_GraphMemoryRecord] = []
        for episode in episodes:
            payload = self._decode_episode_payload(episode.content)
            if not payload:
                continue
            records.append(
                _GraphMemoryRecord(
                    uuid=episode.uuid,
                    memory_type=payload.get("memory_type", "unknown"),
                    content=payload.get("content", ""),
                    metadata=payload.get("metadata", {}),
                    created_at=episode.valid_at or episode.created_at,
                )
            )
        return records

    @staticmethod
    def _decode_episode_payload(content: Any) -> Dict[str, Any]:
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                decoded = json.loads(content)
                if isinstance(decoded, dict):
                    return decoded
            except json.JSONDecodeError:
                return {"memory_type": "episode", "content": content, "metadata": {}}
        return {}

    @staticmethod
    def _summarize_episode_content(content: str, *, max_chars: int = 800) -> str:
        content = (content or "").strip()
        if len(content) <= max_chars:
            return content
        return content[: max_chars - 3].rstrip() + "..."

    def _format_memory_lines(self, records: List[_GraphMemoryRecord]) -> str:
        lines = []
        now = datetime.now(timezone.utc)
        for record in records:
            created_at = record.get("created_at") or now
            age_days = (now - created_at).days
            lines.append(
                f"- {record.get('content', '').strip()} (age: {age_days} days ago)"
            )
        return "\n".join(lines)

    async def _build_user_memory_from_graph(self, agent: "Agent", user: "User") -> str:
        if not getattr(agent, "user_memory_enabled", False):
            return ""

        canonical_user_id = getattr(user, "canonical_user_id", None)
        if not canonical_user_id:
            return ""

        namespace = self._user_namespace(agent.id, canonical_user_id)
        records = await self._fetch_structured_memories(group_id=namespace, limit=50)
        directives = [r for r in records if r.get("memory_type") == "directive"]

        if not directives:
            return ""

        username = getattr(user, "username", "unknown user")
        parts = [f"-- User Memory for {username} --"]
        directive_lines = self._format_memory_lines(directives)
        if directive_lines:
            parts.append(f"## Recent user directives:\n\n{directive_lines}")
        if LOCAL_DEV:
            logger.debug(f"========== User memory: {parts}")
        return "\n\n".join(parts)

    async def _build_agent_memories_from_graph(
        self, agent: "Agent"
    ) -> List[Dict[str, str]]:
        shards = AgentMemory.find({"agent_id": agent.id, "is_active": True})
        shard_memories: List[Dict[str, str]] = []
        for shard in shards:
            namespace = self._shard_namespace(agent.id, shard.id)
            records = await self._fetch_structured_memories(
                group_id=namespace, limit=50
            )
            if not records:
                continue

            facts = [r for r in records if r.get("memory_type") == "fact"]
            suggestions = [r for r in records if r.get("memory_type") == "suggestion"]

            parts: List[str] = []
            if facts:
                parts.append(f"## Shard facts:\n\n{self._format_memory_lines(facts)}")
            if shard.content:
                parts.append(
                    f"## Current consolidated shard memory:\n\n{shard.content}"
                )
            if suggestions:
                suggestion_lines = "\n".join(
                    [f"- {record.get('content', '').strip()}" for record in suggestions]
                )
                parts.append(f"## Recent shard suggestions:\n\n{suggestion_lines}")

            if parts:
                shard_memories.append(
                    {
                        "name": shard.shard_name or "unnamed_shard",
                        "content": "\n\n".join(parts),
                    }
                )

        if LOCAL_DEV:
            logger.debug(f"========== Agent memory: {shard_memories}")
        return shard_memories
