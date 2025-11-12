from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest
from bson import ObjectId

from eve.agent.memory.backends import GraphitiMemoryBackend, MemoryBackend
from eve.agent.memory.service import MemoryService
from eve.agent.session.models import SessionMemoryContext


class DummyEpisode:
    def __init__(self, uuid: str, content: Any, *, ts: datetime | None = None) -> None:
        self.uuid = uuid
        self.content = content
        self.valid_at = ts or datetime.now(timezone.utc)
        self.created_at = self.valid_at


class DummyGraphiti:
    def __init__(self) -> None:
        self.added: List[Dict[str, Any]] = []
        self.namespaced_episodes: Dict[str, List[DummyEpisode]] = defaultdict(list)
        self.retrieve_calls: List[Dict[str, Any]] = []

    async def add_episode(self, **kwargs: Any) -> None:
        self.added.append(kwargs)
        ts = kwargs.get("reference_time", datetime.now(timezone.utc))
        episode = DummyEpisode(
            uuid=str(uuid4()),
            content=kwargs["episode_body"],
            ts=ts,
        )
        self.namespaced_episodes[kwargs["group_id"]].append(episode)

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int,
        group_ids: List[str],
        source: Any,
    ) -> List[DummyEpisode]:
        self.retrieve_calls.append(
            {
                "reference_time": reference_time,
                "last_n": last_n,
                "group_ids": list(group_ids),
                "source": source,
            }
        )
        namespace = group_ids[0]
        episodes = self.namespaced_episodes.get(namespace, [])
        return episodes[-last_n:]


class DummyMessage:
    def __init__(self, role: str, content: str, sender: ObjectId) -> None:
        self.role = role
        self.content = content
        self.sender = sender
        self.tool_calls: List[Any] = []
        self.trigger = None
        self.name = None
        self.id = ObjectId()


class DummySession:
    def __init__(self) -> None:
        self.id = ObjectId()
        self.memory_context = SessionMemoryContext()
        self.saved = False

    def save(self, **_: Any) -> None:
        self.saved = True

    def update(self, **_: Any) -> None:  # pragma: no cover - not used in tests
        pass


class DummyAgent:
    def __init__(self, agent_id: ObjectId, slug: str = "agent-slug") -> None:
        self.id = agent_id
        self.slug = slug
        self.user_memory_enabled = False
        self.agent_extras = None


class DummyUser:
    def __init__(self, user_id: ObjectId, username: str = "tester") -> None:
        self.canonical_user_id = user_id
        self.username = username


class RecordingBackend(MemoryBackend):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: List[str] = []

    async def assemble_memory_context(
        self,
        session: Any,
        agent: Any,
        user: Any,
        *,
        force_refresh: bool = False,
        reason: str = "unknown",
        skip_save: bool = False,
    ) -> str:
        self.calls.append(f"{self.name}:assemble")
        return self.name

    async def maybe_form_memories(
        self,
        agent_id: ObjectId,
        session: Any,
        agent: Optional[Any] = None,
    ) -> bool:
        self.calls.append(f"{self.name}:maybe")
        return True

    async def form_memories(
        self,
        agent_id: ObjectId,
        session: Any,
        agent: Optional[Any] = None,
        *,
        conversation_text: Optional[str] = None,
        char_counts_by_source: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self.calls.append(f"{self.name}:form")
        return True


@pytest.mark.asyncio
async def test_graphiti_backend_ingests_and_reconstructs(monkeypatch: pytest.MonkeyPatch):
    """Ensure Graphiti episodes are created from session slices and replayed into context."""

    dummy_graphiti = DummyGraphiti()
    backend = GraphitiMemoryBackend(graphiti=dummy_graphiti)

    session = DummySession()
    agent_id = ObjectId()
    agent = DummyAgent(agent_id)
    agent.user_memory_enabled = True
    user = DummyUser(ObjectId())

    conversation_text = "User: likes neon gradients\nAgent: acknowledged preference"

    # Prepare minimal recent message list so the backend will ingest.
    dummy_messages = [
        DummyMessage("user", "hello", user.canonical_user_id),
        DummyMessage("assistant", "hi", agent_id),
        DummyMessage("user", "I love neon gradients.", user.canonical_user_id),
        DummyMessage("assistant", "Noted.", agent_id),
    ]
    monkeypatch.setattr(
        "eve.agent.memory.backends.select_messages",
        lambda _session: dummy_messages,
    )

    shard_id = ObjectId()
    extracted_data = {
        "episode": ["Episode summary"],
        "directive": ["Always draw neon gradients."],
        "fact": ["Deadline is Aug 5 per Alex."],
        "suggestion": ["Experiment with 16:9 aspect ratio."],
    }
    memory_to_shard_map = {"fact_0": shard_id, "suggestion_0": shard_id}

    async def fake_extract_all_memories(*args: Any, **kwargs: Any):
        return extracted_data, memory_to_shard_map

    async def fake_save_all_memories(*args: Any, **kwargs: Any):
        fake_save_all_memories.called = True

    fake_save_all_memories.called = False

    async def fake_check_freshness(*_: Any, **__: Any) -> bool:
        return False

    def fake_safe_update(session_obj: DummySession, updates: Dict[str, Any], skip_save: bool = False) -> None:
        if not session_obj.memory_context:
            session_obj.memory_context = SessionMemoryContext()
        for key, value in updates.items():
            setattr(session_obj.memory_context, key, value)

    monkeypatch.setattr(
        "eve.agent.memory.backends.check_memory_freshness", fake_check_freshness
    )
    monkeypatch.setattr(
        "eve.agent.memory.backends.safe_update_memory_context", fake_safe_update
    )
    monkeypatch.setattr(
        "eve.agent.memory.backends._extract_related_users",
        lambda *args, **kwargs: [user.canonical_user_id],
    )
    monkeypatch.setattr(
        "eve.agent.memory.backends._extract_all_memories", fake_extract_all_memories
    )
    monkeypatch.setattr(
        "eve.agent.memory.backends._save_all_memories", fake_save_all_memories
    )
    monkeypatch.setattr(
        "eve.agent.memory.backends.AgentMemory.find",
        lambda query: [
            SimpleNamespace(
                id=shard_id,
                shard_name="creative_shard",
                content="Team agreed to keep visuals slow and meditative.",
                extraction_prompt="Collect art-direction facts.",
                is_active=True,
            )
        ],
    )

    # Force ingestion of an episode slice.
    await backend.form_memories(
        agent_id,
        session,
        agent,
        conversation_text=conversation_text,
        char_counts_by_source={"user": 0, "agent": 0, "tool": 0, "other": 0},
    )

    assert fake_save_all_memories.called

    stored_group_ids = {entry["group_id"] for entry in dummy_graphiti.added}
    assert f"session_{session.id}" in stored_group_ids
    assert any(group.startswith(f"user_{agent_id}") for group in stored_group_ids)
    assert any(group.startswith(f"shard_{agent_id}") for group in stored_group_ids)

    context = await backend.assemble_memory_context(session, agent, user, force_refresh=True)
    assert "Always draw neon gradients." in context
    assert "Deadline is Aug 5 per Alex." in context
    assert "Episode summary" in context


@pytest.mark.asyncio
async def test_memory_service_selects_experimental_backend():
    """Verify the memory service routes calls to a registered backend per agent."""

    default_backend = RecordingBackend("mongo")
    graphiti_backend = RecordingBackend("graphiti")

    service = MemoryService(backend=default_backend)
    service.register_backend("graphiti", lambda: graphiti_backend)

    session = DummySession()
    user = DummyUser(ObjectId())
    agent_with_override = DummyAgent(ObjectId())
    agent_with_override.agent_extras = SimpleNamespace(
        experimental_memory_backend="graphiti"
    )
    agent_default = DummyAgent(ObjectId())

    context = await service.assemble_memory_context(
        session, agent_with_override, user
    )
    assert context == "graphiti"
    assert graphiti_backend.calls == ["graphiti:assemble"]

    # Default agent should continue to hit the mongo backend.
    await service.maybe_form_memories(agent_default.id, session, agent_default)
    assert default_backend.calls == ["mongo:maybe"]
