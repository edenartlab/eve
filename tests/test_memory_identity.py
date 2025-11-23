from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from bson import ObjectId

from eve.agent.memory import memory as memory_module
from eve.agent.memory import memory_assemble_context
from eve.agent.memory.memory_models import SessionMemory, UserMemory
from eve.agent.session.models import ChatMessage
from eve.user import User


class DummySession:
    def __init__(self, session_id: ObjectId):
        self.id = session_id
        self.memory_context = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def create_user(
    username: str, *, user_id: str | None = None, canonical: ObjectId | None = None
) -> User:
    user = User(username=username, userId=user_id)
    if canonical:
        user.eden_user_id = canonical
    user.save()
    return user


def test_user_save_sets_canonical_id():
    user = User(username="test-user", userId="test-user-id")
    user.save()
    assert user.eden_user_id == user.id


def test_canonical_id_map_handles_linked_users():
    primary = create_user("primary", user_id="primary-user")
    alias = create_user("alias", canonical=primary.id)

    mapping = User.get_canonical_id_map([primary.id, alias.id])
    assert mapping[primary.id] == primary.id
    assert mapping[alias.id] == primary.id


@pytest.mark.asyncio
async def test_session_memories_use_canonical_user_ids(monkeypatch):
    primary = create_user("primary", user_id="primary-user")
    alias = create_user("alias", canonical=primary.id)
    agent_id = ObjectId()

    user_message = ChatMessage(
        role="user",
        sender=alias.id,
        content="hello there",
        session=ObjectId(),
    )
    user_message.id = ObjectId()
    assistant_message = ChatMessage(
        role="assistant",
        sender=agent_id,
        content="hi!",
        session=user_message.session,
    )
    assistant_message.id = ObjectId()
    session = DummySession(user_message.session)

    captured_user_ids: list[ObjectId] = []

    async def fake_update(agent_id_arg, user_id_arg, new_directives):
        captured_user_ids.append(user_id_arg)

    monkeypatch.setattr(memory_module, "_update_user_memory", fake_update)
    monkeypatch.setattr(memory_module, "get_agent_owner", lambda _agent_id: None)

    await memory_module._save_all_memories(
        agent_id,
        {"directive": ["remember this user"]},
        [user_message, assistant_message],
        session,
        {},
    )

    stored_memory = SessionMemory.get_collection().find_one({})
    assert stored_memory is not None
    assert stored_memory["related_users"] == [primary.id]
    assert captured_user_ids == [primary.id]


@pytest.mark.asyncio
async def test_assemble_user_memory_targets_canonical_user(monkeypatch):
    agent = SimpleNamespace(id=ObjectId(), user_memory_enabled=True)
    primary = create_user("primary", user_id="primary-user")
    alias = create_user("alias", canonical=primary.id)

    async def fake_regenerate(user_memory):
        user_memory.fully_formed_memory = user_memory.content or ""
        user_memory.last_updated_at = datetime.now(timezone.utc)
        user_memory.save()

    monkeypatch.setattr(
        memory_module,
        "_regenerate_fully_formed_user_memory",
        fake_regenerate,
    )

    content = await memory_assemble_context._assemble_user_memory(agent, alias)
    assert content == ""

    docs = list(UserMemory.get_collection().find({}))
    assert len(docs) == 1
    assert docs[0]["user_id"] == primary.id
