import asyncio
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from eve.agent import Agent
from eve.agent.memory.memory import safe_update_memory_context
from eve.agent.memory.memory_constants import (
    MAX_N_EPISODES_TO_REMEMBER,
    SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES,
)
from eve.agent.memory.memory_models import (
    AgentMemory,
    SessionMemory,
    UserMemory,
    select_messages,
)
from eve.agent.session.models import Session
from eve.user import User

if TYPE_CHECKING:  # pragma: no cover
    from eve.agent.session_new.instrumentation import PromptSessionInstrumentation
from contextlib import nullcontext


def _log_debug(
    message: str, instrumentation=None, payload: Optional[Dict[str, Any]] = None
):
    if instrumentation:
        instrumentation.log_event(message, level="debug", payload=payload)
    else:
        if payload:
            logger.debug(f"{message} | {payload}")
        else:
            logger.debug(message)


async def _assemble_user_memory(agent: Agent, user: User, instrumentation=None) -> str:
    """
    Step 1: Assemble user memory content.
    Returns user_memory_content
    """
    user_memory_content = ""
    user_memory = None

    try:
        if agent is None or not agent.user_memory_enabled:
            return ""
        query_start = time.time()
        user_memory = UserMemory.find_one_or_create(
            {"agent_id": agent.id, "user_id": user.canonical_user_id}
        )
        if user_memory:
            # Check if fully_formed_memory exists and is up-to-date
            if user_memory.fully_formed_memory is not None:
                user_memory_content = user_memory.fully_formed_memory
            else:
                # Regenerate fully formed memory if missing or empty
                from eve.agent.memory.memory import (
                    _regenerate_fully_formed_user_memory,
                )

                await _regenerate_fully_formed_user_memory(user_memory)
                user_memory_content = user_memory.fully_formed_memory or ""

        query_time = time.time() - query_start
        _log_debug(
            "   ⏱️  User Memory Assembly",
            instrumentation,
            {
                "duration_s": round(query_time, 3),
                "has_memory": bool(user_memory),
            },
        )

    except Exception as e:
        logger.error(f"   ❌ Error retrieving user memory: {e}")
        traceback.print_exc()

    return user_memory_content


async def _get_episode_memories(
    session: Session, force_refresh: bool = False, instrumentation=None
) -> List[Dict[str, Any]]:
    """
    Get episode memories with smart caching in session.
    Returns list of episode memory dicts.
    """
    # Check if we have cached episodes:
    safe_update_memory_context(session, {})  # Ensure memory_context exists
    if not force_refresh and session.memory_context.cached_episode_memories is not None:
        return session.memory_context.cached_episode_memories

    # Query and cache episode memories
    try:
        query_start = time.time()
        episode_query = {"source_session_id": session.id, "memory_type": "episode"}

        # Optimized query: Use proper sort format and limit directly in MongoDB
        episode_memories = SessionMemory.find(
            episode_query, sort="createdAt", desc=True, limit=MAX_N_EPISODES_TO_REMEMBER
        )

        # Reverse to put most recent at bottom (after getting limited results)
        episode_memories = list(episode_memories)
        episode_memories.reverse()

        # Cache in session
        cached_episodes = [
            {
                "id": str(e.id),
                "content": e.content,
                "created_at": e.createdAt.isoformat() if e.createdAt else None,
            }
            for e in episode_memories
        ]

        # Run cache update as background task to avoid blocking
        asyncio.create_task(
            asyncio.to_thread(
                safe_update_memory_context,
                session,
                {
                    "cached_episode_memories": cached_episodes,
                    "episode_memories_timestamp": datetime.now(timezone.utc),
                },
            )
        )

        query_time = time.time() - query_start
        _log_debug(
            "   ⏱️  Episode memory query & cache",
            instrumentation,
            {
                "duration_s": round(query_time, 3),
                "episode_count": len(episode_memories),
            },
        )

        return cached_episodes

    except Exception as e:
        logger.error(f"   ❌ Error assembling episode memories: {e}")
        traceback.print_exc()
        return []


async def _assemble_agent_memories(
    agent: Agent, instrumentation=None
) -> List[Dict[str, str]]:
    """
    Step 3: Assemble agent collective memories from active shards.
    Returns list of memory shards with name and content.
    """
    agent_collective_memories = []

    if agent is None:
        return agent_collective_memories

    try:
        query_start = time.time()
        active_shards = AgentMemory.find({"agent_id": agent.id, "is_active": True})

        for shard in active_shards:
            # Check if fully_formed_memory exists and is non-empty
            if shard.fully_formed_memory is not None:
                agent_collective_memories.append(
                    {
                        "name": shard.shard_name or "unnamed_shard",
                        "content": shard.fully_formed_memory,
                    }
                )
            else:
                # Import here to avoid circular imports
                from eve.agent.memory.memory import (
                    _regenerate_fully_formed_agent_memory,
                )

                await _regenerate_fully_formed_agent_memory(shard)

                # Use the regenerated content if it's non-empty
                if shard.fully_formed_memory and shard.fully_formed_memory.strip():
                    agent_collective_memories.append(
                        {
                            "name": shard.shard_name or "unnamed_shard",
                            "content": shard.fully_formed_memory,
                        }
                    )

        query_time = time.time() - query_start
        _log_debug(
            "   ⏱️  Agent Memory Assembly",
            instrumentation,
            {
                "duration_s": round(query_time, 3),
                "active_shards": len(agent_collective_memories),
            },
        )

    except Exception as e:
        logger.error(f"   ❌ Error assembling agent collective memories: {e}")
        traceback.print_exc()

    return agent_collective_memories


async def check_memory_freshness(session: Session, agent: Agent, user: User) -> bool:
    """
    Check if cached memory context is still fresh by comparing timestamps.
    Returns True if cache is fresh, False if needs refresh.
    """
    safe_update_memory_context(session, {})  # Ensure memory_context exists
    # Check agent memory freshness
    if session.memory_context.agent_memory_timestamp:
        try:
            # Query only the most recently updated active agent memory
            agent_memories = AgentMemory.find(
                {"agent_id": str(agent.id), "is_active": True},
                sort="last_updated_at",
                desc=True,
                limit=1,
            )
            agent_memory = agent_memories[0] if agent_memories else None
            if agent_memory and agent_memory.last_updated_at:
                if (
                    agent_memory.last_updated_at
                    > session.memory_context.agent_memory_timestamp
                ):
                    return False
        except Exception:
            return False  # Refresh on error to be safe

    # Check user memory freshness
    if str(user.canonical_user_id) and session.memory_context.user_memory_timestamp:
        try:
            user_memory = UserMemory.find_one(
                {"agent_id": agent.id, "user_id": user.canonical_user_id}
            )
            if user_memory and user_memory.last_updated_at:
                if (
                    user_memory.last_updated_at
                    > session.memory_context.user_memory_timestamp
                ):
                    return False
        except Exception as e:
            logger.error(f"   ⚠️ Error checking user memory freshness: {e}")
            return False  # Refresh on error to be safe

    return True  # Cache is fresh


def _build_memory_xml(
    user_memory_content: str,
    agent_collective_memories: List[Dict[str, str]],
    episode_memories: List[Dict[str, Any]],
) -> str:
    """Build the formatted XML memory context."""

    memory_context = ""
    # Build collective memory section first
    collective_memory_section = ""
    if len(agent_collective_memories) > 0:
        collective_memory_section += '<CollectiveMemory description="Shared memory across all your conversations">\n'
        for shard in agent_collective_memories:
            shard_name = shard["name"]
            shard_content = shard["content"]
            collective_memory_section += f'<MemoryShard name="{shard_name}">\n{shard_content}\n</MemoryShard>\n\n'
        collective_memory_section += "</CollectiveMemory>\n\n"

    # Build user-specific memory section
    user_memory_section = ""
    if user_memory_content and user_memory_content.strip():
        user_memory_section += f'<UserMemory description="Memory and context specific to this user">\n{user_memory_content}\n</UserMemory>\n\n'

    # Build episode memories section
    episode_memory_section = ""
    if len(episode_memories) > 0:
        episode_memory_section += '    <CurrentConversationContext description="Recent exchanges from this conversation (most recent at bottom)">\n'
        for episode in episode_memories:
            episode_memory_section += f"     - {episode['content']}\n"
        episode_memory_section += "</CurrentConversationContext>\n\n"

    # Assemble final memory context with XML hierarchy
    if collective_memory_section or user_memory_section or episode_memory_section:
        memory_context = '  <MemoryContext description="Your complete memory context for this conversation">\n\n'

        if collective_memory_section:
            memory_context += collective_memory_section

        if user_memory_section:
            memory_context += user_memory_section

        if episode_memory_section:
            memory_context += episode_memory_section

        memory_context += "</MemoryContext>"

    return memory_context


async def assemble_memory_context(
    session: Session,
    agent: Agent,
    user: User,
    force_refresh: bool = False,
    reason: str = "unknown",
    skip_save: bool = False,
    instrumentation: Optional["PromptSessionInstrumentation"] = None,
) -> str:
    """
    Assemble relevant memories for context injection into prompts.
    Uses session-level caching to minimize database queries.

    Args:
        agent: Current Agent
        session: Current Session
        user: ID of the user who sent the last message

    Returns:
        Formatted memory context string for prompt injection.
    """
    stage_scope = (
        instrumentation.track_stage(
            "memory.assemble_context",
            level="info",
            metadata={"reason": reason, "force_refresh": force_refresh},
        )
        if instrumentation
        else nullcontext()
    )

    with stage_scope as stage:
        start_time = time.time()

        # Check if we can use cached memory context
        safe_update_memory_context(session, {})  # Ensure memory_context exists
        if (
            session.memory_context.cached_memory_context is not None
        ) and not force_refresh:
            memory_timestamp = session.memory_context.memory_context_timestamp
            if memory_timestamp and memory_timestamp.tzinfo is None:
                memory_timestamp = memory_timestamp.replace(tzinfo=timezone.utc)

            time_since_context_refresh = datetime.now(timezone.utc) - (
                memory_timestamp or datetime.min.replace(tzinfo=timezone.utc)
            )

            if time_since_context_refresh < timedelta(
                minutes=SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES
            ):
                # Optional: Check if memories were updated by other sessions
                if 1:  # or await check_memory_freshness(session, agent, user):
                    if instrumentation and hasattr(stage, "annotate"):
                        stage.annotate(cache_hit=True)
                    return session.memory_context.cached_memory_context
            else:
                reason = "syncing_session_memories"

        _log_debug(
            "   🔄 Rebuilding memory context",
            instrumentation,
            {"reason": reason},
        )

        # Rebuild memory context

        session_messages = select_messages(session)
        session_users = list(
            set([m.sender for m in session_messages if m.role == "user"])
        )
        max_n_user_memories_to_assemble = 4

        # 1. Get user memory (multiple queries if needed)
        if len(session_users) <= max_n_user_memories_to_assemble:
            # Assemble memories for all unique users in the session
            user_memory_contents = []

            users = User.find({"_id": {"$in": session_users}}) if session_users else []
            for user in users:
                user_content = await _assemble_user_memory(
                    agent, user, instrumentation=instrumentation
                )
                if user_content.strip():  # Only include non-empty memories
                    user_memory_contents.append(user_content)

            # Combine all user memories
            user_memory_content = (
                "\n\n".join(user_memory_contents) if user_memory_contents else ""
            )
        else:
            user_memory_content = ""

        # 2. Get agent memories (1 query)
    agent_collective_memories = await _assemble_agent_memories(
        agent, instrumentation=instrumentation
    )

    # 3. Get episode memories (0-1 queries with caching)
    episode_memories = await _get_episode_memories(
        session, force_refresh=force_refresh, instrumentation=instrumentation
    )

    # 4. Build XML context
    memory_context = _build_memory_xml(
        user_memory_content, agent_collective_memories, episode_memories
    )

    # 5. Update session with cached context
    current_time = datetime.now(timezone.utc)
    safe_update_memory_context(
        session,
        {
            "cached_memory_context": memory_context,
            "memory_context_timestamp": current_time,
            "agent_memory_timestamp": current_time,
            "user_memory_timestamp": current_time,
        },
        skip_save=skip_save,
    )

    if not skip_save:
        session.save()

    total_time = time.time() - start_time
    _log_debug(
        "   ✓ Memory context rebuilt and cached",
        instrumentation,
        {"duration_s": round(total_time, 3)},
    )

    if instrumentation and hasattr(stage, "annotate"):
        stage.annotate(cache_hit=False, duration_s=total_time)

    return memory_context
