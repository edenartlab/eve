"""
Memory System v2 - Context Assembly

This module handles the assembly of always-in-context memory for prompt injection.
It builds an XML-formatted memory context from consolidated blobs and recent
unabsorbed reflections for all applicable scopes.

The assembled context is injected into every agent response, regardless of RAG.
"""

import asyncio
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.constants import (
    LOCAL_DEV,
    FACTS_FIFO_ENABLED,
    FACTS_FIFO_LIMIT,
)
from eve.agent.memory2.models import (
    ConsolidatedMemory,
    get_unabsorbed_reflections,
    get_recent_facts_fifo,
)

if TYPE_CHECKING:
    from eve.agent.session.instrumentation import PromptSessionInstrumentation


def _log_debug(
    message: str,
    instrumentation: Optional["PromptSessionInstrumentation"] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Log debug message with optional instrumentation support."""
    if instrumentation:
        instrumentation.log_event(message, level="debug", payload=payload)
    else:
        if payload:
            logger.debug(f"{message} | {payload}")
        else:
            logger.debug(message)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is UTC-aware. Returns None if input is None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def _timed_get_scope_memory(
    scope: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> Tuple[str, Optional[str], Optional[str], float]:
    """
    Fetch scope memory with timing.

    Returns:
        Tuple of (scope_name, blob, recent, duration_seconds)
    """
    start = time.time()
    blob, recent = await _get_scope_memory(
        scope=scope,
        agent_id=agent_id,
        user_id=user_id,
        session_id=session_id,
    )
    duration = time.time() - start
    return scope, blob, recent, duration


async def assemble_always_in_context_memory(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    instrumentation: Optional["PromptSessionInstrumentation"] = None,
    print_context: bool = False,
) -> str:
    """
    Assemble the always-in-context memory for prompt injection.

    This gets injected into agent context on EVERY message, regardless of RAG.
    The final memory_xml can be cached inside the session object to avoid
    running this for every message.

    All memory scopes are fetched in parallel for optimal performance.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional, for user-scope memory)
        session_id: Session ID (optional, for session-scope memory)
        instrumentation: Optional instrumentation for timing logs

    Returns:
        XML-formatted memory context string
    """
    try:
        start_time = time.time()

        # Build list of coroutines to run in parallel
        tasks: List[asyncio.Task] = []

        # Always fetch agent scope
        tasks.append(
            asyncio.create_task(
                _timed_get_scope_memory(scope="agent", agent_id=agent_id)
            )
        )

        # Optionally fetch user scope
        if user_id:
            tasks.append(
                asyncio.create_task(
                    _timed_get_scope_memory(
                        scope="user", agent_id=agent_id, user_id=user_id
                    )
                )
            )

        # Optionally fetch session scope
        if session_id:
            tasks.append(
                asyncio.create_task(
                    _timed_get_scope_memory(
                        scope="session", agent_id=agent_id, session_id=session_id
                    )
                )
            )

        # Run all fetches in parallel
        results = await asyncio.gather(*tasks)

        # Process results and log timings
        agent_blob, agent_recent = None, None
        user_blob, user_recent = None, None
        session_blob, session_recent = None, None

        for scope, blob, recent, duration in results:
            if scope == "agent":
                agent_blob, agent_recent = blob, recent
            elif scope == "user":
                user_blob, user_recent = blob, recent
            elif scope == "session":
                session_blob, session_recent = blob, recent

            scope_label = scope.capitalize()
            _log_debug(
                f"   ⏱️  {scope_label} Memory Assembly",
                instrumentation,
                {
                    "duration_s": round(duration, 3),
                    "has_consolidated": blob is not None,
                    "has_recent": recent is not None,
                },
            )

        # TEMPORARY: Fetch facts via FIFO when enabled (to be replaced by RAG)
        # This retrieves the N most recent facts instead of semantic search.
        # When migrating to full RAG, replace this with RAG retrieval.
        facts_content = None
        if FACTS_FIFO_ENABLED:
            facts_start = time.time()
            facts = await asyncio.to_thread(
                get_recent_facts_fifo,
                agent_id,
                user_id,
                FACTS_FIFO_LIMIT,
            )
            if facts:
                # Format facts with scope indicator
                facts_lines = []
                for fact in facts:
                    scope_str = fact.scope[0] if isinstance(fact.scope, list) else fact.scope
                    facts_lines.append(f"- [{scope_str}] {fact.content}")
                facts_content = "\n".join(facts_lines)

            facts_duration = time.time() - facts_start
            _log_debug(
                "   ⏱️  Facts FIFO Retrieval",
                instrumentation,
                {
                    "duration_s": round(facts_duration, 3),
                    "fact_count": len(facts) if facts else 0,
                },
            )

        # Build XML context
        memory_xml = _build_memory_xml(
            agent_blob=agent_blob,
            agent_recent=agent_recent,
            user_blob=user_blob,
            user_recent=user_recent,
            session_blob=session_blob,
            session_recent=session_recent,
            facts_content=facts_content,
        )

        total_duration = time.time() - start_time
        word_count = len(memory_xml.split()) if memory_xml else 0
        _log_debug(
            "   ✓ Memory context assembled (parallel)",
            instrumentation,
            {
                "duration_s": round(total_duration, 3),
                "word_count": word_count,
                "scopes_fetched": len(tasks),
            },
        )

        if LOCAL_DEV and print_context:
            print(f"\n{'='*60}")
            print(f"MEMORY CONTEXT INJECTED INTO LLM ({word_count} words):")
            print(f"{'='*60}\n{memory_xml if memory_xml else '(empty)'}\n{'='*60}")

        return memory_xml

    except Exception as e:
        logger.error(f"Error assembling always-in-context memory: {e}")
        traceback.print_exc()
        return ""


def _get_scope_memory_sync(
    scope: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Synchronous helper to get consolidated blob and recent reflections for a scope.
    This is run in a thread pool to enable true parallelism.

    Returns:
        Tuple of (consolidated_blob, recent_reflections_formatted)
    """
    try:
        # Build query
        query = {
            "scope_type": scope,
            "agent_id": agent_id,
        }
        if scope == "user" and user_id:
            query["user_id"] = user_id
        if scope == "session" and session_id:
            query["session_id"] = session_id

        # Get consolidated memory
        consolidated = ConsolidatedMemory.find_one(query)
        blob = consolidated.consolidated_content if consolidated else None

        # Get recent unabsorbed reflections
        reflections = get_unabsorbed_reflections(
            scope=scope,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            limit=10,  # Limit recent reflections to avoid context bloat
        )

        recent = None
        if reflections:
            recent = "\n".join(f"- {r.content}" for r in reflections)

        return blob, recent

    except Exception as e:
        logger.error(f"Error getting {scope} memory: {e}")
        return None, None


async def _get_scope_memory(
    scope: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get consolidated blob and recent reflections for a scope.
    Uses asyncio.to_thread to run synchronous MongoDB operations in parallel.

    Returns:
        Tuple of (consolidated_blob, recent_reflections_formatted)
    """
    return await asyncio.to_thread(
        _get_scope_memory_sync,
        scope,
        agent_id,
        user_id,
        session_id,
    )


def _build_memory_xml(
    agent_blob: Optional[str],
    agent_recent: Optional[str],
    user_blob: Optional[str],
    user_recent: Optional[str],
    session_blob: Optional[str],
    session_recent: Optional[str],
    facts_content: Optional[str] = None,
) -> str:
    """
    Build XML-formatted memory context from components.

    Each scope section contains:
    1. Consolidated blob (if any) - summarized historical memory
    2. Recent reflections (if any) - not yet consolidated

    The Facts section (when FIFO mode enabled) contains recent facts
    retrieved via simple FIFO query. This is temporary until full RAG
    is implemented.

    The XML structure provides clear separation between memory scopes
    and makes it easy for the agent to understand and utilize the context.
    """
    sections = []

    # Agent memory section (consolidated first, then recent reflections)
    agent_parts = []
    if agent_blob:
        agent_parts.append(f"<Consolidated>\n{agent_blob}\n</Consolidated>")
    if agent_recent:
        agent_parts.append(f"<RecentReflections>\n{agent_recent}\n</RecentReflections>")

    if agent_parts:
        sections.append(
            f"<AgentMemory>\n{chr(10).join(agent_parts)}\n</AgentMemory>"
        )

    # User memory section (consolidated first, then recent reflections)
    user_parts = []
    if user_blob:
        user_parts.append(f"<Consolidated>\n{user_blob}\n</Consolidated>")
    if user_recent:
        user_parts.append(f"<RecentReflections>\n{user_recent}\n</RecentReflections>")

    if user_parts:
        sections.append(
            f"<UserMemory>\n{chr(10).join(user_parts)}\n</UserMemory>"
        )

    # Session memory section (consolidated first, then recent reflections)
    session_parts = []
    if session_blob:
        session_parts.append(f"<Consolidated>\n{session_blob}\n</Consolidated>")
    if session_recent:
        session_parts.append(f"<RecentReflections>\n{session_recent}\n</RecentReflections>")

    if session_parts:
        sections.append(
            f"<SessionMemory>\n{chr(10).join(session_parts)}\n</SessionMemory>"
        )

    # TEMPORARY: Facts section (FIFO mode - to be replaced by RAG)
    # This section contains recent facts retrieved via simple FIFO query.
    # When RAG is enabled, facts will be retrieved via semantic search instead.
    if facts_content:
        sections.append(
            f"<Facts>\n{facts_content}\n</Facts>"
        )

    # Combine all sections
    if sections:
        return f"<MemoryContext>\n{chr(10).join(sections)}\n</MemoryContext>"

    return ""


async def get_memory_context_for_session(
    session,
    agent_id: ObjectId,
    last_speaker_id: Optional[ObjectId] = None,
    force_refresh: bool = False,
    instrumentation: Optional["PromptSessionInstrumentation"] = None,
) -> str:
    """
    Get memory context for a session, with caching support.

    This is the main entry point for getting memory context in the agent loop.
    It checks the session's cached memory context and refreshes if needed.

    Args:
        session: Session object (must have memory_context attribute)
        agent_id: Agent ID
        last_speaker_id: ID of the last user who spoke (for user-scope)
        force_refresh: Force regeneration of memory context
        instrumentation: Optional instrumentation for timing logs

    Returns:
        Memory context XML string
    """
    try:
        start_time = time.time()

        # Ensure memory_context is an object (not dict)
        memory_context = _ensure_memory_context_object(session)

        # Check if we have a cached context
        if not force_refresh:
            cached = getattr(memory_context, "cached_memory_context", None)
            timestamp = getattr(memory_context, "memory_context_timestamp", None)

            # Check if cache is still fresh (5 minute TTL)
            if cached and timestamp:
                timestamp = ensure_utc(timestamp)
                age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
                if age_seconds < 300:  # 5 minutes
                    _log_debug(
                        "   ✓ Memory context cache hit",
                        instrumentation,
                        {
                            "cache_age_s": round(age_seconds, 1),
                            "word_count": len(cached.split()) if cached else 0,
                        },
                    )
                    return cached

        _log_debug(
            "   🔄 Rebuilding memory context",
            instrumentation,
            {"reason": "force_refresh" if force_refresh else "cache_miss_or_expired"},
        )

        # Assemble fresh context
        memory_xml = await assemble_always_in_context_memory(
            agent_id=agent_id,
            user_id=last_speaker_id,
            session_id=session.id if hasattr(session, "id") else None,
            instrumentation=instrumentation,
            print_context=True,  # Print when replying to user
        )

        # Update session cache
        memory_context.cached_memory_context = memory_xml
        memory_context.memory_context_timestamp = datetime.now(timezone.utc)

        total_duration = time.time() - start_time
        _log_debug(
            "   ✓ Memory context rebuilt and cached",
            instrumentation,
            {"duration_s": round(total_duration, 3)},
        )

        return memory_xml

    except Exception as e:
        logger.error(f"Error getting memory context for session: {e}")
        traceback.print_exc()
        return ""


def get_memory_stats(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> Dict[str, dict]:
    """
    Get statistics about the memory system for a given context.

    Useful for debugging and monitoring.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional)
        session_id: Session ID (optional)

    Returns:
        Dict with stats for each scope
    """
    stats = {}

    for scope in ["agent", "user", "session"]:
        # Skip scopes without required IDs
        if scope == "user" and not user_id:
            continue
        if scope == "session" and not session_id:
            continue

        # Build query
        query = {
            "scope_type": scope,
            "agent_id": agent_id,
        }
        if scope == "user":
            query["user_id"] = user_id
        if scope == "session":
            query["session_id"] = session_id

        try:
            consolidated = ConsolidatedMemory.find_one(query)

            reflections = get_unabsorbed_reflections(
                scope=scope,
                agent_id=agent_id,
                user_id=user_id if scope == "user" else None,
                session_id=session_id if scope == "session" else None,
                limit=1000,  # Get all for counting
            )

            stats[scope] = {
                "has_consolidated": consolidated is not None,
                "consolidated_word_count": (
                    len(consolidated.consolidated_content.split())
                    if consolidated and consolidated.consolidated_content
                    else 0
                ),
                "unabsorbed_count": len(reflections),
                "last_consolidated_at": (
                    consolidated.last_consolidated_at.isoformat()
                    if consolidated and consolidated.last_consolidated_at
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error getting stats for {scope}: {e}")
            stats[scope] = {"error": str(e)}

    return stats


def _ensure_memory_context_object(session):
    """
    Ensure session.memory_context is a SessionMemoryContext object, not a dict.
    """
    from eve.agent.session.models import SessionMemoryContext

    if not hasattr(session, "memory_context") or session.memory_context is None:
        session.memory_context = SessionMemoryContext()
    elif isinstance(session.memory_context, dict):
        session.memory_context = SessionMemoryContext(**session.memory_context)

    return session.memory_context


async def clear_memory_cache(session) -> None:
    """
    Clear the cached memory context for a session.

    Call this when you know the memory has changed and you want
    the next context assembly to be fresh.

    Args:
        session: Session object
    """
    try:
        memory_context = _ensure_memory_context_object(session)
        memory_context.cached_memory_context = None
        memory_context.memory_context_timestamp = None

        if LOCAL_DEV:
            logger.debug("Cleared memory context cache")

    except Exception as e:
        logger.error(f"Error clearing memory cache: {e}")
