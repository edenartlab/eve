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
    Memory2Config,
    is_multi_user_session,
)
from eve.agent.memory2.models import (
    ConsolidatedMemory,
    get_unabsorbed_reflections,
    get_recent_facts_fifo,
)

if TYPE_CHECKING:
    from eve.agent.session.instrumentation import PromptSessionInstrumentation


def _log_memory_event(
    message: str,
    instrumentation: Optional["PromptSessionInstrumentation"] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Log memory event with optional instrumentation support.

    Uses INFO level to ensure logs appear in production regardless of LOCAL_DEV.
    """
    if instrumentation:
        instrumentation.log_event(message, level="info", payload=payload)
    else:
        if payload:
            logger.info(f"{message} | {payload}")
        else:
            logger.info(message)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is UTC-aware. Returns None if input is None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def format_temporal_age(dt: datetime) -> str:
    """
    Format a datetime as a concise human-readable age suffix.

    Returns strings like: "4m ago", "2h ago", "3d ago", "1 week ago", "2 months ago"
    """
    dt = ensure_utc(dt)
    now = datetime.now(timezone.utc)
    delta = now - dt

    total_seconds = delta.total_seconds()

    # Less than 1 hour: show minutes
    if total_seconds < 3600:
        minutes = max(1, int(total_seconds / 60))
        return f"{minutes}m ago"

    # Less than 24 hours: show hours
    if total_seconds < 86400:
        hours = int(total_seconds / 3600)
        return f"{hours}h ago"

    # Less than 7 days: show days
    if total_seconds < 604800:
        days = int(total_seconds / 86400)
        return f"{days}d ago"

    # Less than 4 weeks: show weeks
    if total_seconds < 2419200:
        weeks = int(total_seconds / 604800)
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"

    # Less than 12 months: show months
    if total_seconds < 31536000:
        months = int(total_seconds / 2592000)
        return f"{months} month{'s' if months > 1 else ''} ago"

    # 1+ years
    years = int(total_seconds / 31536000)
    return f"{years} year{'s' if years > 1 else ''} ago"


def format_fact_with_age(fact) -> str:
    """
    Format a Fact's content with a temporal age suffix.

    Uses updated_at if the fact was edited, otherwise uses formed_at.

    Args:
        fact: A Fact object with content, formed_at, and optional updated_at fields

    Returns:
        Fact content with age suffix, e.g. "User likes coffee (3d ago)"
    """
    # Use updated_at if fact was edited, otherwise use formed_at
    timestamp = fact.updated_at if fact.updated_at else fact.formed_at
    age_suffix = format_temporal_age(timestamp)
    return f"{fact.content} ({age_suffix})"


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
    config: Optional[Memory2Config] = None,
) -> str:
    """
    Assemble the always-in-context memory for prompt injection.

    This gets injected into agent context on EVERY message, regardless of RAG.
    The final memory_xml can be cached inside the session object to avoid
    running this for every message.

    Only enabled scopes are fetched and assembled. Scopes are controlled by
    the Memory2Config (user_memory_enabled, agent_memory_enabled flags).
    Session memory is always included when any memory is enabled.

    All enabled memory scopes are fetched in parallel for optimal performance.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional, for user-scope memory)
        session_id: Session ID (optional, for session-scope memory)
        instrumentation: Optional instrumentation for timing logs
        print_context: Whether to print context in LOCAL_DEV mode
        config: Memory2Config with enabled scopes (loaded if not provided)

    Returns:
        XML-formatted memory context string
    """
    try:
        start_time = time.time()

        # Load config if not provided
        if config is None:
            config = Memory2Config.from_agent_id(agent_id)

        # Note: We always proceed because session memory is ALWAYS active.
        # config.reflection_scopes always includes "session" regardless of toggles.

        # Get enabled scopes for reflections (always includes session)
        enabled_scopes = config.reflection_scopes

        # Build list of coroutines to run in parallel (only for enabled scopes)
        tasks: List[asyncio.Task] = []

        # Fetch agent scope if enabled
        if "agent" in enabled_scopes:
            tasks.append(
                asyncio.create_task(
                    _timed_get_scope_memory(scope="agent", agent_id=agent_id)
                )
            )

        # Fetch user scope if enabled and user_id provided
        if "user" in enabled_scopes and user_id:
            tasks.append(
                asyncio.create_task(
                    _timed_get_scope_memory(
                        scope="user", agent_id=agent_id, user_id=user_id
                    )
                )
            )

        # Fetch session scope if enabled and session_id provided
        if "session" in enabled_scopes and session_id:
            tasks.append(
                asyncio.create_task(
                    _timed_get_scope_memory(
                        scope="session", agent_id=agent_id, session_id=session_id
                    )
                )
            )

        # Run all fetches in parallel
        results = await asyncio.gather(*tasks) if tasks else []

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
            _log_memory_event(
                f"   ⏱️  {scope_label} Memory Assembly",
                instrumentation,
                {
                    "duration_s": round(duration, 3),
                    "has_consolidated": blob is not None,
                    "has_recent": recent is not None,
                },
            )

        # TEMPORARY: Fetch facts via FIFO when enabled (to be replaced by RAG)
        # Only fetch facts if FIFO is enabled AND at least one fact scope is enabled
        facts_content = None
        fact_scopes = config.fact_scopes
        if FACTS_FIFO_ENABLED and fact_scopes:
            facts_start = time.time()
            # Pass enabled scopes to filter facts
            facts = await asyncio.to_thread(
                get_recent_facts_fifo,
                agent_id,
                user_id if "user" in fact_scopes else None,
                FACTS_FIFO_LIMIT,
                fact_scopes,  # Filter by enabled scopes
            )
            if facts:
                # Format facts with scope indicator and temporal age
                facts_lines = []
                for fact in facts:
                    scope_str = fact.scope[0] if isinstance(fact.scope, list) else fact.scope
                    fact_with_age = format_fact_with_age(fact)
                    facts_lines.append(f"- [{scope_str}] {fact_with_age}")
                facts_content = "\n".join(facts_lines)

            facts_duration = time.time() - facts_start
            _log_memory_event(
                "   ⏱️  Facts FIFO Retrieval",
                instrumentation,
                {
                    "duration_s": round(facts_duration, 3),
                    "fact_count": len(facts) if facts else 0,
                    "enabled_scopes": fact_scopes,
                },
            )

        # Build XML context (only include enabled scopes)
        memory_xml = _build_memory_xml(
            agent_blob=agent_blob if "agent" in enabled_scopes else None,
            agent_recent=agent_recent if "agent" in enabled_scopes else None,
            user_blob=user_blob if "user" in enabled_scopes else None,
            user_recent=user_recent if "user" in enabled_scopes else None,
            session_blob=session_blob if "session" in enabled_scopes else None,
            session_recent=session_recent if "session" in enabled_scopes else None,
            facts_content=facts_content,
        )

        total_duration = time.time() - start_time
        word_count = len(memory_xml.split()) if memory_xml else 0
        _log_memory_event(
            "   ✓ Memory context assembled (parallel)",
            instrumentation,
            {
                "duration_s": round(total_duration, 3),
                "word_count": word_count,
                "scopes_fetched": len(tasks),
                "enabled_scopes": enabled_scopes,
            },
        )

        if LOCAL_DEV and print_context:
            print(f"\n{'='*60}")
            print(f"MEMORY CONTEXT INJECTED INTO LLM ({word_count} words, scopes: {enabled_scopes}):")
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

    For multi-user sessions (group chats), user-specific memories are omitted
    entirely to prevent memory leakage between users. Only agent and session
    memories are included in such cases.

    Args:
        session: Session object (must have memory_context attribute)
        agent_id: Agent ID
        last_speaker_id: ID of the last user who spoke (for user-scope)
        force_refresh: Force regeneration of memory context
        instrumentation: Optional instrumentation for timing logs

    Returns:
        Memory context XML string (always includes session memory at minimum)
    """
    # Get memory2 configuration for this agent and session context
    # Multi-user sessions automatically disable user memory to prevent leakage
    config = Memory2Config.from_agent_id(agent_id, session=session)

    # Note: We always proceed because session memory is ALWAYS active.
    # config.reflection_scopes always includes "session" regardless of toggles.

    try:
        start_time = time.time()

        # Ensure memory_context is an object (not dict)
        memory_context = _ensure_memory_context_object(session)

        # For multi-user sessions, skip user memories entirely to prevent leakage.
        # The cache is stored at session level without user differentiation, so
        # including user-specific memories would leak one user's data to others.
        is_multi_user = is_multi_user_session(session)
        effective_user_id = None if is_multi_user else last_speaker_id

        if is_multi_user and LOCAL_DEV:
            session_users = getattr(session, "users", None) or []
            logger.debug(
                f"Multi-user session detected ({len(session_users)} users) - "
                "skipping user memories to prevent leakage"
            )

        # Check if we have a cached context
        if not force_refresh:
            cached = getattr(memory_context, "cached_memory_context", None)
            timestamp = getattr(memory_context, "memory_context_timestamp", None)

            # Check if cache is still fresh (5 minute TTL)
            if cached and timestamp:
                timestamp = ensure_utc(timestamp)
                age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
                if age_seconds < 300:  # 5 minutes
                    _log_memory_event(
                        "   ✓ Memory context cache hit",
                        instrumentation,
                        {
                            "cache_age_s": round(age_seconds, 1),
                            "word_count": len(cached.split()) if cached else 0,
                            "is_multi_user": is_multi_user,
                        },
                    )
                    return cached

        _log_memory_event(
            "   🔄 Rebuilding memory context",
            instrumentation,
            {
                "reason": "force_refresh" if force_refresh else "cache_miss_or_expired",
                "is_multi_user": is_multi_user,
                "enabled_scopes": config.reflection_scopes,
            },
        )

        # Assemble fresh context with config
        # For multi-user sessions, effective_user_id is None so user memories are skipped
        memory_xml = await assemble_always_in_context_memory(
            agent_id=agent_id,
            user_id=effective_user_id,
            session_id=session.id if hasattr(session, "id") else None,
            instrumentation=instrumentation,
            print_context=True,  # Print when replying to user
            config=config,
        )

        # Update session cache and persist to DB
        memory_context.cached_memory_context = memory_xml
        memory_context.memory_context_timestamp = datetime.now(timezone.utc)
        session.update(memory_context=memory_context.model_dump())
        _ensure_memory_context_object(session)  # Re-instantiate as model object

        total_duration = time.time() - start_time
        _log_memory_event(
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
