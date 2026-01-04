"""
Memory System v2 - Context Assembly

This module handles the assembly of always-in-context memory for prompt injection.
It builds an XML-formatted memory context from consolidated blobs and recent
unabsorbed reflections for all applicable scopes.

The assembled context is injected into every agent response, regardless of RAG.
"""

import traceback
from datetime import datetime, timezone
from typing import Dict, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.constants import LOCAL_DEV
from eve.agent.memory2.models import ConsolidatedMemory, get_unabsorbed_reflections


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is UTC-aware. Returns None if input is None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def assemble_always_in_context_memory(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> str:
    """
    Assemble the always-in-context memory for prompt injection.

    This gets injected into agent context on EVERY message, regardless of RAG.
    The final memory_xml can be cached inside the session object to avoid
    running this for every message.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional, for user-scope memory)
        session_id: Session ID (optional, for session-scope memory)

    Returns:
        XML-formatted memory context string
    """
    try:
        # Gather all memory components
        agent_blob, agent_recent = await _get_scope_memory(
            scope="agent",
            agent_id=agent_id,
        )

        user_blob, user_recent = None, None
        if user_id:
            user_blob, user_recent = await _get_scope_memory(
                scope="user",
                agent_id=agent_id,
                user_id=user_id,
            )

        session_blob, session_recent = None, None
        if session_id:
            session_blob, session_recent = await _get_scope_memory(
                scope="session",
                agent_id=agent_id,
                session_id=session_id,
            )

        # Build XML context
        memory_xml = _build_memory_xml(
            agent_blob=agent_blob,
            agent_recent=agent_recent,
            user_blob=user_blob,
            user_recent=user_recent,
            session_blob=session_blob,
            session_recent=session_recent,
        )

        if LOCAL_DEV:
            word_count = len(memory_xml.split()) if memory_xml else 0
            logger.debug(f"\n{'='*60}")
            logger.debug(f"MEMORY CONTEXT INJECTED INTO LLM ({word_count} words):")
            logger.debug(f"{'='*60}\n{memory_xml if memory_xml else '(empty)'}\n{'='*60}")

        return memory_xml

    except Exception as e:
        logger.error(f"Error assembling always-in-context memory: {e}")
        traceback.print_exc()
        return ""


async def _get_scope_memory(
    scope: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Get consolidated blob and recent reflections for a scope.

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


def _build_memory_xml(
    agent_blob: Optional[str],
    agent_recent: Optional[str],
    user_blob: Optional[str],
    user_recent: Optional[str],
    session_blob: Optional[str],
    session_recent: Optional[str],
) -> str:
    """
    Build XML-formatted memory context from components.

    Each scope section contains:
    1. Consolidated blob (if any) - summarized historical memory
    2. Recent reflections (if any) - not yet consolidated

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

    # Combine all sections
    if sections:
        return f"<MemoryContext>\n{chr(10).join(sections)}\n</MemoryContext>"

    return ""


async def get_memory_context_for_session(
    session,
    agent_id: ObjectId,
    last_speaker_id: Optional[ObjectId] = None,
    force_refresh: bool = False,
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

    Returns:
        Memory context XML string
    """
    try:
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
                    if LOCAL_DEV:
                        logger.debug(f"Using cached memory context (age: {age_seconds:.0f}s)")
                    return cached

        # Assemble fresh context
        memory_xml = await assemble_always_in_context_memory(
            agent_id=agent_id,
            user_id=last_speaker_id,
            session_id=session.id if hasattr(session, "id") else None,
        )

        # Update session cache
        memory_context.cached_memory_context = memory_xml
        memory_context.memory_context_timestamp = datetime.now(timezone.utc)

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
