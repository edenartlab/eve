"""
Memory System v2 - Reflection Storage

This module handles the storage and retrieval of reflections from MongoDB.
It provides utilities for managing the reflection buffer and querying reflections.
"""

import traceback
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.constants import CONSOLIDATION_THRESHOLDS, LOCAL_DEV
from eve.agent.memory2.models import (
    ConsolidatedMemory,
    Reflection,
    get_unabsorbed_reflections,
)


async def save_reflections(
    reflections_data: Dict[str, List[str]],
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    source_session_id: Optional[ObjectId] = None,
    source_message_ids: Optional[List[ObjectId]] = None,
) -> Dict[str, List[Reflection]]:
    """
    Save extracted reflections to the database.

    Args:
        reflections_data: Dict with keys "agent", "user", "session" containing reflection strings
        agent_id: Agent ID
        user_id: User ID for user-scoped reflections
        session_id: Session ID for session-scoped reflections
        source_session_id: Session where reflections were formed
        source_message_ids: Messages that contributed to these reflections

    Returns:
        Dict mapping scope to list of saved Reflection documents
    """
    saved_by_scope = {"agent": [], "user": [], "session": []}

    try:
        for scope, contents in reflections_data.items():
            if not contents:
                continue

            reflections = []
            for content in contents:
                if not content or not content.strip():
                    continue

                # Determine scope-specific IDs
                reflection_user_id = user_id if scope == "user" else None
                reflection_session_id = session_id if scope == "session" else None

                reflection = Reflection(
                    content=content.strip(),
                    scope=scope,
                    agent_id=agent_id,
                    user_id=reflection_user_id,
                    session_id=reflection_session_id,
                    source_session_id=source_session_id or session_id,
                    source_message_ids=source_message_ids or [],
                )
                reflections.append(reflection)

            if reflections:
                # Batch save
                try:
                    Reflection.save_many(reflections)
                except Exception as e:
                    logger.error(f"Batch save failed for {scope} reflections: {e}")
                    for r in reflections:
                        r.save()

                saved_by_scope[scope] = reflections

                # Add to consolidated memory's unabsorbed list
                await _add_reflections_to_buffer(
                    scope=scope,
                    agent_id=agent_id,
                    user_id=reflection_user_id,
                    session_id=reflection_session_id,
                    reflection_ids=[r.id for r in reflections],
                )

        if LOCAL_DEV:
            total = sum(len(v) for v in saved_by_scope.values())
            logger.debug(f"Saved {total} reflections to database")

        return saved_by_scope

    except Exception as e:
        logger.error(f"Error saving reflections: {e}")
        traceback.print_exc()
        return saved_by_scope


async def _add_reflections_to_buffer(
    scope: Literal["agent", "user", "session"],
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    session_id: Optional[ObjectId],
    reflection_ids: List[ObjectId],
) -> None:
    """
    Add reflection IDs to the appropriate consolidated memory buffer.

    Uses atomic MongoDB operations to prevent race conditions.
    """
    if not reflection_ids:
        return

    try:
        from eve.agent.memory2.constants import CONSOLIDATED_WORD_LIMITS

        # Get or create consolidated memory for this scope
        consolidated = ConsolidatedMemory.get_or_create(
            scope_type=scope,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            word_limit=CONSOLIDATED_WORD_LIMITS.get(scope, 400),
        )

        # Atomic push to unabsorbed_ids
        collection = ConsolidatedMemory.get_collection()
        collection.update_one(
            {"_id": consolidated.id},
            {
                "$push": {"unabsorbed_ids": {"$each": reflection_ids}},
                "$currentDate": {"updatedAt": True},
            },
        )

        if LOCAL_DEV:
            logger.debug(
                f"Added {len(reflection_ids)} reflections to {scope} buffer"
            )

    except Exception as e:
        logger.error(f"Error adding reflections to {scope} buffer: {e}")
        traceback.print_exc()


def get_buffer_size(
    scope: Literal["agent", "user", "session"],
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> int:
    """
    Get the current size of the reflection buffer for a scope.

    Args:
        scope: Scope type
        agent_id: Agent ID
        user_id: User ID (for user scope)
        session_id: Session ID (for session scope)

    Returns:
        Number of unabsorbed reflections in the buffer
    """
    try:
        query = {
            "scope_type": scope,
            "agent_id": agent_id,
        }
        if scope == "user" and user_id:
            query["user_id"] = user_id
        if scope == "session" and session_id:
            query["session_id"] = session_id

        consolidated = ConsolidatedMemory.find_one(query)
        if consolidated:
            return len(consolidated.unabsorbed_ids)
        return 0

    except Exception as e:
        logger.error(f"Error getting buffer size: {e}")
        return 0


def should_consolidate(
    scope: Literal["agent", "user", "session"],
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> bool:
    """
    Check if consolidation should be triggered for a scope.

    Args:
        scope: Scope type
        agent_id: Agent ID
        user_id: User ID (for user scope)
        session_id: Session ID (for session scope)

    Returns:
        True if buffer size >= consolidation threshold
    """
    threshold = CONSOLIDATION_THRESHOLDS.get(scope, 5)
    buffer_size = get_buffer_size(scope, agent_id, user_id, session_id)
    return buffer_size >= threshold


async def get_all_reflections_for_context(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> Dict[str, Dict[str, any]]:
    """
    Get all reflections for context assembly.

    Returns consolidated blobs and unabsorbed reflections for all applicable scopes.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional, for user-scope)
        session_id: Session ID (optional, for session-scope)

    Returns:
        Dict with structure:
        {
            "agent": {
                "consolidated": str,
                "recent": List[str],
            },
            "user": {
                "consolidated": str,
                "recent": List[str],
            },
            "session": {
                "consolidated": str,
                "recent": List[str],
            },
        }
    """
    result = {
        "agent": {"consolidated": "", "recent": []},
        "user": {"consolidated": "", "recent": []},
        "session": {"consolidated": "", "recent": []},
    }

    try:
        # Agent-level
        agent_consolidated = ConsolidatedMemory.find_one({
            "scope_type": "agent",
            "agent_id": agent_id,
        })
        if agent_consolidated:
            result["agent"]["consolidated"] = agent_consolidated.consolidated_content

        agent_reflections = get_unabsorbed_reflections(
            scope="agent",
            agent_id=agent_id,
            limit=20,
        )
        result["agent"]["recent"] = [r.content for r in agent_reflections]

        # User-level (if user_id provided)
        if user_id:
            user_consolidated = ConsolidatedMemory.find_one({
                "scope_type": "user",
                "agent_id": agent_id,
                "user_id": user_id,
            })
            if user_consolidated:
                result["user"]["consolidated"] = user_consolidated.consolidated_content

            user_reflections = get_unabsorbed_reflections(
                scope="user",
                agent_id=agent_id,
                user_id=user_id,
                limit=10,
            )
            result["user"]["recent"] = [r.content for r in user_reflections]

        # Session-level (if session_id provided)
        if session_id:
            session_consolidated = ConsolidatedMemory.find_one({
                "scope_type": "session",
                "agent_id": agent_id,
                "session_id": session_id,
            })
            if session_consolidated:
                result["session"]["consolidated"] = (
                    session_consolidated.consolidated_content
                )

            session_reflections = get_unabsorbed_reflections(
                scope="session",
                agent_id=agent_id,
                session_id=session_id,
                limit=15,
            )
            result["session"]["recent"] = [r.content for r in session_reflections]

    except Exception as e:
        logger.error(f"Error getting reflections for context: {e}")
        traceback.print_exc()

    return result


async def cleanup_session_reflections(session_id: ObjectId) -> int:
    """
    Clean up reflections and consolidated memory for a completed session.

    This should be called when a session ends to prevent orphaned data.

    Args:
        session_id: Session ID to clean up

    Returns:
        Number of reflections deleted
    """
    try:
        # Delete session reflections
        reflection_collection = Reflection.get_collection()
        result = reflection_collection.delete_many({
            "session_id": session_id,
            "scope": "session",
        })
        deleted_count = result.deleted_count

        # Delete session consolidated memory
        consolidated_collection = ConsolidatedMemory.get_collection()
        consolidated_collection.delete_one({
            "session_id": session_id,
            "scope_type": "session",
        })

        if LOCAL_DEV:
            logger.debug(
                f"Cleaned up {deleted_count} session reflections for session {session_id}"
            )

        return deleted_count

    except Exception as e:
        logger.error(f"Error cleaning up session reflections: {e}")
        traceback.print_exc()
        return 0
