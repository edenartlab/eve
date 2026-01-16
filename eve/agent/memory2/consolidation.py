"""
Memory System v2 - Consolidation

This module handles the consolidation of buffered reflections into condensed blobs.
Consolidation is triggered when the buffer threshold is exceeded.

All three scope levels (agent, user, session) consolidate using the same pattern:
1. Load current consolidated blob + unabsorbed reflections
2. Call LLM to merge them into a new consolidated blob
3. Update the consolidated memory and mark reflections as absorbed

Consolidations for different scopes run in PARALLEL since they are independent.
LLM calls include automatic retry with exponential backoff.
"""

import asyncio
import os
import traceback
import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.llm.llm import async_prompt
from eve.utils.system_utils import async_exponential_backoff
from eve.agent.memory2.constants import (
    CONSOLIDATED_WORD_LIMITS,
    CONSOLIDATION_INSTRUCTIONS,
    CONSOLIDATION_PROMPT,
    CONSOLIDATION_THRESHOLDS,
    LOCAL_DEV,
    MEMORY_LLM_MODEL_SLOW,
)
from eve.agent.memory2.models import (
    ConsolidatedMemory,
    Reflection,
    mark_reflections_absorbed,
)
from eve.agent.session.models import (
    ChatMessage,
    LLMConfig,
    LLMContext,
    LLMContextMetadata,
    LLMTraceMetadata,
)


async def consolidate_reflections(
    scope: Literal["agent", "user", "session"],
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    force: bool = False,
    agent_persona: Optional[str] = None,
) -> Optional[str]:
    """
    Consolidate buffered reflections into the consolidated blob.

    This function:
    1. Checks if consolidation threshold is met (unless forced)
    2. Loads current consolidated blob and unabsorbed reflections
    3. Calls LLM to merge them
    4. Updates the consolidated memory
    5. Marks reflections as absorbed

    Args:
        scope: Scope type to consolidate
        agent_id: Agent ID
        user_id: User ID (required for user scope)
        session_id: Session ID (required for session scope)
        force: Force consolidation even if threshold not met
        agent_persona: The agent's persona/description for context

    Returns:
        The new consolidated content, or None if no consolidation occurred
    """
    try:
        # Get consolidated memory document
        consolidated = _get_consolidated_memory(scope, agent_id, user_id, session_id)
        if not consolidated:
            if LOCAL_DEV:
                logger.debug(f"No consolidated memory found for {scope} scope")
            return None

        # Check threshold unless forced
        buffer_size = len(consolidated.unabsorbed_ids)
        threshold = CONSOLIDATION_THRESHOLDS.get(scope, 5)

        if not force and buffer_size < threshold:
            if LOCAL_DEV:
                logger.debug(
                    f"Buffer size ({buffer_size}) below threshold ({threshold}) for {scope}"
                )
            return None

        if buffer_size == 0:
            if LOCAL_DEV:
                logger.debug(f"No unabsorbed reflections to consolidate for {scope}")
            return None

        # Load unabsorbed reflections
        reflections = _load_reflections_by_ids(consolidated.unabsorbed_ids)
        if not reflections:
            logger.warning(
                f"Could not load any reflections from {len(consolidated.unabsorbed_ids)} IDs"
            )
            return None

        # Format reflections for prompt
        reflections_text = "\n".join(f"- {r.content}" for r in reflections)

        # Get scope-specific instructions
        scope_instructions = CONSOLIDATION_INSTRUCTIONS.get(scope, "")

        # Build consolidation prompt
        prompt = CONSOLIDATION_PROMPT.format(
            scope_type=scope,
            agent_persona=agent_persona or "No agent persona available.",
            existing_blob=consolidated.consolidated_content
            or "EMPTY (This is the first consolidation - be concise, more reflections will come!)",
            new_reflections=reflections_text,
            scope_specific_instructions=scope_instructions,
            word_limit=consolidated.word_limit,
        )

        # LLM call
        context = LLMContext(
            messages=[ChatMessage(role="user", content=prompt)],
            config=LLMConfig(model=MEMORY_LLM_MODEL_SLOW),
            metadata=LLMContextMetadata(
                session_id=f"{os.getenv('DB')}-memory2-consolidation-{scope}",
                trace_name="FN_memory2_consolidation",
                trace_id=str(uuid.uuid4()),
                generation_name=f"memory2_consolidate_{scope}",
                trace_metadata=LLMTraceMetadata(
                    agent_id=str(agent_id),
                    user_id=str(user_id) if user_id else None,
                    session_id=str(session_id) if session_id else None,
                ),
            ),
            enable_tracing=True,
        )

        if LOCAL_DEV:
            logger.debug(f"Running {scope} consolidation with {len(reflections)} reflections...")

        # LLM call with automatic retry (3 attempts with exponential backoff)
        response = await async_exponential_backoff(
            lambda: async_prompt(context),
            max_attempts=3,
            initial_delay=2,
            max_jitter=0.5,
        )
        new_content = response.content.strip()

        # Validate word count
        word_count = len(new_content.split())
        if word_count > consolidated.word_limit * 1.2:  # Allow 20% overflow
            logger.warning(
                f"Consolidation exceeded word limit: {word_count} > {consolidated.word_limit}"
            )

        # Update consolidated memory atomically
        reflection_ids = [r.id for r in reflections]
        _update_consolidated_memory(
            consolidated=consolidated,
            new_content=new_content,
            absorbed_ids=reflection_ids,
        )

        # Mark reflections as absorbed
        mark_reflections_absorbed(reflection_ids, consolidated.id)

        if LOCAL_DEV:
            logger.debug(
                f"Consolidated {len(reflections)} {scope} reflections ({word_count} words)"
            )

        return new_content

    except Exception as e:
        logger.error(f"Error consolidating {scope} reflections: {e}")
        traceback.print_exc()
        return None


def _get_consolidated_memory(
    scope: Literal["agent", "user", "session"],
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    session_id: Optional[ObjectId],
) -> Optional[ConsolidatedMemory]:
    """Get the consolidated memory document for a scope."""
    query = {
        "scope_type": scope,
        "agent_id": agent_id,
    }

    if scope == "user":
        if user_id is None:
            return None
        query["user_id"] = user_id

    if scope == "session":
        if session_id is None:
            return None
        query["session_id"] = session_id

    return ConsolidatedMemory.find_one(query)


def _load_reflections_by_ids(reflection_ids: List[ObjectId]) -> List[Reflection]:
    """Load reflections by their IDs."""
    if not reflection_ids:
        return []

    # Limit batch size to prevent resource exhaustion
    MAX_BATCH = 100
    if len(reflection_ids) > MAX_BATCH:
        logger.warning(
            f"Truncating reflection IDs from {len(reflection_ids)} to {MAX_BATCH}"
        )
        reflection_ids = reflection_ids[:MAX_BATCH]

    try:
        reflections = Reflection.find({"_id": {"$in": reflection_ids}})

        # Preserve order
        id_to_ref = {r.id: r for r in reflections}
        return [id_to_ref[rid] for rid in reflection_ids if rid in id_to_ref]

    except Exception as e:
        logger.error(f"Error loading reflections: {e}")
        return []


def _update_consolidated_memory(
    consolidated: ConsolidatedMemory,
    new_content: str,
    absorbed_ids: List[ObjectId],
) -> None:
    """
    Update the consolidated memory atomically.

    Removes absorbed IDs from unabsorbed_ids and updates content.
    Uses MongoDB atomic operations to prevent race conditions.
    """
    try:
        collection = ConsolidatedMemory.get_collection()

        # Atomic update: set new content, pull absorbed IDs, update timestamp
        collection.update_one(
            {"_id": consolidated.id},
            {
                "$set": {
                    "consolidated_content": new_content,
                    "last_consolidated_at": datetime.now(timezone.utc),
                },
                "$pull": {"unabsorbed_ids": {"$in": absorbed_ids}},
                "$currentDate": {"updatedAt": True},
            },
        )

        # Update local state
        consolidated.consolidated_content = new_content
        consolidated.last_consolidated_at = datetime.now(timezone.utc)
        consolidated.unabsorbed_ids = [
            uid for uid in consolidated.unabsorbed_ids if uid not in absorbed_ids
        ]

    except Exception as e:
        logger.error(f"Error updating consolidated memory: {e}")
        # Fallback to non-atomic update
        consolidated.consolidated_content = new_content
        consolidated.last_consolidated_at = datetime.now(timezone.utc)
        consolidated.unabsorbed_ids = [
            uid for uid in consolidated.unabsorbed_ids if uid not in absorbed_ids
        ]
        consolidated.save()


async def maybe_consolidate_all(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    agent_persona: Optional[str] = None,
) -> dict:
    """
    Check and consolidate all applicable scopes if thresholds are met.

    Consolidations run in PARALLEL since they are independent of each other.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional)
        session_id: Session ID (optional)
        agent_persona: The agent's persona/description for context

    Returns:
        Dict with consolidation results for each scope
    """
    # Build list of consolidation tasks to run in parallel
    tasks = []
    scope_names = []

    # Always check agent scope
    tasks.append(consolidate_reflections(scope="agent", agent_id=agent_id, agent_persona=agent_persona))
    scope_names.append("agent")

    # Check user scope if user_id provided
    if user_id:
        tasks.append(consolidate_reflections(scope="user", agent_id=agent_id, user_id=user_id, agent_persona=agent_persona))
        scope_names.append("user")

    # Check session scope if session_id provided
    if session_id:
        tasks.append(consolidate_reflections(scope="session", agent_id=agent_id, session_id=session_id, agent_persona=agent_persona))
        scope_names.append("session")

    # Run all consolidations in parallel
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Map results back to scope names
    results = {"agent": None, "user": None, "session": None}
    for scope_name, result in zip(scope_names, task_results):
        if isinstance(result, Exception):
            logger.error(f"Consolidation failed for {scope_name}: {result}")
            results[scope_name] = None
        else:
            results[scope_name] = result

    consolidated_count = sum(1 for v in results.values() if v is not None)
    if LOCAL_DEV and consolidated_count > 0:
        logger.debug(f"Consolidated {consolidated_count} scope(s) in parallel")

    return results


async def force_consolidate_all(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    agent_persona: Optional[str] = None,
) -> dict:
    """
    Force consolidation for all applicable scopes, regardless of thresholds.

    Consolidations run in PARALLEL since they are independent of each other.
    Useful for session end cleanup or manual maintenance.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional)
        session_id: Session ID (optional)
        agent_persona: The agent's persona/description for context

    Returns:
        Dict with consolidation results for each scope
    """
    # Build list of consolidation tasks to run in parallel
    tasks = []
    scope_names = []

    # Always force agent scope
    tasks.append(consolidate_reflections(scope="agent", agent_id=agent_id, force=True, agent_persona=agent_persona))
    scope_names.append("agent")

    # Force user scope if user_id provided
    if user_id:
        tasks.append(consolidate_reflections(scope="user", agent_id=agent_id, user_id=user_id, force=True, agent_persona=agent_persona))
        scope_names.append("user")

    # Force session scope if session_id provided
    if session_id:
        tasks.append(consolidate_reflections(scope="session", agent_id=agent_id, session_id=session_id, force=True, agent_persona=agent_persona))
        scope_names.append("session")

    # Run all consolidations in parallel
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Map results back to scope names
    results = {"agent": None, "user": None, "session": None}
    for scope_name, result in zip(scope_names, task_results):
        if isinstance(result, Exception):
            logger.error(f"Force consolidation failed for {scope_name}: {result}")
            results[scope_name] = None
        else:
            results[scope_name] = result

    return results


def get_consolidation_status(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
) -> dict:
    """
    Get the current consolidation status for all scopes.

    Returns buffer sizes and whether consolidation is needed.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional)
        session_id: Session ID (optional)

    Returns:
        Dict with status for each scope:
        {
            "agent": {"buffer_size": int, "threshold": int, "needs_consolidation": bool},
            ...
        }
    """
    status = {}

    for scope in ["agent", "user", "session"]:
        # Skip scopes without required IDs
        if scope == "user" and not user_id:
            continue
        if scope == "session" and not session_id:
            continue

        consolidated = _get_consolidated_memory(
            scope=scope,
            agent_id=agent_id,
            user_id=user_id if scope == "user" else None,
            session_id=session_id if scope == "session" else None,
        )

        threshold = CONSOLIDATION_THRESHOLDS.get(scope, 5)
        buffer_size = len(consolidated.unabsorbed_ids) if consolidated else 0

        status[scope] = {
            "buffer_size": buffer_size,
            "threshold": threshold,
            "needs_consolidation": buffer_size >= threshold,
            "word_limit": CONSOLIDATED_WORD_LIMITS.get(scope, 400),
            "current_word_count": (
                len(consolidated.consolidated_content.split())
                if consolidated and consolidated.consolidated_content
                else 0
            ),
        }

    return status
