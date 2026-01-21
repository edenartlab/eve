"""
Memory System v2 - Reflection Extraction

This module handles the extraction of reflections from conversations.
Reflections are extracted with awareness of existing memory context to avoid
redundancy and maintain hierarchical scope ordering (agent → user → session).
"""

import json
import os
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

from bson import ObjectId
from loguru import logger

from eve.agent.llm.llm import async_prompt
from eve.agent.memory2.constants import (
    LOCAL_DEV,
    MEMORY_LLM_MODEL_SLOW,
    REFLECTION_MAX_WORDS,
    build_reflection_extraction_prompt,
)
from eve.utils.system_utils import async_exponential_backoff
from eve.agent.memory2.models import (
    ConsolidatedMemory,
    Reflection,
    ReflectionExtractionResponse,
    get_unabsorbed_reflections,
)
from eve.agent.session.models import (
    ChatMessage,
    LLMConfig,
    LLMContext,
    LLMContextMetadata,
    LLMTraceMetadata,
)


async def extract_reflections(
    conversation_text: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    newly_formed_facts: Optional[List[str]] = None,
    agent_persona: Optional[str] = None,
    model: str = MEMORY_LLM_MODEL_SLOW,
    enabled_scopes: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Extract reflections from conversation text using LLM.

    This is the second LLM call in the extraction pipeline, called AFTER
    fact extraction. It receives newly formed facts to avoid redundancy.

    Reflections are extracted hierarchically (only for enabled scopes):
    1. Agent reflections (broadest scope) - extracted first
    2. User reflections - only what's NOT in agent reflections
    3. Session reflections - only what's NOT in agent/user reflections

    Args:
        conversation_text: The conversation to extract reflections from
        agent_id: Agent ID
        user_id: User ID for user-scoped reflections
        session_id: Session ID (always populated on all reflections)
        newly_formed_facts: Facts extracted from the same conversation (for deduplication)
        agent_persona: The agent's persona/description for context
        model: LLM model to use
        enabled_scopes: List of enabled scopes ["session", "user", "agent"] or subset

    Returns:
        Dictionary with keys for enabled scopes containing lists of reflection strings
    """
    # Default to all scopes if not specified
    if enabled_scopes is None:
        enabled_scopes = ["session", "user", "agent"]

    # Skip if no scopes enabled
    if not enabled_scopes:
        return {"agent": [], "user": [], "session": []}

    try:
        # Gather existing memory context (only for enabled scopes)
        memory_context = await _gather_memory_context(
            agent_id, user_id, session_id, enabled_scopes=enabled_scopes
        )

        # Build prompt with only enabled scopes
        prompt = build_reflection_extraction_prompt(
            conversation_text=conversation_text,
            agent_persona=agent_persona or "No agent persona available.",
            memory_context=memory_context,
            newly_formed_facts=_format_facts(newly_formed_facts),
            enabled_scopes=enabled_scopes,
            max_words=REFLECTION_MAX_WORDS,
        )

        if LOCAL_DEV:
            print("\n" + "="*60)
            print("REFLECTION EXTRACTION PROMPT:")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")

        # LLM call with structured output
        context = LLMContext(
            messages=[ChatMessage(role="user", content=prompt)],
            config=LLMConfig(
                model=model,
                response_format=ReflectionExtractionResponse,
            ),
            metadata=LLMContextMetadata(
                session_id=f"{os.getenv('DB')}-{str(session_id)}"
                if session_id
                else f"{os.getenv('DB')}-memory2-reflection-extraction",
                trace_name="FN_memory2_extract_reflections",
                trace_id=str(uuid.uuid4()),
                generation_name="memory2_reflection_extraction",
                trace_metadata=LLMTraceMetadata(
                    session_id=str(session_id) if session_id else None,
                    user_id=str(user_id) if user_id else None,
                    agent_id=str(agent_id),
                ),
            ),
            enable_tracing=True,
        )

        if LOCAL_DEV:
            logger.debug("Running reflection extraction LLM call...")

        # LLM call with automatic retry (3 attempts with exponential backoff)
        response = await async_exponential_backoff(
            lambda: async_prompt(context),
            max_attempts=3,
            initial_delay=2,
            max_jitter=0.5,
        )

        # Parse structured JSON response
        if not response.content or not response.content.strip():
            logger.warning("LLM returned empty response for reflection extraction")
            return {"agent": [], "user": [], "session": []}

        extracted = ReflectionExtractionResponse(**json.loads(response.content))

        # Convert to simple dict format, only including enabled scopes
        result = {}
        if "agent" in enabled_scopes:
            result["agent"] = [r.content for r in extracted.agent_reflections]
        else:
            result["agent"] = []

        if "user" in enabled_scopes:
            result["user"] = [r.content for r in extracted.user_reflections]
        else:
            result["user"] = []

        if "session" in enabled_scopes:
            result["session"] = [r.content for r in extracted.session_reflections]
        else:
            result["session"] = []

        if LOCAL_DEV:
            total = sum(len(v) for v in result.values())
            logger.debug(f"Extracted {total} reflections (enabled scopes: {enabled_scopes}):")
            for scope, reflections in result.items():
                if reflections:
                    logger.debug(f"  {scope}: {len(reflections)}")
                    for r in reflections:
                        logger.debug(f"    - {r[:80]}...")

        return result

    except Exception as e:
        logger.error(f"Error extracting reflections: {e}")
        traceback.print_exc()
        return {"agent": [], "user": [], "session": []}


async def _gather_memory_context(
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    session_id: Optional[ObjectId],
    enabled_scopes: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]:
    """
    Gather existing memory context for reflection extraction.

    Returns consolidated blobs and recent unabsorbed reflections for enabled scopes only.

    Args:
        agent_id: Agent ID
        user_id: User ID for user-scoped context
        session_id: Session ID for session-scoped context
        enabled_scopes: List of enabled scopes to gather context for

    Returns:
        Dict with keys like "agent_blob", "agent_recent", etc.
    """
    if enabled_scopes is None:
        enabled_scopes = ["session", "user", "agent"]

    context = {
        "agent_blob": None,
        "agent_recent": None,
        "user_blob": None,
        "user_recent": None,
        "session_blob": None,
        "session_recent": None,
    }

    try:
        # Agent-level memory (only if enabled)
        if "agent" in enabled_scopes:
            agent_consolidated = ConsolidatedMemory.find_one({
                "scope_type": "agent",
                "agent_id": agent_id,
            })
            if agent_consolidated:
                context["agent_blob"] = agent_consolidated.consolidated_content

            agent_reflections = get_unabsorbed_reflections(
                scope="agent",
                agent_id=agent_id,
                limit=10,
            )
            if agent_reflections:
                context["agent_recent"] = "\n".join(
                    f"- {r.content}" for r in agent_reflections
                )

        # User-level memory (only if enabled and user_id provided)
        if "user" in enabled_scopes and user_id:
            user_consolidated = ConsolidatedMemory.find_one({
                "scope_type": "user",
                "agent_id": agent_id,
                "user_id": user_id,
            })
            if user_consolidated:
                context["user_blob"] = user_consolidated.consolidated_content

            user_reflections = get_unabsorbed_reflections(
                scope="user",
                agent_id=agent_id,
                user_id=user_id,
                limit=10,
            )
            if user_reflections:
                context["user_recent"] = "\n".join(
                    f"- {r.content}" for r in user_reflections
                )

        # Session-level memory (only if enabled and session_id provided)
        if "session" in enabled_scopes and session_id:
            session_consolidated = ConsolidatedMemory.find_one({
                "scope_type": "session",
                "agent_id": agent_id,
                "session_id": session_id,
            })
            if session_consolidated:
                context["session_blob"] = session_consolidated.consolidated_content

            session_reflections = get_unabsorbed_reflections(
                scope="session",
                agent_id=agent_id,
                session_id=session_id,
                limit=10,
            )
            if session_reflections:
                context["session_recent"] = "\n".join(
                    f"- {r.content}" for r in session_reflections
                )

    except Exception as e:
        logger.error(f"Error gathering memory context: {e}")
        traceback.print_exc()

    return context


def _format_facts(facts: Optional[List[str]]) -> str:
    """
    Format newly formed facts for inclusion in prompt.

    Returns empty string if no facts, allowing the facts section
    to be completely omitted from the prompt.
    """
    if not facts:
        return ""

    return "\n".join(f"- {fact}" for fact in facts)


async def extract_and_save_reflections(
    conversation_text: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    message_ids: Optional[List[ObjectId]] = None,
    newly_formed_facts: Optional[List[str]] = None,
    agent_persona: Optional[str] = None,
    enabled_scopes: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[Reflection]], int]:
    """
    Extract reflections from conversation and save to database.

    This is the main entry point for reflection extraction. It:
    1. Calls LLM to extract reflections (only for enabled scopes)
    2. Creates Reflection documents for each extracted reflection
    3. Saves to MongoDB
    4. Updates consolidated memory unabsorbed_ids lists

    Args:
        conversation_text: The conversation to extract reflections from
        agent_id: Agent ID
        user_id: User ID for user-scoped reflections
        session_id: Session ID (always populated on all reflections)
        message_ids: IDs of messages that were processed
        newly_formed_facts: Facts extracted from the same conversation
        agent_persona: The agent's persona/description for context
        enabled_scopes: List of enabled scopes ["session", "user", "agent"] or subset

    Returns:
        Tuple of (reflections_by_scope dict, total count)
    """
    # Default to all scopes if not specified
    if enabled_scopes is None:
        enabled_scopes = ["session", "user", "agent"]

    try:
        # Extract reflections for enabled scopes only
        extracted = await extract_reflections(
            conversation_text=conversation_text,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            newly_formed_facts=newly_formed_facts,
            agent_persona=agent_persona,
            enabled_scopes=enabled_scopes,
        )

        reflections_by_scope = {"agent": [], "user": [], "session": []}
        total_count = 0

        # Process each enabled scope
        for scope, contents in extracted.items():
            # Skip scopes that aren't enabled
            if scope not in enabled_scopes:
                continue

            if not contents:
                continue

            reflections = []
            for content in contents:
                # Skip empty content
                if not content or not content.strip():
                    continue

                # Determine scope-specific IDs
                reflection_user_id = user_id if scope == "user" else None

                reflection = Reflection(
                    content=content.strip(),
                    scope=scope,
                    agent_id=agent_id,
                    user_id=reflection_user_id,
                    session_id=session_id,  # Always track which session created this
                    source_message_ids=message_ids or [],
                )
                reflections.append(reflection)

            if reflections:
                # Assign IDs before save_many (which mutates list to dicts)
                for r in reflections:
                    if not r.id:
                        r.id = ObjectId()
                reflection_ids = [r.id for r in reflections]

                # Keep copies for return value before save_many mutates them
                saved_reflections = [
                    Reflection(**r.model_dump()) for r in reflections
                ]

                # Batch save reflections
                try:
                    Reflection.save_many(reflections)
                except Exception as e:
                    logger.error(
                        f"Batch save failed for {scope} reflections, falling back: {e}"
                    )
                    for r in saved_reflections:
                        r.save()

                reflections_by_scope[scope] = saved_reflections
                total_count += len(saved_reflections)

                # Update consolidated memory unabsorbed_ids
                await _add_to_unabsorbed(
                    scope=scope,
                    agent_id=agent_id,
                    user_id=user_id if scope == "user" else None,
                    session_id=session_id if scope == "session" else None,
                    reflection_ids=reflection_ids,
                )

        if LOCAL_DEV:
            logger.debug(f"Saved {total_count} reflections to database")

        return reflections_by_scope, total_count

    except Exception as e:
        logger.error(f"Error in extract_and_save_reflections: {e}")
        traceback.print_exc()
        return {"agent": [], "user": [], "session": []}, 0


async def _add_to_unabsorbed(
    scope: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    session_id: Optional[ObjectId],
    reflection_ids: List[ObjectId],
) -> None:
    """
    Add reflection IDs to the consolidated memory's unabsorbed_ids list.

    This is done atomically to prevent race conditions.
    """
    if not reflection_ids:
        return

    try:
        from eve.agent.memory2.constants import CONSOLIDATED_WORD_LIMITS

        # Get or create consolidated memory
        consolidated = ConsolidatedMemory.get_or_create(
            scope_type=scope,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            word_limit=CONSOLIDATED_WORD_LIMITS.get(scope, 400),
        )

        # Atomic update to add reflection IDs
        collection = ConsolidatedMemory.get_collection()
        collection.update_one(
            {"_id": consolidated.id},
            {"$push": {"unabsorbed_ids": {"$each": reflection_ids}}},
        )

    except Exception as e:
        logger.error(f"Error adding to unabsorbed for {scope}: {e}")
        traceback.print_exc()
