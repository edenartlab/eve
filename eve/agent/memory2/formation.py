"""
Memory System v2 - Memory Formation Orchestration

This module orchestrates the complete memory formation pipeline:
1. Extract facts from conversation (LLM Call 1 - fast model)
2. Process facts through deduplication pipeline (LLM Call 1.5)
3. Extract reflections with awareness of facts (LLM Call 2 - full model)
4. Store reflections and trigger consolidation if needed
5. Assemble and cache memory context

Memory formation is async and never blocks user-facing responses.
"""

import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from loguru import logger


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is UTC-aware. Returns None if input is None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

from eve.agent.memory2.consolidation import maybe_consolidate_all
from eve.agent.memory2.constants import (
    LOCAL_DEV,
    MEMORY_FORMATION_MSG_INTERVAL,
    MEMORY_FORMATION_TOKEN_INTERVAL,
    NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES,
    RAG_ENABLED,
    FACTS_FIFO_ENABLED,
    Memory2Config,
)
from eve.agent.memory2.context_assembly import get_memory_context_for_session
from eve.agent.memory2.reflection_extraction import extract_and_save_reflections


# Character weighting for token estimation (same as v1)
USER_MULTIPLIER = 1.0
TOOL_MULTIPLIER = 0.5
AGENT_MULTIPLIER = 0.2
OTHER_MULTIPLIER = 0.5


def messages_to_text(
    messages: List,
    skip_trigger_messages: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """
    Convert messages to readable text for LLM processing.

    Args:
        messages: List of ChatMessage objects
        skip_trigger_messages: Skip messages from triggers

    Returns:
        Tuple of (formatted_text, char_counts_by_source)
    """
    # Import here to avoid circular imports
    from eve.agent.memory2.utils import (
        get_sender_id_to_sender_name_map,
    )

    sender_map = get_sender_id_to_sender_name_map(messages)
    text_parts = []
    char_counts = {"user": 0, "agent": 0, "tool": 0, "other": 0}

    for msg in messages:
        # Skip system messages
        if msg.role == "system":
            continue

        # Skip trigger messages if requested
        if skip_trigger_messages and getattr(msg, "trigger", False):
            continue

        # Skip messages with no content
        content = (msg.content or "").strip()
        if not content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            continue

        speaker = sender_map.get(msg.sender) or msg.name or msg.role

        # Count characters by source
        if msg.role == "user":
            char_counts["user"] += len(content)
        elif msg.role in ["agent", "assistant", "eden"]:
            char_counts["agent"] += len(content)
        else:
            char_counts["other"] += len(content)

        # Add tool call summaries
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tools_used = [tc.tool for tc in msg.tool_calls if hasattr(tc, "tool")]
            if tools_used:
                tools_summary = f" [Used tools: {', '.join(tools_used)}]"
                content += tools_summary
                char_counts["tool"] += len(tools_summary)

        text_parts.append(f"{speaker}: {content}")

    return "\n".join(text_parts), char_counts


def estimate_weighted_tokens(char_counts: Dict[str, int]) -> int:
    """
    Estimate token count with source-weighted character counts.

    User messages are weighted more heavily than agent messages.
    """
    weighted_chars = (
        char_counts.get("user", 0) * USER_MULTIPLIER
        + char_counts.get("tool", 0) * TOOL_MULTIPLIER
        + char_counts.get("agent", 0) * AGENT_MULTIPLIER
        + char_counts.get("other", 0) * OTHER_MULTIPLIER
    )
    return int(weighted_chars / 4.5)  # ~4.5 chars per token


def should_form_memories(
    agent_id: ObjectId,
    session,
    messages: List,
) -> Tuple[bool, Optional[str], Optional[Dict[str, int]]]:
    """
    Check if memory formation should run.

    Returns:
        Tuple of (should_form, conversation_text, char_counts)
        conversation_text and char_counts are only populated if should_form is True
    """
    try:
        if not agent_id or not messages or len(messages) == 0:
            return False, None, None

        # Get last memory formation position
        memory_context = getattr(session, "memory_context", None)
        last_memory_message_id = None
        if memory_context:
            last_memory_message_id = getattr(
                memory_context, "last_memory_message_id", None
            )

        # Find position of last memory message
        message_id_to_idx = {msg.id: i for i, msg in enumerate(messages)}
        last_memory_idx = -1
        if last_memory_message_id:
            last_memory_idx = message_id_to_idx.get(last_memory_message_id, -1)

        # Get messages since last formation
        recent_messages = messages[last_memory_idx + 1:]
        messages_since_last = len(recent_messages)

        # Check minimum threshold
        if messages_since_last < NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES:
            return False, None, None

        # Check message-based trigger
        if messages_since_last >= MEMORY_FORMATION_MSG_INTERVAL:
            return True, None, None

        # Check token-based trigger
        conversation_text, char_counts = messages_to_text(recent_messages)
        tokens_since_last = estimate_weighted_tokens(char_counts)

        if tokens_since_last >= MEMORY_FORMATION_TOKEN_INTERVAL:
            return True, conversation_text, char_counts

        return False, None, None

    except Exception as e:
        logger.error(f"Error in should_form_memories: {e}")
        traceback.print_exc()
        return False, None, None


async def form_memories(
    agent_id: ObjectId,
    session,
    messages: List,
    user_id: Optional[ObjectId] = None,
    conversation_text: Optional[str] = None,
    char_counts: Optional[Dict[str, int]] = None,
    config: Optional[Memory2Config] = None,
) -> bool:
    """
    Form memories from conversation messages.

    This is the main entry point for memory formation. It orchestrates:
    1. Fact extraction (Phase 2) - only for enabled scopes
    2. Reflection extraction - only for enabled scopes
    3. Storage and consolidation
    4. Context refresh

    Args:
        agent_id: Agent ID
        session: Session object
        messages: List of messages to process
        user_id: User ID for user-scoped memories
        conversation_text: Pre-computed conversation text
        char_counts: Pre-computed character counts
        config: Memory2Config with enabled scopes (loaded if not provided)

    Returns:
        True if memories were formed successfully
    """
    start_time = time.time()

    try:
        if not messages or len(messages) < NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES:
            return False

        # Load config if not provided
        # Pass session to detect multi-user context and disable user memory if needed
        if config is None:
            config = _get_memory2_config(agent_id, session=session)

        # Note: We always proceed even if config.any_enabled is False
        # because session reflections are ALWAYS formed regardless of toggles.
        # config.reflection_scopes always includes "session".

        session_id = getattr(session, "id", None)

        # Get last memory formation position
        memory_context = getattr(session, "memory_context", None)
        last_memory_message_id = None
        if memory_context:
            last_memory_message_id = getattr(
                memory_context, "last_memory_message_id", None
            )

        # Find position of last memory message
        message_id_to_idx = {msg.id: i for i, msg in enumerate(messages)}
        last_memory_idx = -1
        if last_memory_message_id:
            last_memory_idx = message_id_to_idx.get(last_memory_message_id, -1)

        # Get messages since last formation
        recent_messages = messages[last_memory_idx + 1:]
        # Extract message IDs only from the messages being processed
        message_ids = [msg.id for msg in recent_messages if hasattr(msg, "id")]

        if len(recent_messages) < NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES:
            return False

        # IMMEDIATELY claim messages to prevent race conditions
        last_message_id = messages[-1].id if messages else None
        _update_memory_context(
            session,
            last_memory_message_id=last_message_id,
            last_activity=datetime.now(timezone.utc),
        )

        # Compute conversation text if not provided
        if conversation_text is None:
            conversation_text, char_counts = messages_to_text(recent_messages)

        # Extract user ID from messages if not provided
        if user_id is None:
            user_id = _extract_user_id(recent_messages, agent_id, session)

        # Fetch agent persona for memory extraction context
        agent_persona = _get_agent_persona(agent_id)

        if LOCAL_DEV:
            logger.debug(f"\n{'='*60}")
            logger.debug(f"MEMORY FORMATION STARTING")
            logger.debug(f"{'='*60}")
            logger.debug(f"  Agent ID: {agent_id}")
            logger.debug(f"  User ID: {user_id}")
            logger.debug(f"  Session ID: {session_id}")
            logger.debug(f"  Multi-user session: {config.is_multi_user}")
            if config.is_multi_user:
                session_users = getattr(session, "users", []) or []
                logger.debug(f"  Session users count: {len(session_users)} (user memory DISABLED)")
            logger.debug(f"  Enabled scopes - facts: {config.fact_scopes}, reflections: {config.reflection_scopes}")
            logger.debug(f"  Messages to process: {len(recent_messages)}")
            logger.debug(f"  Conversation length: {len(conversation_text)} chars")
            logger.debug(f"{'='*60}")

        # --- LLM CALL 1: Fact Extraction ---
        # Facts are extracted when either RAG or FIFO mode is enabled AND
        # at least one fact scope is enabled (user or agent).
        newly_formed_facts: List[str] = []
        fact_count = 0
        saved_facts = []

        if (RAG_ENABLED or FACTS_FIFO_ENABLED) and config.fact_scopes:
            from eve.agent.memory2.fact_extraction import extract_and_prepare_facts

            # Extract facts only for enabled scopes
            prepared_facts, fact_contents = await extract_and_prepare_facts(
                conversation_text=conversation_text,
                agent_id=agent_id,
                user_id=user_id,
                session_id=session_id,
                message_ids=message_ids,
                agent_persona=agent_persona,
                enabled_scopes=config.fact_scopes,
            )

            if prepared_facts:
                if RAG_ENABLED:
                    # Full RAG pipeline: deduplication + vector search
                    from eve.agent.memory2.fact_management import process_extracted_facts

                    saved_facts, newly_formed_facts = await process_extracted_facts(
                        extracted_facts=prepared_facts,
                        agent_id=agent_id,
                        user_id=user_id,
                    )
                    fact_count = len(saved_facts)
                else:
                    # TEMPORARY: FIFO mode - store facts with embeddings, skip dedup LLM
                    from eve.agent.memory2.fact_storage import store_facts_batch

                    saved_facts = await store_facts_batch(prepared_facts)
                    newly_formed_facts = [
                        f.content if hasattr(f, 'content') else f.get('content', '')
                        for f in saved_facts
                    ]
                    fact_count = len(saved_facts)

            if LOCAL_DEV:
                mode = "RAG" if RAG_ENABLED else "FIFO"
                if fact_count > 0:
                    print(f"\n✓ Formed {fact_count} new facts ({mode} mode, scopes: {config.fact_scopes}):")
                    for fact in saved_facts:
                        scope = fact.scope if hasattr(fact, 'scope') else fact.get('scope', [])
                        content = fact.content if hasattr(fact, 'content') else fact.get('content', '')
                        scope_str = ", ".join(scope) if isinstance(scope, list) else str(scope)
                        print(f"    - [{scope_str}] {content[:80]}...")
                else:
                    print(f"No new facts generated. ({mode} mode, scopes: {config.fact_scopes})")

        # --- LLM CALL 2: Reflection Extraction (with fact awareness) ---
        reflections_by_scope, reflection_count = await extract_and_save_reflections(
            conversation_text=conversation_text,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            message_ids=message_ids,
            newly_formed_facts=newly_formed_facts,
            agent_persona=agent_persona,
            enabled_scopes=config.reflection_scopes,
            model=config.slow_model,
        )

        if LOCAL_DEV:
            print(f"\n✓ Formed {reflection_count} new reflections (scopes: {config.reflection_scopes}):")
            for scope, reflections in reflections_by_scope.items():
                if reflections:
                    print(f"  {len(reflections)} x {scope}:")
                    for r in reflections:
                        print(f"    - {r.content[:100]}...")

        # --- Check and trigger consolidation (only for enabled scopes) ---
        consolidation_results = await maybe_consolidate_all(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            agent_persona=agent_persona,
            enabled_scopes=config.reflection_scopes,
            model=config.slow_model,
        )

        if LOCAL_DEV:
            consolidated_scopes = [k for k, v in consolidation_results.items() if v]
            if consolidated_scopes:
                print(f"\n✓ Consolidated scopes: {', '.join(consolidated_scopes)}")

        # --- Refresh memory context (force rebuild after forming memories) ---
        memory_xml = await get_memory_context_for_session(
            session=session,
            agent_id=agent_id,
            last_speaker_id=user_id,
            force_refresh=True,
        )

        elapsed = time.time() - start_time
        if LOCAL_DEV:
            word_count = len(memory_xml.split()) if memory_xml else 0
            print(f"\n✓ Memory formation completed in {elapsed:.2f}s")
            print(f"  Memory context: {word_count} words")

        return True

    except Exception as e:
        logger.error(f"Error in form_memories: {e}")
        traceback.print_exc()
        return False


def _get_memory2_config(agent_id: ObjectId, session=None) -> Memory2Config:
    """
    Get the Memory2Config for the given agent and session context.

    Memory2 is controlled by user_memory_enabled and agent_memory_enabled
    flags on the Agent model. Session memory is always active when any
    memory is enabled.

    IMPORTANT: For multi-user sessions (group chats), user memory is automatically
    disabled to prevent memory leakage between users.

    Args:
        agent_id: The agent's ObjectId
        session: Optional session object to detect multi-user context

    Returns:
        Memory2Config with enabled scopes (user scope disabled for multi-user sessions)
    """
    return Memory2Config.from_agent_id(agent_id, session=session)


async def maybe_form_memories(
    agent_id: ObjectId,
    session,
    messages: List,
    user_id: Optional[ObjectId] = None,
) -> bool:
    """
    Check if memory formation should run, and run it if so.

    This is the main entry point called from the agent loop.

    Session reflections are ALWAYS formed, regardless of user/agent toggles.
    User and agent reflections/facts are only formed when their respective
    toggles are enabled.

    Args:
        agent_id: Agent ID
        session: Session object
        messages: List of messages
        user_id: User ID (optional)

    Returns:
        True if memories were formed
    """
    # Get memory2 configuration for this agent and session context
    # Multi-user sessions automatically disable user memory to prevent leakage
    config = _get_memory2_config(agent_id, session=session)

    # Session reflections are always formed - no early return here
    # (config.reflection_scopes always includes "session")

    # Check if incognito mode
    if hasattr(session, "extras") and session.extras:
        if getattr(session.extras, "incognito", False):
            return False

    should_form, conversation_text, char_counts = should_form_memories(
        agent_id=agent_id,
        session=session,
        messages=messages,
    )

    if not should_form:
        return False

    return await form_memories(
        agent_id=agent_id,
        session=session,
        messages=messages,
        user_id=user_id,
        conversation_text=conversation_text,
        char_counts=char_counts,
        config=config,
    )


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


def _update_memory_context(
    session,
    last_memory_message_id: Optional[ObjectId] = None,
    last_activity: Optional[datetime] = None,
) -> None:
    """
    Update the session's memory context.

    Uses the existing SessionMemoryContext pattern from v1.
    """
    try:
        memory_context = _ensure_memory_context_object(session)

        # Update fields
        if last_memory_message_id:
            memory_context.last_memory_message_id = last_memory_message_id

        if last_activity:
            memory_context.last_activity = last_activity

        memory_context.messages_since_memory_formation = 0

        # Save to database
        session.update(memory_context=memory_context.model_dump())
        _ensure_memory_context_object(session)  # Re-instantiate as model object

    except Exception as e:
        logger.error(f"Error updating memory context: {e}")


def _extract_user_id(
    messages: List, agent_id: ObjectId, session=None
) -> Optional[ObjectId]:
    """
    Extract the primary user ID from messages or session.

    Fallback order:
    1. Last non-agent sender from messages with role="user"
    2. First user in session.users list
    3. Session owner (session.owner)
    """
    try:
        # Try to extract from messages first
        for msg in reversed(messages):
            sender = getattr(msg, "sender", None)
            if sender and sender != agent_id:
                role = getattr(msg, "role", None)
                if role == "user":
                    return sender

        # Fall back to session.users if available
        if session is not None:
            users = getattr(session, "users", None)
            if users and len(users) > 0:
                return users[0]

            # Final fallback: session owner
            owner = getattr(session, "owner", None)
            if owner:
                return owner

        return None

    except Exception as e:
        logger.error(f"Error extracting user ID: {e}")
        return None


def _get_agent_persona(agent_id: ObjectId) -> Optional[str]:
    """
    Fetch the agent's persona from the database.

    Args:
        agent_id: The agent's ObjectId

    Returns:
        The agent's persona string, or None if not found
    """
    try:
        from eve.agent.agent import Agent

        agent = Agent.from_mongo(agent_id)
        if agent:
            return agent.persona
        return None

    except Exception as e:
        logger.error(f"Error fetching agent persona: {e}")
        return None


async def process_cold_session(
    session,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
) -> bool:
    """
    Process a cold (inactive) session for memory formation.

    Called by background job when session has been inactive for
    CONSIDER_COLD_AFTER_MINUTES.

    Args:
        session: Session object
        agent_id: Agent ID
        user_id: User ID (optional)

    Returns:
        True if memories were formed
    """
    # Get memory2 configuration for this agent and session context
    # Multi-user sessions automatically disable user memory to prevent leakage
    config = _get_memory2_config(agent_id, session=session)

    # Note: We always proceed even if config.any_enabled is False
    # because session reflections are ALWAYS formed regardless of toggles.

    messages = None
    try:
        # Import here to avoid circular imports
        from eve.agent.memory2.utils import select_messages

        messages = select_messages(session)
        if not messages:
            return False

        # Force memory formation regardless of thresholds
        return await form_memories(
            agent_id=agent_id,
            session=session,
            messages=messages,
            user_id=user_id,
            config=config,
        )

    except Exception as e:
        logger.error(f"Error processing cold session {session.id}: {e}")
        traceback.print_exc()

        # On error, advance last_memory_message_id to skip these messages
        # in future runs (prevents infinite retry loops on bad data)
        if messages:
            last_message_id = messages[-1].id if messages else None
            if last_message_id:
                _update_memory_context(
                    session,
                    last_memory_message_id=last_message_id,
                    last_activity=datetime.now(timezone.utc),
                )
                logger.error(
                    f"Skipping messages up to {last_message_id} for session {session.id} due to error"
                )

        return False
