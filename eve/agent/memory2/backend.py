"""
Memory System v2 - Backend

This module provides the memory backend abstraction and the Memory2Backend
implementation that powers the memory system.
"""

import logging
import traceback
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.constants import LOCAL_DEV

if TYPE_CHECKING:
    from eve.agent import Agent
    from eve.agent.session.models import Session
    from eve.user import User

# Use standard logging for abstract class
_logger = logging.getLogger(__name__)


class MemoryBackend(ABC):
    """Minimal interface for memory backends."""

    @abstractmethod
    async def assemble_memory_context(
        self,
        session: "Session",
        agent: "Agent",
        user: "User",
        *,
        force_refresh: bool = False,
        reason: str = "unknown",
        skip_save: bool = False,
        instrumentation=None,
    ) -> str:
        """Return the memory context string for the given session."""

    @abstractmethod
    async def maybe_form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
    ) -> bool:
        """Conditionally form memories for the supplied session."""

    @abstractmethod
    async def form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
        *,
        conversation_text: Optional[str] = None,
        char_counts_by_source: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Force memory formation for the supplied session."""


class Memory2Backend(MemoryBackend):
    """
    Memory v2 backend implementation.

    This backend uses the new reflection-based always-in-context system
    and (optionally) the RAG-based fact retrieval system.
    """

    def __init__(self):
        self._indexes_ensured = False

    def _ensure_indexes(self) -> None:
        """Ensure MongoDB indexes are created (once per process)."""
        if self._indexes_ensured:
            return

        try:
            from eve.agent.memory2.models import (
                ConsolidatedMemory,
                Fact,
                Reflection,
            )

            Reflection.ensure_indexes()
            ConsolidatedMemory.ensure_indexes()
            # Only create Fact indexes if RAG is enabled
            from eve.agent.memory2.constants import RAG_ENABLED
            if RAG_ENABLED:
                Fact.ensure_indexes()

            self._indexes_ensured = True
            if LOCAL_DEV:
                logger.debug("Memory2 indexes ensured")

        except Exception as e:
            logger.error(f"Error ensuring memory2 indexes: {e}")

    async def assemble_memory_context(
        self,
        session: "Session",
        agent: "Agent",
        user: "User",
        *,
        force_refresh: bool = False,
        reason: str = "unknown",
        skip_save: bool = False,
        instrumentation=None,
    ) -> str:
        """
        Assemble memory context for prompt injection.

        This returns XML-formatted memory containing:
        - Agent-level consolidated + recent reflections
        - User-level consolidated + recent reflections
        - Session-level consolidated + recent reflections
        """
        self._ensure_indexes()

        try:
            from eve.agent.memory2.context_assembly import (
                get_memory_context_for_session,
            )

            # Get user ID for user-scoped memory
            user_id = user.id if user else None

            # Get memory context (handles caching internally)
            memory_xml = await get_memory_context_for_session(
                session=session,
                agent_id=agent.id,
                last_speaker_id=user_id,
                force_refresh=force_refresh,
                instrumentation=instrumentation,
            )

            if LOCAL_DEV and not instrumentation:
                word_count = len(memory_xml.split()) if memory_xml else 0
                logger.debug(
                    f"Memory context assembled ({word_count} words) - reason: {reason}"
                )

            return memory_xml

        except Exception as e:
            logger.error(f"Error assembling memory context: {e}")
            traceback.print_exc()
            return ""

    async def maybe_form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
    ) -> bool:
        """
        Check if memory formation should run, and run it if so.

        This is called as a background task after agent responses.
        """
        self._ensure_indexes()

        try:
            from eve.agent.memory2.utils import select_messages
            from eve.agent.memory2.formation import maybe_form_memories

            # Get messages for memory formation
            messages = select_messages(session)

            # Extract user ID from session/messages
            user_id = self._extract_user_id(messages, agent_id, session)

            return await maybe_form_memories(
                agent_id=agent_id,
                session=session,
                messages=messages,
                user_id=user_id,
            )

        except Exception as e:
            logger.error(f"Error in maybe_form_memories: {e}")
            traceback.print_exc()
            return False

    async def form_memories(
        self,
        agent_id: ObjectId,
        session: "Session",
        agent: Optional["Agent"] = None,
        *,
        conversation_text: Optional[str] = None,
        char_counts_by_source: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Force memory formation regardless of thresholds.
        """
        self._ensure_indexes()

        try:
            from eve.agent.memory2.utils import select_messages
            from eve.agent.memory2.formation import form_memories

            # Get messages
            messages = select_messages(session)

            # Extract user ID
            user_id = self._extract_user_id(messages, agent_id, session)

            return await form_memories(
                agent_id=agent_id,
                session=session,
                messages=messages,
                user_id=user_id,
                conversation_text=conversation_text,
                char_counts=char_counts_by_source,
            )

        except Exception as e:
            logger.error(f"Error in form_memories: {e}")
            traceback.print_exc()
            return False

    def _extract_user_id(
        self,
        messages: list,
        agent_id: ObjectId,
        session=None,
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
        except Exception:
            return None


# Singleton instance for easy access
memory2_backend = Memory2Backend()
