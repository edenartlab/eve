"""
Memory System v2 - Service Layer

This module provides a high-level facade for the memory system.
It exposes a clean API for interacting with memories from the agent system.

The MemoryService class is the main entry point for:
- Memory formation (extracting and storing memories from conversations)
- Memory retrieval (getting memory context for prompts)
- Consolidation management
- Memory statistics and debugging
"""

import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.constants import (
    ALWAYS_IN_CONTEXT_ENABLED,
    LOCAL_DEV,
    RAG_ENABLED,
)


class MemoryService:
    """
    High-level facade for the Memory System v2.

    This class provides a clean API for the agent system to interact
    with memories without needing to know the internal implementation.
    """

    def __init__(
        self,
        agent_id: ObjectId,
        enable_rag: bool = RAG_ENABLED,
        enable_always_in_context: bool = ALWAYS_IN_CONTEXT_ENABLED,
    ):
        """
        Initialize the memory service.

        Args:
            agent_id: Agent ID for this service instance
            enable_rag: Whether RAG retrieval is enabled
            enable_always_in_context: Whether always-in-context memory is enabled
        """
        self.agent_id = agent_id
        self.enable_rag = enable_rag
        self.enable_always_in_context = enable_always_in_context

    # -------------------------------------------------------------------------
    # Memory Formation
    # -------------------------------------------------------------------------

    async def maybe_form_memories(
        self,
        session,
        messages: List,
        user_id: Optional[ObjectId] = None,
    ) -> bool:
        """
        Check if memory formation should run, and run it if so.

        This is the main entry point for memory formation, typically called
        after the agent sends a response.

        Args:
            session: Session object
            messages: List of messages in the session
            user_id: User ID for user-scoped memories

        Returns:
            True if memories were formed
        """
        if not self.enable_always_in_context and not self.enable_rag:
            return False

        from eve.agent.memory2.formation import maybe_form_memories

        return await maybe_form_memories(
            agent_id=self.agent_id,
            session=session,
            messages=messages,
            user_id=user_id,
        )

    async def force_form_memories(
        self,
        session,
        messages: List,
        user_id: Optional[ObjectId] = None,
    ) -> bool:
        """
        Force memory formation regardless of thresholds.

        Useful for session end cleanup or manual triggering.

        Args:
            session: Session object
            messages: List of messages
            user_id: User ID

        Returns:
            True if memories were formed
        """
        from eve.agent.memory2.formation import form_memories

        return await form_memories(
            agent_id=self.agent_id,
            session=session,
            messages=messages,
            user_id=user_id,
        )

    # -------------------------------------------------------------------------
    # Memory Retrieval
    # -------------------------------------------------------------------------

    async def get_memory_context(
        self,
        session,
        user_id: Optional[ObjectId] = None,
        force_refresh: bool = False,
    ) -> str:
        """
        Get the always-in-context memory for prompt injection.

        This returns an XML-formatted string containing consolidated
        memory blobs and recent reflections for all applicable scopes.

        Args:
            session: Session object (for caching)
            user_id: User ID for user-scope memory
            force_refresh: Force regeneration of memory context

        Returns:
            XML-formatted memory context string
        """
        if not self.enable_always_in_context:
            return ""

        from eve.agent.memory2.context_assembly import get_memory_context_for_session

        return await get_memory_context_for_session(
            session=session,
            agent_id=self.agent_id,
            last_speaker_id=user_id,
            force_refresh=force_refresh,
        )

    async def search_facts(
        self,
        query: str,
        user_id: Optional[ObjectId] = None,
        limit: int = 10,
        search_type: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        """
        Search facts using RAG retrieval.

        Uses hybrid search (semantic + text) with Reciprocal Rank Fusion.

        Args:
            query: Search query
            user_id: User ID to filter user-scoped facts
            limit: Maximum number of results
            search_type: "hybrid", "semantic", or "text"

        Returns:
            List of fact documents with relevance scores
        """
        if not self.enable_rag:
            return []

        from eve.agent.memory2.rag import search_facts

        return await search_facts(
            query=query,
            agent_id=self.agent_id,
            user_id=user_id,
            match_count=limit,
            search_type=search_type,
        )

    async def get_relevant_facts(
        self,
        query: str,
        user_id: Optional[ObjectId] = None,
        max_facts: int = 5,
    ) -> str:
        """
        Get relevant facts formatted for context injection.

        This is a convenience method for getting facts to inject into
        agent context alongside the always-in-context memory.

        Args:
            query: Query text (usually the user's message)
            user_id: User ID
            max_facts: Maximum facts to include

        Returns:
            Formatted string of relevant facts
        """
        if not self.enable_rag:
            return ""

        from eve.agent.memory2.rag import get_relevant_facts_for_context

        return await get_relevant_facts_for_context(
            query=query,
            agent_id=self.agent_id,
            user_id=user_id,
            max_facts=max_facts,
        )

    def get_memory_search_tool(
        self,
        user_id: Optional[ObjectId] = None,
    ):
        """
        Get a memory search tool for the agent.

        This returns a tool that can be added to the agent's tool list
        to enable explicit memory retrieval via tool calls.

        Args:
            user_id: User ID (optional)

        Returns:
            MemorySearchTool instance
        """
        from eve.agent.memory2.rag_tool import get_memory_tool

        return get_memory_tool(self.agent_id, user_id)

    # -------------------------------------------------------------------------
    # Consolidation
    # -------------------------------------------------------------------------

    async def consolidate_scope(
        self,
        scope: str,
        user_id: Optional[ObjectId] = None,
        session_id: Optional[ObjectId] = None,
        force: bool = False,
    ) -> Optional[str]:
        """
        Consolidate reflections for a specific scope.

        Args:
            scope: Scope to consolidate ("agent", "user", "session")
            user_id: User ID (required for user scope)
            session_id: Session ID (required for session scope)
            force: Force consolidation even if threshold not met

        Returns:
            New consolidated content, or None if no consolidation occurred
        """
        from eve.agent.memory2.consolidation import consolidate_reflections

        return await consolidate_reflections(
            scope=scope,
            agent_id=self.agent_id,
            user_id=user_id,
            session_id=session_id,
            force=force,
        )

    async def consolidate_all(
        self,
        user_id: Optional[ObjectId] = None,
        session_id: Optional[ObjectId] = None,
        force: bool = False,
    ) -> Dict[str, Optional[str]]:
        """
        Check and consolidate all applicable scopes.

        Args:
            user_id: User ID (optional)
            session_id: Session ID (optional)
            force: Force consolidation even if thresholds not met

        Returns:
            Dict with consolidation results for each scope
        """
        from eve.agent.memory2.consolidation import (
            force_consolidate_all,
            maybe_consolidate_all,
        )

        if force:
            return await force_consolidate_all(
                agent_id=self.agent_id,
                user_id=user_id,
                session_id=session_id,
            )
        else:
            return await maybe_consolidate_all(
                agent_id=self.agent_id,
                user_id=user_id,
                session_id=session_id,
            )

    # -------------------------------------------------------------------------
    # Statistics and Debugging
    # -------------------------------------------------------------------------

    def get_consolidation_status(
        self,
        user_id: Optional[ObjectId] = None,
        session_id: Optional[ObjectId] = None,
    ) -> Dict[str, dict]:
        """
        Get consolidation status for all scopes.

        Returns buffer sizes, thresholds, and whether consolidation is needed.

        Args:
            user_id: User ID (optional)
            session_id: Session ID (optional)

        Returns:
            Dict with status for each scope
        """
        from eve.agent.memory2.consolidation import get_consolidation_status

        return get_consolidation_status(
            agent_id=self.agent_id,
            user_id=user_id,
            session_id=session_id,
        )

    def get_memory_stats(
        self,
        user_id: Optional[ObjectId] = None,
        session_id: Optional[ObjectId] = None,
    ) -> Dict[str, dict]:
        """
        Get statistics about the memory system.

        Returns word counts, reflection counts, and timestamps.

        Args:
            user_id: User ID (optional)
            session_id: Session ID (optional)

        Returns:
            Dict with stats for each scope
        """
        from eve.agent.memory2.context_assembly import get_memory_stats

        return get_memory_stats(
            agent_id=self.agent_id,
            user_id=user_id,
            session_id=session_id,
        )

    # -------------------------------------------------------------------------
    # Session Lifecycle
    # -------------------------------------------------------------------------

    async def on_session_start(
        self,
        session,
        user_id: Optional[ObjectId] = None,
    ) -> str:
        """
        Handle session start - prepare memory context.

        Call this when a new session starts to pre-load memory context.

        Args:
            session: Session object
            user_id: User ID

        Returns:
            Initial memory context
        """
        return await self.get_memory_context(
            session=session,
            user_id=user_id,
            force_refresh=True,
        )

    async def on_session_end(
        self,
        session,
        messages: List,
        user_id: Optional[ObjectId] = None,
    ) -> None:
        """
        Handle session end - finalize memories.

        Call this when a session ends to:
        - Force final memory formation
        - Consolidate any remaining reflections
        - Clean up session-specific data

        Args:
            session: Session object
            messages: List of messages
            user_id: User ID
        """
        try:
            session_id = getattr(session, "id", None)

            # Force final memory formation
            await self.force_form_memories(
                session=session,
                messages=messages,
                user_id=user_id,
            )

            # Force consolidation for all scopes
            await self.consolidate_all(
                user_id=user_id,
                session_id=session_id,
                force=True,
            )

            # Clean up session reflections (optional - depends on retention policy)
            # from eve.agent.memory2.reflection_storage import cleanup_session_reflections
            # await cleanup_session_reflections(session_id)

            if LOCAL_DEV:
                logger.debug(f"Session end cleanup completed for {session_id}")

        except Exception as e:
            logger.error(f"Error in on_session_end: {e}")
            traceback.print_exc()

    async def process_cold_session(
        self,
        session,
        user_id: Optional[ObjectId] = None,
    ) -> bool:
        """
        Process a cold (inactive) session.

        Called by background job when session has been inactive.

        Args:
            session: Session object
            user_id: User ID

        Returns:
            True if memories were formed
        """
        from eve.agent.memory2.formation import process_cold_session

        return await process_cold_session(
            session=session,
            agent_id=self.agent_id,
            user_id=user_id,
        )

    # -------------------------------------------------------------------------
    # Index Management
    # -------------------------------------------------------------------------

    @staticmethod
    def ensure_indexes() -> None:
        """
        Ensure all MongoDB indexes are created.

        Call this on application startup to ensure optimal query performance.
        """
        from eve.agent.memory2.models import (
            ConsolidatedMemory,
            Fact,
            Reflection,
        )

        try:
            Fact.ensure_indexes()
            Reflection.ensure_indexes()
            ConsolidatedMemory.ensure_indexes()

            if LOCAL_DEV:
                logger.debug("Memory2 indexes ensured")

        except Exception as e:
            logger.error(f"Error ensuring indexes: {e}")
            traceback.print_exc()


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

async def get_memory_service(agent_id: ObjectId) -> MemoryService:
    """
    Get a memory service instance for an agent.

    This is a convenience function for getting a service instance.

    Args:
        agent_id: Agent ID

    Returns:
        MemoryService instance
    """
    return MemoryService(agent_id)


async def quick_memory_context(
    agent_id: ObjectId,
    session,
    user_id: Optional[ObjectId] = None,
) -> str:
    """
    Quick helper to get memory context without creating a service instance.

    Args:
        agent_id: Agent ID
        session: Session object
        user_id: User ID

    Returns:
        Memory context XML string
    """
    service = MemoryService(agent_id)
    return await service.get_memory_context(session, user_id)
