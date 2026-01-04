"""
Memory System v2 - Eden Agent Memory Architecture

This module implements a redesigned memory system with two independent subsystems:

1. Always-in-Context Memory (Reflections):
   - Always injected into agent context
   - Contains agent ideas, plans, user preferences
   - Evolves over time via buffer → consolidation
   - Three scopes: session, user, agent

2. RAG Memory (Facts):
   - Stored in vector database (MongoDB Atlas)
   - Retrieved via semantic search
   - Infinitely scalable
   - Two scopes: user, agent (no session scope)

Key concepts:
- Fact: Atomic, objective statement for semantic retrieval
- Reflection: Interpreted memory that evolves agent persona
- Consolidation: Merging buffered reflections into condensed blob
- Scope: Where memory lives (session, user, agent)

Usage:
    from eve.agent.memory2 import MemoryService

    # Create service
    service = MemoryService(agent_id)

    # Get memory context for prompts
    context = await service.get_memory_context(session, user_id)

    # Form memories from conversation
    await service.maybe_form_memories(session, messages, user_id)

    # Search facts via RAG
    facts = await service.search_facts("user preferences", user_id)
"""

# Models
from eve.agent.memory2.models import (
    ConsolidatedMemory,
    ExtractedFact,
    ExtractedReflection,
    Fact,
    FactDecision,
    FactDecisionEvent,
    FactDecisionResponse,
    FactExtractionResponse,
    MemoryScope,
    Reflection,
    ReflectionExtractionResponse,
    get_unabsorbed_reflections,
    mark_reflections_absorbed,
)

# Constants
from eve.agent.memory2.constants import (
    ALWAYS_IN_CONTEXT_ENABLED,
    CONSOLIDATED_WORD_LIMITS,
    CONSOLIDATION_THRESHOLDS,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    FACT_MAX_WORDS,
    MEMORY_LLM_MODEL_FAST,
    MEMORY_LLM_MODEL_SLOW,
    RAG_ENABLED,
    RAG_TOP_K,
    REFLECTION_MAX_WORDS,
    SIMILARITY_THRESHOLD,
)

# Service layer
from eve.agent.memory2.service import (
    MemoryService,
    get_memory_service,
    quick_memory_context,
)

# Formation
from eve.agent.memory2.formation import (
    form_memories,
    maybe_form_memories,
    process_cold_session,
    should_form_memories,
)

# Context assembly
from eve.agent.memory2.context_assembly import (
    assemble_always_in_context_memory,
    clear_memory_cache,
    get_memory_context_for_session,
    get_memory_stats,
)

# Consolidation
from eve.agent.memory2.consolidation import (
    consolidate_reflections,
    force_consolidate_all,
    get_consolidation_status,
    maybe_consolidate_all,
)

# Reflection handling
from eve.agent.memory2.reflection_extraction import (
    extract_and_save_reflections,
    extract_reflections,
)
from eve.agent.memory2.reflection_storage import (
    cleanup_session_reflections,
    get_all_reflections_for_context,
    get_buffer_size,
    save_reflections,
    should_consolidate,
)

# Fact handling
from eve.agent.memory2.fact_extraction import (
    create_fact_document,
    extract_and_prepare_facts,
    extract_facts,
    save_facts,
)
from eve.agent.memory2.fact_storage import (
    check_duplicate_by_hash,
    delete_fact,
    get_embedding,
    get_embeddings_batch,
    get_fact_count,
    get_facts_by_scope,
    store_fact,
    store_facts_batch,
    update_fact,
)
from eve.agent.memory2.fact_management import (
    deduplicate_facts,
    execute_decision,
    llm_memory_update_decision,
    process_extracted_facts,
    search_similar_facts,
)

# RAG
from eve.agent.memory2.rag import (
    format_facts_for_tool_response,
    get_relevant_facts_for_context,
    search_facts,
)
from eve.agent.memory2.rag_tool import (
    MEMORY_SEARCH_TOOL,
    MemorySearchTool,
    build_rag_context_section,
    get_memory_tool,
    get_memory_tool_definition,
    handle_memory_search,
    proactive_memory_retrieval,
)

__all__ = [
    # Service
    "MemoryService",
    "get_memory_service",
    "quick_memory_context",
    # Models
    "Fact",
    "Reflection",
    "ConsolidatedMemory",
    "MemoryScope",
    "FactDecision",
    "FactDecisionEvent",
    "ExtractedFact",
    "ExtractedReflection",
    "FactExtractionResponse",
    "ReflectionExtractionResponse",
    "FactDecisionResponse",
    "get_unabsorbed_reflections",
    "mark_reflections_absorbed",
    # Constants
    "CONSOLIDATION_THRESHOLDS",
    "CONSOLIDATED_WORD_LIMITS",
    "MEMORY_LLM_MODEL_FAST",
    "MEMORY_LLM_MODEL_SLOW",
    "SIMILARITY_THRESHOLD",
    "FACT_MAX_WORDS",
    "REFLECTION_MAX_WORDS",
    "RAG_ENABLED",
    "ALWAYS_IN_CONTEXT_ENABLED",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "RAG_TOP_K",
    # Formation
    "maybe_form_memories",
    "form_memories",
    "should_form_memories",
    "process_cold_session",
    # Context assembly
    "assemble_always_in_context_memory",
    "get_memory_context_for_session",
    "clear_memory_cache",
    "get_memory_stats",
    # Consolidation
    "consolidate_reflections",
    "maybe_consolidate_all",
    "force_consolidate_all",
    "get_consolidation_status",
    # Reflection handling
    "extract_reflections",
    "extract_and_save_reflections",
    "save_reflections",
    "get_buffer_size",
    "should_consolidate",
    "get_all_reflections_for_context",
    "cleanup_session_reflections",
    # Fact handling
    "extract_facts",
    "extract_and_prepare_facts",
    "create_fact_document",
    "save_facts",
    "store_fact",
    "store_facts_batch",
    "update_fact",
    "delete_fact",
    "get_facts_by_scope",
    "get_fact_count",
    "check_duplicate_by_hash",
    "get_embedding",
    "get_embeddings_batch",
    "process_extracted_facts",
    "search_similar_facts",
    "llm_memory_update_decision",
    "execute_decision",
    "deduplicate_facts",
    # RAG
    "search_facts",
    "get_relevant_facts_for_context",
    "format_facts_for_tool_response",
    "MEMORY_SEARCH_TOOL",
    "MemorySearchTool",
    "get_memory_tool",
    "get_memory_tool_definition",
    "handle_memory_search",
    "proactive_memory_retrieval",
    "build_rag_context_section",
]
