# Eden Memory System: Analysis & Redesign Proposal

*Document created: 2025-12-18*
*Purpose: Comprehensive reference for memory system refactoring*

---

## Table of Contents
1. [Current System Overview](#1-current-system-overview)
2. [File-by-File Analysis](#2-file-by-file-analysis)
3. [Current Problems & Challenges](#3-current-problems--challenges)

---

## 1. Current System Overview

### High-Level Architecture

The memory system has **three distinct memory types** that share conceptual similarities but have different implementations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY FORMATION FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Chat Messages                                                               │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ maybe_form_memories() - Trigger checks:                     │            │
│  │   • Message count >= 45 (MEMORY_FORMATION_MSG_INTERVAL)     │            │
│  │   • OR Token count >= 1000 (MEMORY_FORMATION_TOKEN_INTERVAL)│            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ _extract_all_memories()                                      │            │
│  │   ├── LLM Call 1: Regular extraction (episode + directive)  │            │
│  │   └── LLM Call 2-N: Per shard extraction (fact + suggestion)│            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ _save_all_memories() → SessionMemory records                │            │
│  │   ├── _update_user_memory() → maybe consolidate directives  │            │
│  │   └── _update_agent_memory() → maybe consolidate suggestions│            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ assemble_memory_context() → XML for prompt injection        │            │
│  │   ├── _assemble_user_memory()                               │            │
│  │   ├── _assemble_agent_memories()                            │            │
│  │   └── _get_episode_memories()                               │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Types Comparison

| Aspect | Episode (Session) | Directive (User) | Fact (Agent) | Suggestion (Agent) |
|--------|-------------------|------------------|--------------|-------------------|
| **Scope** | Per session | Per agent+user | Per agent+shard | Per agent+shard |
| **Raw Storage** | SessionMemory | SessionMemory | SessionMemory | SessionMemory |
| **Consolidated Storage** | None | UserMemory | N/A (FIFO only) | AgentMemory |
| **Extraction Prompt** | Hardcoded | Hardcoded | Per-shard configurable | Per-shard configurable |
| **Consolidation Prompt** | N/A | Hardcoded | N/A | Per-shard (uses default) |
| **Buffer Strategy** | Session-cached FIFO | Count-triggered consolidation | FIFO with eviction | Count-triggered consolidation |
| **Buffer Limit** | 8 episodes | 5 before consolidation | 100 facts (FIFO) | 16 before consolidation |
| **Max Words per Item** | 50 | 25 | 30 | 35 |
| **Consolidated Max Words** | N/A | 400 | N/A | 1000 |

### Key Insight: Two-Part Memory Pattern

All memory types follow a fundamental pattern:
1. **Consolidated Content**: Evolving summary built from observations, always injected into context
2. **Facts**: Growing list of discrete items (for RAG retrieval)

**Memory Atom Types:**
- **Observation**: Temporary item that accumulates in a buffer until consolidated into a summary blob
- **Fact**: Discrete item stored long-term with RAG indexing for semantic retrieval

| Type | Has Observations? | Has Facts? |
|------|-------------------|------------|
| Session | No (could add) | Yes (session facts) |
| User | Yes (user observations → consolidation) | No (could add) |
| Agent Shard | Yes (shard observations → consolidation) | Yes (shard facts → RAG) |

---

## 2. File-by-File Analysis

### `memory/__init__.py`
**Purpose**: Public exports for the memory module
**Exports**: `MemoryBackend`, `MongoMemoryBackend`, `MemoryService`, `memory_service`

### `memory/memory_models.py` (546 lines)
**Purpose**: Data models and utility functions

**Key Classes:**
- `SessionMemory` (Collection: `memory_sessions`) - Individual memory records
  - Fields: `agent_id`, `source_session_id`, `memory_type`, `content`, `source_message_ids`, `related_users`, `shard_id`, `agent_owner`
  - Indexes: `directive_lookup_idx`, `episode_memories_idx`

- `UserMemory` (Collection: `memory_user`) - Consolidated user memory blob
  - Fields: `agent_id`, `user_id`, `content`, `unabsorbed_memory_ids`, `fully_formed_memory`, `last_updated_at`
  - Index: unique on `(agent_id, user_id)`

- `AgentMemory` (Collection: `memory_agent`) - Agent collective memory shard
  - Fields: `agent_id`, `shard_name`, `extraction_prompt`, `is_active`, `content`, `unabsorbed_memory_ids`, `facts`, `fully_formed_memory`, `last_updated_at`, `max_facts_per_shard`, `max_agent_memories_before_consolidation`
  - Indexes: `agent_id_is_active_idx`, `agent_memory_freshness_idx`

**Key Functions:**
- `messages_to_text()` - Converts ChatMessage list to text with character counts by source
- `calculate_dynamic_limits()` - Adjusts extraction limits based on conversation length
- `select_messages()` - Retrieves messages from session with limit handling
- `_format_memories_with_age()` - Formats memories with age information
- `estimate_tokens()` - Rough token estimation (chars / 4.5)

**Constants/Multipliers:**
- `USER_MULTIPLIER = 1.0`, `TOOL_MULTIPLIER = 0.5`, `AGENT_MULTIPLIER = 0.2`, `OTHER_MULTIPLIER = 0.5`

### `memory/memory_constants.py` (318 lines)
**Purpose**: Configuration constants and prompt templates

**Key Constants:**
```python
# Production values:
MEMORY_LLM_MODEL_FAST = "gpt-5-mini"
MEMORY_LLM_MODEL_SLOW = "gpt-5.1"
MEMORY_FORMATION_MSG_INTERVAL = 45
MEMORY_FORMATION_TOKEN_INTERVAL = 1000
MAX_USER_MEMORIES_BEFORE_CONSOLIDATION = 5
MAX_N_EPISODES_TO_REMEMBER = 8
MAX_AGENT_MEMORIES_BEFORE_CONSOLIDATION = 16
MAX_FACTS_PER_SHARD = 100
CONSIDER_COLD_AFTER_MINUTES = 10
CLEANUP_COLD_SESSIONS_EVERY_MINUTES = 10
SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES = 5
NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = 4

# Word limits:
SESSION_EPISODE_MEMORY_MAX_WORDS = 50
SESSION_DIRECTIVE_MEMORY_MAX_WORDS = 25
SESSION_SUGGESTION_MEMORY_MAX_WORDS = 35
SESSION_FACT_MEMORY_MAX_WORDS = 30
USER_MEMORY_BLOB_MAX_WORDS = 400
AGENT_MEMORY_BLOB_MAX_WORDS = 1000
```

**MemoryType Class:**
```python
class MemoryType:
    name: str
    min_items: int
    max_items: int
    custom_prompt: str
```

**Prompt Templates:**
- `REGULAR_MEMORY_EXTRACTION_PROMPT` - For episode + directive extraction
- `AGENT_MEMORY_EXTRACTION_PROMPT` - For fact + suggestion extraction (per shard)
- `USER_MEMORY_CONSOLIDATION_PROMPT` - For consolidating user directives
- `AGENT_MEMORY_CONSOLIDATION_PROMPT` - For consolidating agent suggestions

**Token System:**
- `-&&-conversation_text-&&-` - Replaced with conversation
- `-&&-shard_extraction_prompt-&&-` - Replaced with shard context
- `-&&-fully_formed_agent_memory-&&-` - Replaced with shard's fully formed memory
- `-&&-directive_min-&&-`, `-&&-directive_max-&&-` - Dynamic limits
- `-&&-fact_min-&&-`, `-&&-fact_max-&&-` - Dynamic limits
- `-&&-suggestion_min-&&-`, `-&&-suggestion_max-&&-` - Dynamic limits

### `memory/memory.py` (1226 lines)
**Purpose**: Core memory formation logic

**Entry Points:**
- `maybe_form_memories(agent_id, session, agent)` - Checks triggers, calls form_memories
- `form_memories(agent_id, session, agent, conversation_text, char_counts)` - Main memory formation

**Key Functions:**
- `should_form_memories()` - Returns (should_form, conversation_text, char_counts)
- `_extract_all_memories()` - Orchestrates extraction (regular + per-shard)
- `extract_memories_with_llm()` - Generic LLM extraction with structured output
- `_save_all_memories()` - Stores SessionMemory records, triggers updates
- `_store_memories_by_type()` - Batch inserts memories
- `_update_user_memory()` - Adds directives, maybe consolidates
- `_update_agent_memory()` - Adds facts (FIFO) and suggestions, maybe consolidates
- `_add_memories_and_maybe_consolidate()` - Generic add + consolidation trigger
- `_consolidate_with_llm()` - Generic LLM consolidation
- `_consolidate_user_directives()` - Specific user consolidation
- `_consolidate_agent_suggestions()` - Specific agent consolidation
- `_load_memories_by_ids()` - Batch loads SessionMemory by IDs
- `_regenerate_fully_formed_user_memory()` - Builds user's fully_formed
- `_regenerate_fully_formed_agent_memory()` - Builds shard's fully_formed
- `safe_update_memory_context()` - Updates session.memory_context safely

### `memory/memory_assemble_context.py` (439 lines)
**Purpose**: Assembles memory context for prompt injection

**Entry Point:**
- `assemble_memory_context(session, agent, user, force_refresh, reason, skip_save, instrumentation)` - Returns XML memory context string

**Key Functions:**
- `_assemble_user_memory()` - Gets user's fully_formed_memory
- `_get_episode_memories()` - Gets cached/fresh episode memories
- `_assemble_agent_memories()` - Gets all active shards' fully_formed_memory
- `check_memory_freshness()` - Checks if cached context is stale
- `_build_memory_xml()` - Constructs XML structure

**Output Format:**
```xml
<MemoryContext description="Your complete memory context for this conversation">
  <CollectiveMemory description="Shared memory across all your conversations">
    <MemoryShard name="shard_name">
      ## Shard facts:
      - fact 1 (age: X days ago)

      ## Current consolidated shard memory:
      [consolidated content]

      ## Recent shard suggestions:
      - suggestion 1
    </MemoryShard>
  </CollectiveMemory>

  <UserMemory description="Memory and context specific to this user">
    -- User Memory for username --
    ## Consolidated user memory:
    [content]
    ## Recent user directives:
    - directive 1 (age: X days ago)
  </UserMemory>

  <CurrentConversationContext description="Recent exchanges from this conversation">
    - episode 1
    - episode 2
  </CurrentConversationContext>
</MemoryContext>
```

### `memory/memory_cold_sessions_processor.py` (258 lines)
**Purpose**: Background processing for cold sessions (Modal deployment)

**Key Function:**
- `process_cold_sessions()` - Finds sessions with no activity for 10+ minutes and triggers memory formation

**Queries:**
1. Sessions with `memory_context.last_activity < cutoff` AND `messages_since_memory_formation >= 4`
2. Sessions without `memory_context` (legacy) via aggregation pipeline

### `memory/service.py` (79 lines)
**Purpose**: High-level facade for memory operations

**Class:**
```python
class MemoryService:
    def __init__(self, backend: MemoryBackend = MongoMemoryBackend()):
        self._backend = backend

    async def assemble_memory_context(...)
    async def maybe_form_memories(...)
    async def form_memories(...)
```

### `memory/backends.py` (168 lines)
**Purpose**: Backend abstraction

**Classes:**
- `MemoryBackend` (ABC) - Abstract interface
- `MongoMemoryBackend` - Concrete implementation (wraps existing functions)

### `memory/memory_update_shard.py` (81 lines)
**Purpose**: Utility script to update shard facts manually

---

## 3. Current Problems & Challenges

### 3.1 Code Duplication
- Similar logic repeated for user vs agent memory (add, consolidate, regenerate)
- `_update_user_memory()` and `_update_agent_memory()` follow same pattern
- `_consolidate_user_directives()` and `_consolidate_agent_suggestions()` are nearly identical
- `_regenerate_fully_formed_user_memory()` and `_regenerate_fully_formed_agent_memory()` share structure

### 3.2 Scattered Configuration
- Constants spread across `memory_constants.py`
- Some values hardcoded in prompts
- Some stored in documents (`AgentMemory.max_facts_per_shard`)
- No single source of truth for a stream's configuration

### 3.3 Inconsistent Abstractions
- Episodes don't have a consolidated blob model (could benefit from session summarization)
- Facts use FIFO without consolidation
- User memory lacks customizable extraction prompts (hardcoded in `REGULAR_MEMORY_EXTRACTION_PROMPT`)
- Agent shards have configurable `extraction_prompt` but not consolidation prompts

### 3.4 Coupled Concerns
- Extraction, storage, consolidation, and context assembly are intertwined in `memory.py`
- Hard to modify one aspect without affecting others
- Testing individual components is difficult

### 3.5 No Unified Interface
- Each memory type has bespoke functions rather than a common interface
- Adding a new memory type requires touching multiple files
- No clear contract for what a "memory stream" should support

### 3.6 LLM Call Efficiency
**Major concern**: Current system requires multiple LLM calls per memory formation:
- 1 call for episode + directive extraction
- N calls for N active agent shards (fact + suggestion each)
- Occasional consolidation calls

With 3 active shards: **4+ LLM calls per formation event**

This is expensive and adds latency.

### 3.7 Prompt Complexity
- Extraction prompts are complex with many inline instructions
- Dynamic limits use token replacement which is fragile
- Hard to customize prompts without understanding the token system

### 3.8 RAG Not Implemented
- Atomic items (facts, episodes) are stored but not vector-indexed
- No semantic search capability
- Currently relies on FIFO recency, not relevance
- **Solution**: Adopt MongoDB Atlas vector search pipeline (see Section 4)