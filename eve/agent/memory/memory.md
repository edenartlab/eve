# Eden Memory System: Analysis & Redesign Proposal

*Document created: 2025-12-18*
*Purpose: Comprehensive reference for memory system refactoring*

---

## Table of Contents
1. [Current System Overview](#1-current-system-overview)
2. [File-by-File Analysis](#2-file-by-file-analysis)
3. [Current Problems & Challenges](#3-current-problems--challenges)
4. [Proposed New Architecture](#4-proposed-new-architecture)
5. [Open Questions & Decisions](#5-open-questions--decisions)
6. [Implementation Plan](#6-implementation-plan)
7. [MongoDB Atlas Setup Requirements](#7-mongodb-atlas-setup-requirements)

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

---

## 4. Proposed New Architecture

### 4.1 Core Concept: MemoryStream

A **MemoryStream** is a unified primitive that handles all memory operations. Each memory type becomes a stream instance with different configuration.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MemoryStream                                 │
├─────────────────────────────────────────────────────────────────────┤
│ IDENTITY & SCOPE                                                     │
│   stream_type: "session" | "user" | "agent"                         │
│   scope: Dict (e.g., {"agent_id": ..., "user_id": ...})             │
│   name: Optional[str] (e.g., shard name)                            │
│   is_active: bool                                                   │
├─────────────────────────────────────────────────────────────────────┤
│ EXTRACTION CONFIG (fully stored in DB)                              │
│   extraction_prompt: str                                            │
│   extraction_fields: ["observation", "fact"]                        │
│   observation_extraction_min/max: int                               │
│   observation_max_words: int                                        │
│   fact_extraction_min/max: int                                      │
│   fact_max_words: int                                               │
├─────────────────────────────────────────────────────────────────────┤
│ CONSOLIDATION CONFIG                                                │
│   consolidation_prompt: Optional[str]                               │
│   consolidation_trigger: int                                        │
│   consolidated_max_words: int                                       │
├─────────────────────────────────────────────────────────────────────┤
│ FACT BUFFER CONFIG                                                  │
│   fact_buffer_max: int (FIFO limit)                                 │
│   fact_rag_enabled: bool (future)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ STATE                                                               │
│   consolidated_content: Optional[str]                               │
│   unabsorbed_observation_ids: List[ObjectId]                        │
│   fact_ids: List[ObjectId] (FIFO buffer)                            │
│   fully_formed: Optional[str]                                       │
│   last_updated_at: datetime                                         │
├─────────────────────────────────────────────────────────────────────┤
│ OPERATIONS                                                          │
│   extract(conversation_text) → Dict[str, List[MemoryAtom]]          │
│   add(atoms) → triggers consolidation if threshold met              │
│   consolidate() → merges unabsorbed observations into consolidated  │
│   assemble() → builds fully_formed context string                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Models

```python
@Collection("memory_streams")
class MemoryStream(Document):
    """Unified memory primitive - replaces UserMemory, AgentMemory, episode handling"""

    # Identity
    stream_type: Literal["session", "user", "agent"]
    scope: Dict[str, Any]   # {"agent_id": ..., "user_id": ...} or {"session_id": ...}
    name: Optional[str] = None
    is_active: bool = True

    # Extraction config (full prompts in DB)
    extraction_prompt: str
    extraction_fields: List[str] = ["observation", "fact"]  # observation=pre-consolidation, fact=long-term RAG
    observation_extraction_min: int = 0
    observation_extraction_max: int = 1
    observation_max_words: int = 50
    fact_extraction_min: int = 0
    fact_extraction_max: int = 5
    fact_max_words: int = 30

    # Consolidation config
    consolidation_prompt: Optional[str] = None
    consolidation_trigger: int = 10
    consolidated_max_words: int = 500

    # Fact buffer config (long-term RAG storage)
    fact_buffer_max: int = 100
    fact_rag_enabled: bool = False

    # State
    consolidated_content: Optional[str] = None
    unabsorbed_observation_ids: List[ObjectId] = []  # Observations awaiting consolidation
    fact_ids: List[ObjectId] = []  # Long-term facts for RAG retrieval
    fully_formed: Optional[str] = None
    last_updated_at: Optional[datetime] = None

    # Optional LLM config override
    extraction_model: Optional[str] = None
    consolidation_model: Optional[str] = None


@Collection("memory_atoms")
class MemoryAtom(Document):
    """Single extracted memory unit - replaces SessionMemory"""

    stream_id: ObjectId
    atom_type: Literal["observation", "fact"]  # observation=pre-consolidation buffer, fact=long-term RAG storage
    content: str

    # Temporal (CRITICAL: every memory must have formation timestamp)
    formed_at: datetime  # When this memory was extracted/formed
    created_at: datetime = datetime.now(timezone.utc)

    # Access tracking (for future memory decay/importance scoring)
    access_count: int = 0  # Incremented each time this memory is retrieved via RAG
    last_accessed_at: Optional[datetime] = None  # Updated on RAG retrieval

    # Provenance
    source_session_id: ObjectId
    source_message_ids: List[ObjectId] = []
    related_users: List[ObjectId] = []

    # RAG support (for long-term storage atoms)
    embedding: Optional[List[float]] = None  # 1536-dim vector for text-embedding-3-small
    embedding_model: Optional[str] = None
    rag_indexed: bool = False  # Whether this atom is searchable via RAG
```

### 4.3 Stream Type Mappings

| Current Type | Stream Config |
|-------------|---------------|
| **Session Episode** | `stream_type="session"`, `extraction_fields=["fact"]` (or `["observation", "fact"]` if adding session summaries), `consolidation_prompt=None` or session summary prompt, scope=`{session_id}` |
| **User Directive** | `stream_type="user"`, `extraction_fields=["observation"]`, `consolidation_prompt=USER_PROMPT`, scope=`{agent_id, user_id}` |
| **Agent Shard** | `stream_type="agent"`, `extraction_fields=["observation", "fact"]`, `consolidation_prompt=AGENT_PROMPT`, scope=`{agent_id, shard_name}` |

### 4.4 Operations Interface

```python
class MemoryStreamOps:
    """Operations for a MemoryStream"""

    def __init__(self, stream: MemoryStream):
        self.stream = stream

    async def extract(
        self,
        conversation_text: str,
        char_counts: dict
    ) -> Dict[str, List[MemoryAtom]]:
        """Extract atoms from conversation using stream's extraction_prompt"""
        prompt = self._prepare_prompt(conversation_text)
        response = await llm_extract(prompt, self.stream.extraction_fields)
        return {"observation": [...], "fact": [...]}

    async def add(self, atoms: Dict[str, List[MemoryAtom]]) -> None:
        """Add atoms to stream, triggering consolidation if needed"""
        # Add observations to unabsorbed list (awaiting consolidation)
        for atom in atoms.get("observation", []):
            atom.save()
            self.stream.unabsorbed_observation_ids.append(atom.id)

        # Check consolidation trigger
        if len(self.stream.unabsorbed_observation_ids) >= self.stream.consolidation_trigger:
            await self.consolidate()

        # Add facts to fact_ids (FIFO, long-term RAG storage)
        for atom in atoms.get("fact", []):
            atom.save()
            self.stream.fact_ids.append(atom.id)

        # FIFO eviction for facts
        if len(self.stream.fact_ids) > self.stream.fact_buffer_max:
            self.stream.fact_ids = self.stream.fact_ids[-self.stream.fact_buffer_max:]

        await self.regenerate_fully_formed()
        self.stream.save()

    async def consolidate(self) -> None:
        """
        Merge unabsorbed observations into consolidated_content.

        ERROR HANDLING STRATEGY:
        - NEVER remove observations from unabsorbed_observation_ids until consolidated_content is saved
        - If consolidation fails, observations remain in unabsorbed list (now 1+ over threshold)
        - Next memory formation will trigger consolidation again
        - This ensures we never lose memories even if LLM call fails
        """
        if not self.stream.consolidation_prompt:
            return

        unabsorbed = MemoryAtom.find({"_id": {"$in": self.stream.unabsorbed_observation_ids}})

        try:
            new_content = await llm_consolidate(
                self.stream.consolidation_prompt,
                self.stream.consolidated_content,
                unabsorbed,
                max_words=self.stream.consolidated_max_words
            )

            # CRITICAL: Only clear unabsorbed list AFTER successful save
            # This is atomic - if save fails, unabsorbed_observation_ids remain intact
            self.stream.consolidated_content = new_content
            self.stream.unabsorbed_observation_ids = []
            self.stream.save()

        except Exception as e:
            # Consolidation failed - DO NOT clear unabsorbed_observation_ids
            # They will be retried on next memory formation
            logger.error(f"Consolidation failed for stream {self.stream.id}: {e}")
            # Stream state unchanged, observations safe in unabsorbed list

    async def regenerate_fully_formed(self) -> None:
        """Build fully_formed string from current state"""
        parts = []

        if self.stream.fact_ids:
            facts = MemoryAtom.find({"_id": {"$in": self.stream.fact_ids}})
            parts.append(self._format_facts(facts))

        if self.stream.consolidated_content:
            parts.append(f"## Memory:\n{self.stream.consolidated_content}")

        if self.stream.unabsorbed_observation_ids:
            observations = MemoryAtom.find({"_id": {"$in": self.stream.unabsorbed_observation_ids}})
            parts.append(self._format_observations(observations))

        self.stream.fully_formed = "\n\n".join(parts)
        self.stream.last_updated_at = datetime.now(timezone.utc)
```

### 4.5 MongoDB RAG Pipeline Integration

The RAG system is adopted from the MongoDB-RAG-Agent pattern and provides semantic retrieval for long-term memory storage.

#### 4.5.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY RETRIEVAL FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AUTOMATIC (every user message):                                            │
│  ───────────────────────────────                                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ STEP 1: RAG Query Generation (Fast LLM Call)                 │            │
│  │                                                              │            │
│  │   User message arrives                                       │            │
│  │         │                                                    │            │
│  │         ▼                                                    │            │
│  │   A single, fast, cheap LLM call (e.g., gpt-4o-mini):        │            │
│  │     INPUT:  Last N conversation messages + new user message  │            │
│  │     OUTPUT: Query string for memory retrieval                │            │
│  │             (or empty string if no retrieval needed)         │            │
│  │                                                              │            │
│  │   This is invisible to the main agent.                       │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ STEP 2: RAG Retrieval (if query string is non-empty)         │            │
│  │   ├── Query embedding generation (OpenAI text-embedding-3)   │            │
│  │   ├── Hybrid search (semantic + keyword via RRF)             │            │
│  │   ├── Filter by stream scope (user/collective)               │            │
│  │   └── Update access_count & last_accessed_at on results      │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ STEP 3: Context Assembly                                     │            │
│  │   ├── Always-in-Context Layer (Static):                      │            │
│  │   │     ├── Consolidated memory blobs (fully_formed)         │            │
│  │   │     └── Recent unabsorbed observations                   │            │
│  │   │                                                          │            │
│  │   └── RAG Results Layer (Dynamic):                           │            │
│  │         └── Retrieved memories from hybrid search            │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  Complete Memory Context (XML) injected into main agent prompt              │
│       │                                                                      │
│       ▼                                                                      │
│  Main agent responds (unaware that RAG query generation ran)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Design Decision**: RAG query generation is handled by a **separate, fast LLM call that is invisible to the main agent**:
- A fast, cheap LLM (e.g., gpt-4o-mini) examines the last N messages and generates a query string
- The main agent never sees or knows about this query generation step
- If the fast LLM determines no retrieval is needed, it returns an empty string and RAG is skipped
- Retrieved memories are seamlessly added to the main agent's context before it responds
- This keeps RAG automatic without burdening the main agent with tool-calling decisions

#### 4.5.2 MongoDB Collections & Indexes

**`memory_atoms` Collection Schema:**
```python
{
  "_id": ObjectId,
  "stream_id": ObjectId,           # Foreign key to memory_streams
  "atom_type": "observation" | "fact",  # observation=pre-consolidation, fact=long-term RAG
  "content": str,                  # The memory text
  "embedding": [float, ...],       # 1536-dim vector (MUST be native array, NOT string)
  "formed_at": datetime,           # When memory was extracted (CRITICAL)
  "created_at": datetime,
  "source_session_id": ObjectId,
  "source_message_ids": [ObjectId],
  "related_users": [ObjectId],
  "embedding_model": str,
  "rag_indexed": bool
}
```

**Required MongoDB Atlas Search Indexes:**

1. **Vector Search Index** (`memory_vector_index`):
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "stream_id"
    },
    {
      "type": "filter",
      "path": "rag_indexed"
    }
  ]
}
```

2. **Text Search Index** (`memory_text_index`):
```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {
        "type": "string",
        "analyzer": "lucene.standard"
      }
    }
  }
}
```

#### 4.5.3 Embedding Generation

```python
class MemoryEmbedder:
    """Generates embeddings for memory atoms."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ):
        self.model = model
        self.batch_size = batch_size
        self.dimensions = 1536
        self.client = openai.AsyncOpenAI()

    async def embed_atoms(
        self,
        atoms: List[MemoryAtom]
    ) -> List[MemoryAtom]:
        """Generate embeddings for atoms in batches."""
        for i in range(0, len(atoms), self.batch_size):
            batch = atoms[i:i + self.batch_size]
            texts = [atom.content for atom in batch]

            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            for atom, embedding_data in zip(batch, response.data):
                atom.embedding = embedding_data.embedding  # Native list[float]
                atom.embedding_model = self.model
                atom.rag_indexed = True

        return atoms
```

#### 4.5.4 Hybrid Search with Reciprocal Rank Fusion

```python
async def search_memories(
    query: str,
    stream_ids: List[ObjectId],       # Filter by relevant streams
    match_count: int = 10,
    formed_after: Optional[datetime] = None,  # Temporal filtering
    search_type: str = "hybrid"       # "hybrid", "semantic", or "text"
) -> List[MemorySearchResult]:
    """
    Search memories using MongoDB Atlas hybrid search.

    Combines:
    - Semantic search: vector similarity on embeddings
    - Text search: keyword matching with fuzzy support
    - Reciprocal Rank Fusion: merge rankings without score normalization
    """

    if search_type == "hybrid":
        # Run both searches concurrently
        semantic_results, text_results = await asyncio.gather(
            _semantic_search(query, stream_ids, match_count * 2, formed_after),
            _text_search(query, stream_ids, match_count * 2, formed_after)
        )

        # Merge with RRF (k=60 is industry standard)
        return _reciprocal_rank_fusion(
            [semantic_results, text_results],
            k=60,
            limit=match_count
        )
    elif search_type == "semantic":
        return await _semantic_search(query, stream_ids, match_count, formed_after)
    else:
        return await _text_search(query, stream_ids, match_count, formed_after)


async def _semantic_search(
    query: str,
    stream_ids: List[ObjectId],
    match_count: int,
    formed_after: Optional[datetime]
) -> List[MemorySearchResult]:
    """MongoDB $vectorSearch aggregation."""

    query_embedding = await get_embedding(query)

    # Build filter for stream scope
    filter_conditions = {
        "stream_id": {"$in": stream_ids},
        "rag_indexed": True
    }
    if formed_after:
        filter_conditions["formed_at"] = {"$gte": formed_after}

    pipeline = [
        {
            "$vectorSearch": {
                "index": "memory_vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": match_count * 10,  # 10x for quality
                "limit": match_count,
                "filter": filter_conditions
            }
        },
        {
            "$project": {
                "content": 1,
                "stream_id": 1,
                "formed_at": 1,
                "similarity": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    collection = db.memory_atoms
    return [doc async for doc in collection.aggregate(pipeline)]


def _reciprocal_rank_fusion(
    result_lists: List[List[MemorySearchResult]],
    k: int = 60,
    limit: int = 10
) -> List[MemorySearchResult]:
    """
    Merge ranked lists using RRF.

    Formula: RRF_score(d) = Σ(1 / (k + rank))

    Benefits:
    - Scale-independent: works with different scoring systems
    - No normalization needed
    - k=60 is battle-tested across datasets
    """
    rrf_scores: Dict[str, float] = {}
    atom_map: Dict[str, MemorySearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            atom_id = str(result["_id"])
            rrf_score = 1.0 / (k + rank)

            if atom_id in rrf_scores:
                rrf_scores[atom_id] += rrf_score
            else:
                rrf_scores[atom_id] = rrf_score
                atom_map[atom_id] = result

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [atom_map[atom_id] for atom_id, _ in sorted_ids[:limit]]
```

#### 4.5.5 Memory Flow: Extraction → Storage → Retrieval

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          UNIFIED MEMORY FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: EXTRACTION (Single LLM Call)                                       │
│  ─────────────────────────────────────                                       │
│  Conversation messages → Single extraction prompt with sub-prompts:          │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ <extraction_prompt>                                            │          │
│  │   <session_memories>                                           │          │
│  │     Extract session episodes (temporary, for current context)  │          │
│  │   </session_memories>                                          │          │
│  │                                                                │          │
│  │   <user_memories>                                              │          │
│  │     Extract user directives (long-term, for RAG)               │          │
│  │   </user_memories>                                             │          │
│  │                                                                │          │
│  │   <collective_memories>                                        │          │
│  │     For each active shard:                                     │          │
│  │       Extract facts (long-term, for RAG)                       │          │
│  │       Extract suggestions (for consolidation)                  │          │
│  │   </collective_memories>                                       │          │
│  │ </extraction_prompt>                                           │          │
│  └───────────────────────────────────────────────────────────────┘          │
│       │                                                                      │
│       ▼                                                                      │
│  STEP 2: ROUTING (Based on memory_type and atom_type)                       │
│  ─────────────────────────────────────────────────────                       │
│  Each extracted memory has:                                                  │
│    - memory_type: session | user | collective                               │
│    - atom_type: observation | fact                                          │
│    - formed_at: timestamp (ALWAYS present)                                  │
│    - access_count: 0 (initialized)                                          │
│    - last_accessed_at: None (until first RAG retrieval)                     │
│                                                                              │
│  ┌────────────────┬────────────────┬─────────────────────────────┐          │
│  │ Memory Type    │ Atom Type      │ Destination                 │          │
│  ├────────────────┼────────────────┼─────────────────────────────┤          │
│  │ session        │ fact           │ Temporary: recent context   │          │
│  │ user           │ observation    │ Pre-consolidation buffer    │          │
│  │ collective     │ observation    │ Pre-consolidation buffer    │          │
│  │ collective     │ fact           │ Long-term: RAG-indexed      │          │
│  └────────────────┴────────────────┴─────────────────────────────┘          │
│       │                                                                      │
│       ▼                                                                      │
│  STEP 3: STORAGE & EMBEDDING                                                │
│  ────────────────────────────                                               │
│  For facts (long-term RAG storage):                                         │
│    1. Generate embedding (text-embedding-3-small)                           │
│    2. Store in memory_atoms with embedding                                  │
│    3. Add to stream's fact_ids list (FIFO)                                  │
│                                                                              │
│  For observations (pre-consolidation buffer):                               │
│    1. Store in memory_atoms (no embedding needed)                           │
│    2. Add to stream's unabsorbed_observation_ids list                       │
│    3. Trigger consolidation if threshold met (error-safe: never lose obs)   │
│       │                                                                      │
│       ▼                                                                      │
│  STEP 4: CONTEXT ASSEMBLY (Automatic, on each message)                      │
│  ─────────────────────────────────────────────────────                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ Always-in-Context (STATIC):                                 │            │
│  │   ├── Session: recent facts (FIFO, no embedding)            │            │
│  │   ├── User: consolidated blob + recent observations         │            │
│  │   └── Collective: consolidated blobs + recent observations  │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  STEP 5: RAG RETRIEVAL (On-demand, agent tool call)                         │
│  ──────────────────────────────────────────────────                         │
│  Agent calls search_long_term_memory() when needed:                         │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ search_long_term_memory(query, memory_types, match_count)   │            │
│  │   1. Embed query                                            │            │
│  │   2. Hybrid search (semantic + text + RRF)                  │            │
│  │   3. Update access_count & last_accessed_at on results      │            │
│  │   4. Return formatted results to agent                      │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.5.6 Automatic RAG via Query Generation LLM

**Important**: RAG retrieval is automatic on every user message, but the query generation is handled by a **separate, fast LLM call** that runs before the main agent sees the message. The main agent is completely unaware of this step.

**How it works:**

1. **User message arrives** - before the main agent processes it
2. **Fast LLM generates query** - a cheap, fast model (e.g., gpt-4o-mini) examines:
   - The last N conversation messages (for context)
   - The new user message
   - Outputs a query string optimized for memory retrieval (or empty if no retrieval needed)
3. **RAG executes** - if query is non-empty, hybrid search retrieves relevant memories
4. **Context assembled** - retrieved memories are added to the main agent's context
5. **Main agent responds** - sees enriched context, unaware of the RAG mechanics

**Benefits of this approach:**
- Main agent doesn't need to decide when to search—it's automatic
- Fast LLM can intelligently skip RAG when not needed (returns empty query)
- Query is optimized for retrieval, not constrained by agent's tool-calling format
- Reduces main agent complexity and token usage
- Low latency: fast LLM + parallel embedding generation

```python
# RAG query generation happens BEFORE the main agent runs:

RAG_QUERY_GENERATION_PROMPT = """
You are a memory retrieval assistant. Given a conversation and a new user message,
generate a search query to retrieve relevant memories from long-term storage.

CONVERSATION (last {N} messages):
{conversation_context}

NEW USER MESSAGE:
{user_message}

INSTRUCTIONS:
- If the user is asking about past events, preferences, or context that might be in memory, generate a search query
- If the conversation is casual/simple and doesn't need memory retrieval, return an empty string
- The query should be optimized for semantic search (natural language, not keywords)
- Keep the query concise (1-2 sentences max)

OUTPUT: Return ONLY the query string, or empty string if no retrieval needed.
"""

async def generate_rag_query(
    conversation_messages: List[ChatMessage],
    user_message: str,
    n_context_messages: int = 10
) -> str:
    """
    Generate a RAG query using a fast, cheap LLM.
    Returns empty string if no retrieval is needed.
    """
    # Get last N messages for context
    context = conversation_messages[-n_context_messages:]
    context_text = messages_to_text(context)

    prompt = RAG_QUERY_GENERATION_PROMPT.format(
        N=n_context_messages,
        conversation_context=context_text,
        user_message=user_message
    )

    # Use fast, cheap model for query generation
    response = await llm_call(
        model="gpt-4o-mini",  # Fast and cheap
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    return response.strip()


async def retrieve_memories_for_message(
    session: Session,
    agent: Agent,
    user: User,
    user_message: str
) -> List[MemorySearchResult]:
    """
    Main entry point: generates query and retrieves memories.
    Called automatically before main agent processes the message.
    """
    # Step 1: Generate query with fast LLM
    query = await generate_rag_query(
        conversation_messages=session.messages,
        user_message=user_message
    )

    # Step 2: If no query needed, skip retrieval
    if not query:
        return []

    # Step 3: Get relevant stream IDs
    stream_ids = await _get_stream_ids_for_search(
        agent_id=agent.id,
        user_id=user.id,
        memory_types=["user", "collective"]
    )

    # Step 4: Perform hybrid search
    results = await search_memories(
        query=query,
        stream_ids=stream_ids,
        match_count=5,
        search_type="hybrid"
    )

    # Step 5: Update access tracking
    if results:
        atom_ids = [r["_id"] for r in results]
        await MemoryAtom.update_many(
            {"_id": {"$in": atom_ids}},
            {
                "$inc": {"access_count": 1},
                "$set": {"last_accessed_at": datetime.now(timezone.utc)}
            }
        )

    return results
```

#### 4.5.7 Context Assembly (With Automatic RAG)

Context assembly combines the static "always-in-context" layer with dynamically retrieved RAG results. The RAG retrieval happens automatically before context assembly via the fast LLM query generation (see 4.5.6).

```python
async def assemble_memory_context(
    session: Session,
    agent: Agent,
    user: User,
    user_message: str,
) -> str:
    """
    Assemble memory context for prompt injection.

    This provides BOTH layers:
    1. STATIC always-in-context layer:
       - Session: recent episodes (FIFO)
       - User: consolidated blob + recent unabsorbed directives
       - Collective: consolidated blobs per shard + recent suggestions

    2. DYNAMIC RAG layer (automatic, invisible to main agent):
       - Fast LLM generates query from conversation context
       - Hybrid search retrieves relevant memories
       - Results included in context
    """

    # Get all relevant streams (static layer)
    session_stream = await get_stream(stream_type="session", scope={"session_id": session.id})
    user_stream = await get_stream(stream_type="user", scope={"agent_id": agent.id, "user_id": user.id})
    agent_streams = await get_streams(stream_type="agent", scope={"agent_id": agent.id}, is_active=True)

    # Automatic RAG retrieval (dynamic layer)
    # This runs a fast LLM to generate the query - invisible to main agent
    rag_results = await retrieve_memories_for_message(
        session=session,
        agent=agent,
        user=user,
        user_message=user_message
    )

    # Build XML context with both layers
    return _build_memory_xml(
        session_stream=session_stream,
        user_stream=user_stream,
        agent_streams=agent_streams,
        rag_results=rag_results
    )


def _build_memory_xml(
    session_stream: MemoryStream,
    user_stream: MemoryStream,
    agent_streams: List[MemoryStream],
    rag_results: List[MemorySearchResult],
) -> str:
    """Build XML memory context with static and dynamic layers."""

    parts = ['<MemoryContext>']

    # Collective/Agent memories (static)
    parts.append('  <CollectiveMemory>')
    for stream in agent_streams:
        parts.append(f'    <MemoryShard name="{stream.name}">')
        parts.append(f'      {stream.fully_formed}')
        parts.append('    </MemoryShard>')
    parts.append('  </CollectiveMemory>')

    # User memory (static)
    parts.append('  <UserMemory>')
    parts.append(f'    {user_stream.fully_formed}')
    parts.append('  </UserMemory>')

    # Session context (static)
    parts.append('  <CurrentConversation>')
    parts.append(f'    {session_stream.fully_formed}')
    parts.append('  </CurrentConversation>')

    # RAG results (dynamic - retrieved based on current message)
    if rag_results:
        parts.append('  <RetrievedMemories description="Relevant memories retrieved for this message">')
        for result in rag_results:
            age = _format_age(result.get("formed_at"))
            parts.append(f'    - {result["content"]} ({age})')
        parts.append('  </RetrievedMemories>')

    parts.append('</MemoryContext>')
    return '\n'.join(parts)
```

### 4.6 Cross-Session Memory Synchronization

When an agent is interacting with multiple users in parallel sessions, memories formed in one session need to propagate to other sessions. The current system uses a simple time-based refresh strategy.

#### 4.6.1 Current Implementation (to preserve)

From `memory_assemble_context.py`:
```python
SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES = 5

# In assemble_memory_context():
if session.memory_context.cached_memory_context is not None and not force_refresh:
    time_since_context_refresh = datetime.now(timezone.utc) - memory_timestamp

    if time_since_context_refresh < timedelta(minutes=SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES):
        return session.memory_context.cached_memory_context  # Use cache
    else:
        reason = "syncing_session_memories"  # Force refresh to sync
```

This approach:
- Sessions cache their assembled memory context
- Cache is valid for 5 minutes
- After 5 minutes, context is rebuilt to pick up any updates from other sessions
- Simple, low-overhead, works well for typical conversation cadence

#### 4.6.2 New System Integration

The MemoryStream architecture should preserve this pattern:

```python
class Session:
    memory_context: MemoryContext  # Existing field

class MemoryContext:
    # Existing cache fields
    cached_memory_context: Optional[str] = None
    memory_context_timestamp: Optional[datetime] = None

    # Stream-specific timestamps for targeted refresh
    stream_timestamps: Dict[str, datetime] = {}  # {stream_id: last_fetched_at}

async def assemble_memory_context(session, agent, user):
    """
    Assembles memory context with cross-session sync support.

    Sync Strategy:
    1. Use cached context if < SYNC_INTERVAL old
    2. After SYNC_INTERVAL, check if any relevant streams have updated
    3. Only rebuild sections that have changed
    """

    if session.memory_context.cached_memory_context and not force_refresh:
        age = datetime.now(timezone.utc) - session.memory_context.memory_context_timestamp

        if age < timedelta(minutes=SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES):
            return session.memory_context.cached_memory_context

    # Rebuild context (existing logic)
    ...
```

#### 4.6.3 Freshness Check (Optional Enhancement)

The current system has a disabled `check_memory_freshness()` function that compares timestamps. This can be enabled for more aggressive sync when needed:

```python
async def check_memory_freshness(session: Session, streams: List[MemoryStream]) -> bool:
    """
    Returns True if all relevant streams are older than session's cached timestamps.
    Returns False if any stream has been updated since last cache.
    """
    for stream in streams:
        cached_ts = session.memory_context.stream_timestamps.get(str(stream.id))
        if cached_ts and stream.last_updated_at > cached_ts:
            return False  # Stream was updated, cache is stale
    return True
```

This is currently disabled (line 356 in current code: `if 1:`) because the simple 5-minute refresh works well and avoids extra queries on every message.

### 4.8 Proposed File Structure

```
eve/agent/memory2/              # New folder alongside existing memory/
├── __init__.py                 # Public API exports
├── models.py                   # MemoryStream, MemoryAtom (with access_count, last_accessed_at)
├── ops.py                      # MemoryStreamOps class
├── constants.py                # Default values, LLM models, SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES
├── prompts.py                  # Default prompt templates (including RAG_QUERY_GENERATION_PROMPT)
├── context.py                  # assemble_memory_context() - static + dynamic RAG layer
├── query_generation.py         # generate_rag_query(), retrieve_memories_for_message()
│                               # Fast LLM call to generate RAG query (invisible to main agent)
├── formation.py                # maybe_form_memories(), form_memories()
├── extraction.py               # LLM extraction logic (single batched call)
├── consolidation.py            # LLM consolidation logic (error-safe, never loses memories)
├── embedder.py                 # MemoryEmbedder class for embedding generation
├── rag.py                      # search_memories(), hybrid search, RRF, access tracking
├── migration.py                # One-time migration script
├── service.py                  # High-level MemoryService facade
└── backends.py                 # Backend abstraction
```

---

## 5. Open Questions & Decisions

### 5.1 LLM Call Efficiency (DECIDED)

**Problem**: Multiple LLM calls per memory formation is expensive.

**Decision**: **Option A - Batched Multi-Stream Extraction**

Single LLM call extracts for ALL active memory types at once. The prompt includes sub-prompts for each memory type that is active (session, user, collective).

```python
UNIFIED_EXTRACTION_PROMPT = """
Extract memories from this conversation for multiple memory types.

<conversation>{conversation_text}</conversation>

{# Only include sub-prompts for active memory types #}
{% if session_active %}
<session_memories>
Extract 1-2 session facts summarizing key events (≤50 words each).
These are temporary facts for current conversation context only.
</session_memories>
{% endif %}

{% if user_active %}
<user_memories>
Extract 0-4 user observations about preferences, instructions, or context (≤25 words each).
These observations will be consolidated into the user's long-term memory.
</user_memories>
{% endif %}

{% if collective_active %}
<collective_memories>
{% for shard in active_shards %}
<shard name="{shard.name}">
<context>{shard.extraction_prompt}</context>
<instructions>
  - facts: 0-3 factual items (≤30 words, long-term RAG storage)
  - observations: 0-2 behavioral observations (≤35 words, for consolidation)
</instructions>
</shard>
{% endfor %}
</collective_memories>
{% endif %}

Return JSON with formed_at timestamp for each memory:
{
  "session": [{"content": "...", "formed_at": "ISO8601", "atom_type": "fact"}],
  "user": [{"content": "...", "formed_at": "ISO8601", "atom_type": "observation"}],
  "collective": {
    "shard_name": {
      "facts": [{"content": "...", "formed_at": "ISO8601", "atom_type": "fact"}],
      "observations": [{"content": "...", "formed_at": "ISO8601", "atom_type": "observation"}]
    }
  }
}
"""
```

**Benefits**:
- Single LLM call regardless of active stream count
- 75-90% cost reduction compared to separate calls
- All memories extracted with consistent timestamps
- Sub-prompts can be conditionally included based on active memory types

### 5.2 Session Consolidation

**Decision Made**: Yes, add session consolidation (session summary blob).

This means session streams will have:
- `extraction_fields=["observation", "fact"]`
- `consolidation_prompt` for session summary
- `fact_ids` for recent session facts
- `consolidated_content` for evolving session summary

### 5.3 Migration Strategy

**Decision Made**: Hack migration approach:
- Jam existing `UserMemory.content` into stream's `consolidated_content`
- Jam existing `AgentMemory.content` into stream's `consolidated_content`
- Copy `AgentMemory.facts` to stream's `fact_ids`
- Discard episodes (ephemeral anyway)

### 5.4 Prompt Storage

**Decision Made**: Full prompts stored in DB (not templates with parameters).

This provides maximum flexibility but requires careful prompt management.

---

## 6. Implementation Plan

This implementation follows a **staged approach** that enables incremental development and testing. Each phase produces a working component that can be tested independently before moving to the next phase.

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION PHASES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Memory Formation Foundation                                        │
│  ─────────────────────────────────────                                       │
│  Create memory2/ folder with core models and maybe_form_memories()           │
│  Output: Working memory extraction and storage (NO RAG)                      │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│  Phase 2: Static Context Assembly                                            │
│  ────────────────────────────────                                            │
│  Create assemble_memory_context() returning full memory context              │
│  Output: Fast context assembly without RAG (facts excluded from context)     │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│  Phase 3: RAG Storage Infrastructure                                         │
│  ───────────────────────────────────                                         │
│  Add embedding generation and MongoDB models for RAG-enabled memories        │
│  Output: Memories stored with embeddings, MongoDB indexes created            │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│  Phase 4: RAG Retrieval Functions                                            │
│  ────────────────────────────────                                            │
│  Create query generation (cheap LLM) and hybrid search pipeline              │
│  Output: Working RAG retrieval returning relevant memories                   │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│  Phase 5: RAG Integration                                                    │
│  ────────────────────────────                                                │
│  Integrate RAG retrieval into assemble_memory_context()                      │
│  Output: Full memory context with dynamic RAG results                        │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│  Phase 6: Migration                                                          │
│  ──────────────────                                                          │
│  Migrate existing MongoDB documents to new architecture                      │
│  Output: All existing memories converted to new system                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Memory Formation Foundation

**Goal**: Create the `memory2/` folder with all new memory formation mechanisms that extract memories from chat and store them in MongoDB. This phase produces a working `maybe_form_memories()` function.

**Scope**: Memory extraction and storage ONLY. No RAG, no embedding generation, no retrieval.

**Files to Create**:

```
eve/agent/memory2/
├── __init__.py                 # Public exports
├── models.py                   # MemoryStream, MemoryAtom document models
├── constants.py                # Configuration values, LLM models, thresholds
├── prompts.py                  # Extraction and consolidation prompt templates
├── extraction.py               # LLM extraction logic (single batched call)
├── consolidation.py            # LLM consolidation logic (error-safe)
├── formation.py                # maybe_form_memories(), form_memories()
└── ops.py                      # MemoryStreamOps class (stream operations)
```

**Deliverables**:

1. **`models.py`** - Core document models:
   ```python
   class MemoryStream(Document):
       """Unified memory primitive - replaces UserMemory, AgentMemory"""
       stream_type: Literal["session", "user", "agent"]
       scope: Dict[str, Any]  # {"agent_id": ..., "user_id": ...}
       name: Optional[str] = None
       is_active: bool = True

       # Extraction config
       extraction_prompt: str
       extraction_fields: List[str] = ["observation", "fact"]
       # ... other config fields

       # State (NO embedding fields in this phase)
       consolidated_content: Optional[str] = None
       unabsorbed_observation_ids: List[ObjectId] = []
       fact_ids: List[ObjectId] = []  # Stored but not RAG-indexed yet
       fully_formed: Optional[str] = None
       last_updated_at: Optional[datetime] = None

   class MemoryAtom(Document):
       """Single extracted memory unit"""
       stream_id: ObjectId
       atom_type: Literal["observation", "fact"]
       content: str
       formed_at: datetime
       # ... provenance fields

       # RAG fields (defined but NOT populated in this phase)
       embedding: Optional[List[float]] = None
       rag_indexed: bool = False
   ```

2. **`extraction.py`** - Unified extraction with single LLM call:
   - `extract_memories_with_llm()` - handles all memory types in one call
   - Conditional sub-prompts for session/user/collective
   - Returns structured JSON with `formed_at` timestamps

3. **`consolidation.py`** - Error-safe consolidation:
   - `consolidate_stream()` - merges observations into consolidated blob
   - **Critical**: NEVER remove observations until consolidated_content is saved
   - On failure, observations remain in buffer for retry

4. **`formation.py`** - Entry points:
   - `maybe_form_memories(agent_id, session, agent)` - checks triggers, calls form_memories
   - `form_memories()` - orchestrates extraction → storage → consolidation

5. **`ops.py`** - Stream operations:
   - `MemoryStreamOps.extract()` - extract memories from conversation
   - `MemoryStreamOps.add()` - add atoms, trigger consolidation if needed
   - `MemoryStreamOps.consolidate()` - merge observations
   - `MemoryStreamOps.regenerate_fully_formed()` - rebuild fully_formed string

**Testing Checkpoint**:
- [ ] Can create MemoryStream and MemoryAtom documents
- [ ] Can extract memories from a conversation using single LLM call
- [ ] Can store extracted memories as MemoryAtom documents
- [ ] Can consolidate observations into consolidated_content
- [ ] Consolidation is error-safe (no memory loss on failure)
- [ ] `maybe_form_memories()` triggers correctly based on message/token thresholds

---

### Phase 2: Static Context Assembly

**Goal**: Create the `assemble_memory_context()` function that runs all needed steps and returns the agent's full memory context as fast as possible. This phase does NOT include RAG retrieval.

**Scope**: Context assembly with static content only. Facts are stored but NOT included in context (they will be retrieved via RAG in later phases).

**Files to Create/Modify**:

```
eve/agent/memory2/
├── context.py                  # assemble_memory_context() - static layer only
└── (modify) ops.py             # Add assembly helpers
```

**Deliverables**:

1. **`context.py`** - Static context assembly:
   ```python
   async def assemble_memory_context(
       session: Session,
       agent: Agent,
       user: User,
       force_refresh: bool = False,
   ) -> str:
       """
       Assemble memory context for prompt injection.

       This phase includes STATIC content only:
       - Session: recent episodes (FIFO from session stream)
       - User: consolidated blob + recent unabsorbed observations
       - Collective: consolidated blobs per shard + recent observations

       Facts are stored but NOT included (will be retrieved via RAG in Phase 5).
       """
   ```

2. **Context caching with cross-session sync**:
   - Cache assembled context in session
   - Refresh after `SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES` (5 min)
   - Preserve existing sync mechanism from current system

3. **XML output format** (without RAG section):
   ```xml
   <MemoryContext>
     <CollectiveMemory>
       <MemoryShard name="shard_name">
         ## Current consolidated shard memory:
         [consolidated content]

         ## Recent shard observations:
         - observation 1 (age: X days ago)
       </MemoryShard>
     </CollectiveMemory>

     <UserMemory>
       ## Consolidated user memory:
       [content]
       ## Recent user observations:
       - observation 1 (age: X days ago)
     </UserMemory>

     <CurrentConversationContext>
       - episode 1
       - episode 2
     </CurrentConversationContext>

     <!-- NO <RetrievedMemories> section yet - added in Phase 5 -->
   </MemoryContext>
   ```

**Testing Checkpoint**:
- [ ] `assemble_memory_context()` returns valid XML structure
- [ ] Consolidated blobs are included for all active streams
- [ ] Recent observations are included with age formatting
- [ ] Session episodes are included (FIFO)
- [ ] Context caching works correctly
- [ ] Cross-session sync triggers refresh after 5 minutes
- [ ] Performance: context assembly completes in < 100ms (no LLM calls)

---

### Phase 3: RAG Storage Infrastructure

**Goal**: Create all components related to storing RAG-enabled memories, computing embeddings, and creating the MongoDB models/indexes for this.

**Scope**: Embedding generation and storage. No retrieval yet.

**Files to Create**:

```
eve/agent/memory2/
├── embedder.py                 # MemoryEmbedder class for embedding generation
└── (modify) models.py          # Ensure embedding fields are properly defined
```

**MongoDB Atlas Setup** (manual):

```
Create these indexes in MongoDB Atlas UI:

1. Vector Search Index (name: memory_vector_index, collection: memory_atoms):
{
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"},
    {"type": "filter", "path": "stream_id"},
    {"type": "filter", "path": "rag_indexed"},
    {"type": "filter", "path": "formed_at"}
  ]
}

2. Text Search Index (name: memory_text_index, collection: memory_atoms):
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {"type": "string", "analyzer": "lucene.standard"}
    }
  }
}
```

**Deliverables**:

1. **`embedder.py`** - Embedding generation:
   ```python
   class MemoryEmbedder:
       """Generates embeddings for memory atoms."""

       def __init__(
           self,
           model: str = "text-embedding-3-small",
           batch_size: int = 100
       ):
           self.model = model
           self.batch_size = batch_size
           self.dimensions = 1536

       async def embed_atoms(self, atoms: List[MemoryAtom]) -> List[MemoryAtom]:
           """Generate embeddings for atoms in batches."""
           # Batch API calls to OpenAI
           # Store embeddings as native Python lists
           # Set rag_indexed = True

       async def embed_single(self, text: str) -> List[float]:
           """Generate embedding for a single text (for queries)."""
   ```

2. **Modify `formation.py`** - Wire embedding into formation flow:
   - After extracting facts (collective memory), generate embeddings
   - Store atoms with `embedding` field populated
   - Set `rag_indexed = True` for embedded atoms

3. **Modify `models.py`** - Ensure proper field definitions:
   ```python
   class MemoryAtom(Document):
       # ... existing fields

       # RAG fields (now populated)
       embedding: Optional[List[float]] = None  # 1536-dim vector
       embedding_model: Optional[str] = None
       rag_indexed: bool = False

       # Access tracking (for future importance scoring)
       access_count: int = 0
       last_accessed_at: Optional[datetime] = None
   ```

**Testing Checkpoint**:
- [ ] Can generate embeddings for memory atoms
- [ ] Embeddings are stored as native MongoDB arrays (NOT strings)
- [ ] `rag_indexed` is set to True for embedded atoms
- [ ] MongoDB Atlas vector index is created and functional
- [ ] MongoDB Atlas text index is created and functional
- [ ] Batch embedding (100 atoms) completes in < 5 seconds

---

### Phase 4: RAG Retrieval Functions

**Goal**: Create the RAG functions that generate a query vector (with a cheap LLM), run the RAG pipeline, and return RAG context.

**Scope**: Query generation and retrieval. Not yet integrated into context assembly.

**Files to Create**:

```
eve/agent/memory2/
├── query_generation.py         # Fast LLM query generation
└── rag.py                      # Hybrid search (semantic + text + RRF)
```

**Deliverables**:

1. **`query_generation.py`** - Fast LLM query generation:
   ```python
   RAG_QUERY_GENERATION_PROMPT = """
   You are a memory retrieval assistant. Given a conversation and a new user message,
   generate a search query to retrieve relevant memories from long-term storage.

   CONVERSATION (last {N} messages):
   {conversation_context}

   NEW USER MESSAGE:
   {user_message}

   INSTRUCTIONS:
   - If the user is asking about past events, preferences, or context, generate a query
   - If the conversation is casual/simple and doesn't need memory retrieval, return empty string
   - Keep the query concise (1-2 sentences max)

   OUTPUT: Return ONLY the query string, or empty string if no retrieval needed.
   """

   async def generate_rag_query(
       conversation_messages: List[ChatMessage],
       user_message: str,
       n_context_messages: int = 10
   ) -> str:
       """
       Generate a RAG query using a fast, cheap LLM (gpt-4o-mini).
       Returns empty string if no retrieval is needed.
       """
   ```

2. **`rag.py`** - Hybrid search implementation:
   ```python
   async def search_memories(
       query: str,
       stream_ids: List[ObjectId],
       match_count: int = 10,
       formed_after: Optional[datetime] = None,
       search_type: str = "hybrid"
   ) -> List[MemorySearchResult]:
       """Main entry point for memory search."""

   async def _semantic_search(...) -> List[MemorySearchResult]:
       """MongoDB $vectorSearch aggregation."""

   async def _text_search(...) -> List[MemorySearchResult]:
       """MongoDB Atlas Search with fuzzy matching."""

   def _reciprocal_rank_fusion(
       result_lists: List[List[MemorySearchResult]],
       k: int = 60,
       limit: int = 10
   ) -> List[MemorySearchResult]:
       """Merge ranked lists using RRF. COPY FROM MongoDB-RAG-Agent EXACTLY."""
   ```

3. **Main retrieval entry point**:
   ```python
   async def retrieve_memories_for_message(
       session: Session,
       agent: Agent,
       user: User,
       user_message: str
   ) -> List[MemorySearchResult]:
       """
       Main entry point for automatic RAG retrieval.
       1. Generate query with fast LLM
       2. If empty query, return []
       3. Perform hybrid search
       4. Update access tracking (access_count, last_accessed_at)
       5. Return results
       """
   ```

**Testing Checkpoint**:
- [ ] `generate_rag_query()` returns appropriate queries for memory-relevant messages
- [ ] `generate_rag_query()` returns empty string for casual messages
- [ ] `_semantic_search()` returns results sorted by vector similarity
- [ ] `_text_search()` returns results with fuzzy matching
- [ ] `_reciprocal_rank_fusion()` correctly merges results with k=60
- [ ] `search_memories()` combines semantic + text via RRF
- [ ] Access tracking is updated on retrieval (access_count++, last_accessed_at)
- [ ] End-to-end: `retrieve_memories_for_message()` returns relevant memories

---

### Phase 5: RAG Integration

**Goal**: Integrate RAG retrieval into the `assemble_memory_context()` function from Phase 2.

**Scope**: Wire up RAG retrieval to context assembly. Full system integration.

**Files to Modify**:

```
eve/agent/memory2/
├── (modify) context.py         # Add RAG results to context assembly
├── service.py                  # High-level MemoryService facade
└── backends.py                 # Backend abstraction
```

**Deliverables**:

1. **Modify `context.py`** - Add RAG to context assembly:
   ```python
   async def assemble_memory_context(
       session: Session,
       agent: Agent,
       user: User,
       user_message: str,  # NEW: needed for RAG query generation
       force_refresh: bool = False,
   ) -> str:
       """
       Assemble memory context with both layers:

       1. STATIC (always-in-context):
          - Consolidated blobs
          - Recent observations
          - Session episodes

       2. DYNAMIC (RAG retrieval):
          - Fast LLM generates query (invisible to main agent)
          - Hybrid search retrieves relevant facts
          - Results included in <RetrievedMemories> section
       """

       # Static layer (from Phase 2)
       session_stream = await get_stream(...)
       user_stream = await get_stream(...)
       agent_streams = await get_streams(...)

       # Dynamic layer (NEW)
       rag_results = await retrieve_memories_for_message(
           session, agent, user, user_message
       )

       return _build_memory_xml(
           session_stream, user_stream, agent_streams,
           rag_results  # NEW
       )
   ```

2. **Update XML output format** (with RAG section):
   ```xml
   <MemoryContext>
     <CollectiveMemory>...</CollectiveMemory>
     <UserMemory>...</UserMemory>
     <CurrentConversationContext>...</CurrentConversationContext>

     <!-- NEW: RAG results -->
     <RetrievedMemories description="Relevant memories retrieved for this message">
       - Memory content 1 (age: X days ago)
       - Memory content 2 (age: Y hours ago)
     </RetrievedMemories>
   </MemoryContext>
   ```

3. **`service.py`** - High-level facade:
   ```python
   class MemoryService:
       async def assemble_memory_context(...)
       async def maybe_form_memories(...)
       async def form_memories(...)
       async def search_memories(...)  # Direct RAG access if needed
   ```

4. **`backends.py`** - Backend abstraction:
   ```python
   class MemoryBackend(ABC):
       @abstractmethod
       async def assemble_memory_context(...) -> str
       @abstractmethod
       async def maybe_form_memories(...)
       # ...

   class MongoMemoryBackend(MemoryBackend):
       # Concrete implementation wrapping memory2 functions
   ```

5. **Integration with message processing pipeline**:
   - Call `assemble_memory_context()` with user_message parameter
   - RAG runs automatically (invisible to main agent)
   - Main agent receives enriched context

**Testing Checkpoint**:
- [ ] `assemble_memory_context()` includes RAG results when relevant
- [ ] `assemble_memory_context()` skips RAG when not relevant (empty query)
- [ ] XML output includes `<RetrievedMemories>` section when RAG runs
- [ ] Access tracking is updated on RAG retrieval
- [ ] Cross-session sync still works correctly
- [ ] MemoryService facade provides clean API
- [ ] End-to-end: user message → RAG query → retrieval → enriched context

---

### Phase 6: Migration

**Goal**: Create a migration script to convert existing MongoDB documents to the new architecture.

**Scope**: One-time migration of existing data. Backfill embeddings.

**Files to Create**:

```
eve/agent/memory2/
└── migration.py                # One-time migration script
```

**Migration Strategy**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MIGRATION MAPPING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  UserMemory (memory_user collection)                                        │
│  ───────────────────────────────────                                        │
│  → Create MemoryStream (stream_type="user")                                 │
│  → UserMemory.content → stream.consolidated_content                         │
│  → UserMemory.unabsorbed_memory_ids → create MemoryAtom observations        │
│  → UserMemory.fully_formed_memory → stream.fully_formed                     │
│                                                                              │
│  AgentMemory (memory_agent collection)                                      │
│  ─────────────────────────────────────                                      │
│  → Create MemoryStream (stream_type="agent")                                │
│  → AgentMemory.content → stream.consolidated_content                        │
│  → AgentMemory.facts → create MemoryAtom facts (with embeddings)            │
│  → AgentMemory.unabsorbed_memory_ids → create MemoryAtom observations       │
│  → AgentMemory.fully_formed_memory → stream.fully_formed                    │
│                                                                              │
│  SessionMemory (memory_sessions collection)                                 │
│  ──────────────────────────────────────────                                 │
│  → Create MemoryAtom for each SessionMemory                                 │
│  → Map memory_type to atom_type:                                            │
│      "episode" → fact (session stream)                                      │
│      "directive" → observation (user stream)                                │
│      "fact" → fact (agent stream, needs embedding)                          │
│      "suggestion" → observation (agent stream)                              │
│                                                                              │
│  DISCARD:                                                                   │
│  → Session episodes older than current session (ephemeral anyway)           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Deliverables**:

1. **`migration.py`** - Migration script:
   ```python
   async def migrate_memory_system():
       """
       One-time migration from old memory system to new.

       Steps:
       1. Migrate UserMemory documents → user MemoryStreams
       2. Migrate AgentMemory documents → agent MemoryStreams
       3. Migrate SessionMemory documents → MemoryAtoms
       4. Backfill embeddings for facts (batch process)
       5. Validate data integrity
       6. Create MongoDB indexes if not exists
       """

   async def migrate_user_memories():
       """Migrate all UserMemory documents."""

   async def migrate_agent_memories():
       """Migrate all AgentMemory documents (includes facts)."""

   async def migrate_session_memories():
       """Migrate SessionMemory documents to MemoryAtoms."""

   async def backfill_embeddings():
       """Generate embeddings for all facts that don't have them."""

   async def validate_migration():
       """Verify data integrity after migration."""
   ```

2. **Migration safety**:
   - Run in dry-run mode first
   - Keep old collections intact (don't delete)
   - Log all migration actions
   - Support incremental migration (can be re-run safely)

3. **Embedding backfill**:
   - Process in batches of 100
   - Skip atoms that already have embeddings
   - Log progress and errors
   - Estimate: ~$0.001 per 1000 memories

**Testing Checkpoint**:
- [ ] Migration script runs without errors in dry-run mode
- [ ] All UserMemory documents are converted to user streams
- [ ] All AgentMemory documents are converted to agent streams
- [ ] All relevant SessionMemory documents are converted to MemoryAtoms
- [ ] Embeddings are generated for all facts
- [ ] Data integrity validation passes
- [ ] Old collections are preserved (not deleted)
- [ ] System works correctly with migrated data

---

### Post-Implementation: Testing & Optimization

After all phases are complete:

1. **Functional Testing**:
   - Test full memory formation flow (extraction → storage → consolidation)
   - Test context assembly with and without RAG
   - Test cross-session memory sync
   - Test error recovery (consolidation failures, API failures)

2. **Performance Testing**:
   - Memory formation latency (single LLM call vs N calls)
   - Context assembly latency (< 100ms without RAG, < 500ms with RAG)
   - RAG query generation latency (fast LLM should be < 200ms)
   - Embedding generation throughput (100 atoms in < 5s)

3. **Quality Tuning**:
   - RAG relevance quality (semantic vs text vs hybrid)
   - RRF k parameter (start with k=60, adjust if needed)
   - Query generation prompt tuning
   - Match count tuning (start with 5, adjust based on context window)

4. **Cost Monitoring**:
   - LLM call costs (extraction, consolidation, query generation)
   - Embedding API costs (~$0.02 per 1M tokens)
   - MongoDB Atlas costs (if applicable)

---

## 7. MongoDB Atlas Setup Requirements

### 7.1 Required Search Indexes

These must be created manually in MongoDB Atlas UI:

**1. Vector Search Index** (name: `memory_vector_index`, collection: `memory_atoms`):
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "stream_id"
    },
    {
      "type": "filter",
      "path": "rag_indexed"
    },
    {
      "type": "filter",
      "path": "formed_at"
    }
  ]
}
```

**2. Text Search Index** (name: `memory_text_index`, collection: `memory_atoms`):
```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {
        "type": "string",
        "analyzer": "lucene.standard"
      }
    }
  }
}
```

### 7.2 Cost Considerations

- **MongoDB Atlas M0 (Free Tier)**: Supports vector search and Atlas Search
- **Embedding API costs**: ~$0.02 per 1M tokens for text-embedding-3-small
- **Estimated memory extraction**: ~50 tokens per memory → ~$0.001 per 1000 memories embedded
- **Search is free**: No additional cost for vector/text search queries

---

## Appendix: Key Code Snippets for Reference

### Current Extraction Flow (memory.py)
```python
async def _extract_all_memories(agent_id, conversation_text, session, ...):
    # LLM Call 1: Regular memories
    regular_memories = await extract_memories_with_llm(
        conversation_text,
        extraction_prompt=REGULAR_MEMORY_EXTRACTION_PROMPT,
        extraction_elements=["episode", "directive"],
        ...
    )
    extracted_data.update(regular_memories)

    # LLM Calls 2-N: Per shard
    for shard in active_shards:
        populated_prompt = AGENT_MEMORY_EXTRACTION_PROMPT.replace(...)
        shard_memories = await extract_memories_with_llm(
            conversation_text,
            extraction_prompt=populated_prompt,
            extraction_elements=["fact", "suggestion"],
            ...
        )
        # Merge into extracted_data with shard mapping

    return extracted_data, memory_to_shard_map
```

### Current Consolidation Flow (memory.py)
```python
async def _add_memories_and_maybe_consolidate(
    memory_doc,           # UserMemory or AgentMemory
    new_memory_ids,       # List[ObjectId]
    unabsorbed_field,     # "unabsorbed_memory_ids"
    max_before_consolidation,
    consolidation_func,   # _consolidate_user_directives or _consolidate_agent_suggestions
    memory_type,
):
    # Atomic add to unabsorbed list
    collection.update_one({"_id": ...}, {"$push": {unabsorbed_field: {"$each": new_memory_ids}}})

    # Check threshold
    if len(unabsorbed_list) >= max_before_consolidation:
        await consolidation_func(memory_doc)
```

### Current Context Assembly (memory_assemble_context.py)
```python
async def assemble_memory_context(session, agent, user, ...):
    # 1. Get user memory
    user_memory_content = await _assemble_user_memory(agent, user)

    # 2. Get agent memories (all active shards)
    agent_collective_memories = await _assemble_agent_memories(agent)

    # 3. Get episode memories (cached or fresh)
    episode_memories = await _get_episode_memories(session, force_refresh)

    # 4. Build XML
    return _build_memory_xml(user_memory_content, agent_collective_memories, episode_memories)
```

---

## 8. MongoDB-RAG-Agent Reference Implementation

This section provides specific file and function references from the `/Users/xandersteenbrugge/Documents/GitHub/MongoDB-RAG-Agent` repository. These components should be adapted for Eve's memory system.

### 8.1 Repository Structure Overview

```
MongoDB-RAG-Agent/src/
├── agent.py           # Agent orchestration with tool registration
├── cli.py             # CLI interface with streaming support
├── dependencies.py    # MongoDB + OpenAI client management
├── prompts.py         # System prompts for RAG agent
├── providers.py       # LLM provider abstraction
├── settings.py        # Pydantic settings management
├── tools.py           # ★ CRITICAL: Search functions (semantic, text, hybrid, RRF)
└── ingestion/
    ├── ingest.py      # Document ingestion pipeline (not needed for Eve)
    ├── chunker.py     # Document chunking (not needed for Eve)
    └── embedder.py    # ★ CRITICAL: Embedding generation
```

### 8.2 Components to Copy/Adapt

#### 8.2.1 Embedding Generation

**Source File**: `MongoDB-RAG-Agent/src/ingestion/embedder.py`

**Key Class**: `EmbeddingGenerator` (lines 29-193)

```python
# Key configuration (lines 32-57):
class EmbeddingGenerator:
    def __init__(
        self,
        model: str = "text-embedding-3-small",  # 1536 dimensions
        batch_size: int = 100
    ):
        self.model = model
        self.batch_size = batch_size
        self.dimensions = 1536
        self.max_tokens = 8191
```

**Critical Functions to Copy**:
| Function | Lines | Purpose | Copy to Eve |
|----------|-------|---------|-------------|
| `__init__()` | 32-57 | Initialize embedder with model config | `memory2/embedder.py` |
| `generate_embeddings_batch()` | 80-105 | Batch API call to OpenAI | `memory2/embedder.py` |
| `embed_chunks()` | 107-163 | Batch processing with progress | `memory2/embedder.py` |
| `create_embedder()` | 182-193 | Factory function | `memory2/embedder.py` |

**Eve Adaptation Notes**:
- Memory atoms are already small (~30-50 words), no truncation needed
- Keep batch_size=100 for efficient API usage
- Store embeddings as native Python lists (MongoDB handles BSON conversion)

#### 8.2.2 Semantic Vector Search

**Source File**: `MongoDB-RAG-Agent/src/tools.py`

**Function**: `semantic_search()` (lines 27-128)

```python
# Key aggregation pipeline (lines 50-95):
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",          # Index name
            "queryVector": query_embedding,   # 1536-dim vector
            "path": "embedding",              # Field containing vectors
            "numCandidates": 100,             # Search space (10x limit)
            "limit": match_count
        }
    },
    {
        "$lookup": {                          # Join with parent collection
            "from": "documents",
            "localField": "document_id",
            "foreignField": "_id",
            "as": "document_info"
        }
    },
    {
        "$project": {
            "chunk_id": "$_id",
            "content": 1,
            "similarity": {"$meta": "vectorSearchScore"},
            "metadata": 1
        }
    }
]
```

**Eve Adaptation**:
- Replace `documents` join with `memory_streams` join
- Add filter for `stream_id` (scope filtering)
- Add filter for `rag_indexed: true`
- Add optional `formed_at` temporal filter

#### 8.2.3 Text Search (Keyword/Fuzzy)

**Source File**: `MongoDB-RAG-Agent/src/tools.py`

**Function**: `text_search()` (lines 131-239)

```python
# Key pipeline (lines 150-190):
pipeline = [
    {
        "$search": {
            "index": "text_index",
            "text": {
                "query": query,
                "path": "content",
                "fuzzy": {
                    "maxEdits": 2,       # Typo tolerance
                    "prefixLength": 3
                }
            }
        }
    },
    {
        "$limit": match_count * 2        # Over-fetch for RRF merge
    },
    # ... lookup and project stages
]
```

**Eve Adaptation**:
- Same pattern applies to `memory_atoms.content`
- Keep fuzzy matching for typo tolerance
- Over-fetch (2x) for better RRF results

#### 8.2.4 Reciprocal Rank Fusion (RRF) ★ CRITICAL

**Source File**: `MongoDB-RAG-Agent/src/tools.py`

**Function**: `reciprocal_rank_fusion()` (lines 242-313)

```python
def reciprocal_rank_fusion(
    search_results_list: List[List[SearchResult]],
    k: int = 60                          # Industry standard, don't change
) -> List[SearchResult]:
    """
    Formula: RRF_score(d) = Σ(1 / (k + rank))

    Benefits:
    - Scale-independent (works with different scoring systems)
    - No normalization needed
    - Automatic deduplication
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchResult] = {}

    for results in search_results_list:
        for rank, result in enumerate(results):
            chunk_id = result.chunk_id
            rrf_score = 1.0 / (k + rank)   # Key formula

            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf_score  # Accumulate
            else:
                rrf_scores[chunk_id] = rrf_score
                chunk_map[chunk_id] = result

    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[chunk_id] for chunk_id, _ in sorted_chunks]
```

**COPY THIS FUNCTION EXACTLY** - it's algorithm-complete and well-tested.

#### 8.2.5 Hybrid Search Orchestration

**Source File**: `MongoDB-RAG-Agent/src/tools.py`

**Function**: `hybrid_search()` (lines 316-402)

```python
async def hybrid_search(
    ctx: RunContext[StateDeps[RAGState]],
    query: str,
    match_count: int = 5,
    text_weight: float = 0.3           # Not actually used, RRF handles weights
) -> str:
    """Combines semantic + text search using RRF."""

    fetch_count = match_count * 2       # Over-fetch for quality

    # Run both searches concurrently
    semantic_results, text_results = await asyncio.gather(
        semantic_search(ctx, query, fetch_count),
        text_search(ctx, query, fetch_count),
        return_exceptions=True          # Graceful degradation
    )

    # Handle failures gracefully
    results_lists = []
    if not isinstance(semantic_results, Exception):
        results_lists.append(semantic_results)
    if not isinstance(text_results, Exception):
        results_lists.append(text_results)

    # Merge with RRF
    merged = reciprocal_rank_fusion(results_lists, k=60)
    return merged[:match_count]
```

**Key Pattern**: Graceful degradation if one search fails.

#### 8.2.6 Dependencies Management

**Source File**: `MongoDB-RAG-Agent/src/dependencies.py`

**Class**: `AgentDependencies` (lines 14-126)

**Key Methods**:
| Method | Lines | Purpose |
|--------|-------|---------|
| `initialize()` | 29-74 | Connect to MongoDB + OpenAI with timeouts |
| `get_embedding()` | 84-104 | Generate single embedding for query |
| `cleanup()` | 76-82 | Close connections |

```python
# Critical: get_embedding for query embedding (lines 84-104)
async def get_embedding(self, text: str) -> List[float]:
    """Generate embedding for a single text (query)."""
    response = await self._openai_client.embeddings.create(
        model=self._settings.embedding_model,
        input=text
    )
    return response.data[0].embedding
```

#### 8.2.7 Settings Pattern

**Source File**: `MongoDB-RAG-Agent/src/settings.py`

**Class**: `Settings` (lines 12-90)

```python
class Settings(BaseSettings):
    # MongoDB
    mongodb_uri: str
    mongodb_database: str = "rag_db"
    mongodb_collection_documents: str = "documents"
    mongodb_collection_chunks: str = "chunks"
    mongodb_vector_index: str = "vector_index"
    mongodb_text_index: str = "text_index"

    # Embeddings
    embedding_provider: str = "openai"
    embedding_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Search defaults
    default_match_count: int = 10
    max_match_count: int = 50
```

**Eve Adaptation**:
- Add to existing Eve settings or create `memory2/settings.py`
- Keep embedding config centralized

### 8.3 Code Mapping: MongoDB-RAG-Agent → Eve Memory2

| MongoDB-RAG-Agent File | Function/Class | → Eve Memory2 File | Notes |
|------------------------|----------------|-------------------|-------|
| `ingestion/embedder.py` | `EmbeddingGenerator` | `memory2/embedder.py` | Copy with minor adaptations |
| `ingestion/embedder.py` | `embed_chunks()` | `memory2/embedder.py` → `embed_atoms()` | Rename chunks→atoms |
| `tools.py` | `semantic_search()` | `memory2/rag.py` → `_semantic_search()` | Add stream_id filter |
| `tools.py` | `text_search()` | `memory2/rag.py` → `_text_search()` | Add stream_id filter |
| `tools.py` | `reciprocal_rank_fusion()` | `memory2/rag.py` | **Copy exactly** |
| `tools.py` | `hybrid_search()` | `memory2/rag.py` → `search_memories()` | Adapt context/args |
| `dependencies.py` | `get_embedding()` | `memory2/embedder.py` | For query embedding |
| `settings.py` | `Settings` | `memory2/constants.py` | Merge with existing |

### 8.4 Components NOT Needed from MongoDB-RAG-Agent

| File | Reason |
|------|--------|
| `ingestion/ingest.py` | Eve doesn't ingest documents; memory atoms are already small |
| `ingestion/chunker.py` | No chunking needed; atoms are ~30-50 words each |
| `agent.py` | Eve has its own agent framework |
| `cli.py` | Eve has its own interface |
| `prompts.py` | Eve has its own prompt system |

### 8.5 MongoDB Index Configuration (From Repo)

**Source**: The repo uses these index definitions:

**Vector Index** (create in Atlas UI):
```json
{
  "name": "vector_index",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1536,
        "similarity": "cosine"
      }
    ]
  }
}
```

**Text Index** (create in Atlas UI):
```json
{
  "name": "text_index",
  "definition": {
    "mappings": {
      "dynamic": false,
      "fields": {
        "content": {
          "type": "string",
          "analyzer": "lucene.standard"
        }
      }
    }
  }
}
```

**For Eve**, add filter fields:
```json
{
  "name": "memory_vector_index",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1536,
        "similarity": "cosine"
      },
      {"type": "filter", "path": "stream_id"},
      {"type": "filter", "path": "rag_indexed"},
      {"type": "filter", "path": "formed_at"}
    ]
  }
}
```

### 8.6 Implementation Sequence

**Step 1**: Create `memory2/embedder.py`
- Copy `EmbeddingGenerator` class from `MongoDB-RAG-Agent/src/ingestion/embedder.py`
- Rename `embed_chunks()` → `embed_atoms()`
- Keep batch_size=100, model="text-embedding-3-small"

**Step 2**: Create `memory2/rag.py`
- Copy `reciprocal_rank_fusion()` **exactly** from `MongoDB-RAG-Agent/src/tools.py:242-313`
- Adapt `semantic_search()` from `tools.py:27-128` with stream_id filtering
- Adapt `text_search()` from `tools.py:131-239` with stream_id filtering
- Create `search_memories()` orchestration from `hybrid_search()` pattern

**Step 3**: Create MongoDB Atlas Indexes
- Vector index on `memory_atoms.embedding` with filter fields
- Text index on `memory_atoms.content`

**Step 4**: Wire into memory formation flow
- Call `embed_atoms()` for new facts before saving
- Set `rag_indexed=True` on embedded atoms

**Step 5**: Wire into context assembly
- Implement `retrieve_memories_for_message()` calling `search_memories()`
- Add RAG results to memory context XML

### 8.7 Key Differences: MongoDB-RAG-Agent vs Eve Memory

| Aspect | MongoDB-RAG-Agent | Eve Memory System |
|--------|-------------------|-------------------|
| **Unit** | Document chunks (~512 tokens) | Memory atoms (~30-50 words) |
| **Ingestion** | Full pipeline (parse, chunk, embed) | Extract → embed (no parsing) |
| **Scope** | Single knowledge base | Multiple streams (user, agent, session) |
| **Query Trigger** | Agent tool call | Automatic (fast LLM generates query) |
| **Context** | RAG results only | Static blobs + RAG results |
| **Access Tracking** | None | `access_count`, `last_accessed_at` |

---

*End of document. Last updated: 2025-12-18 (Added MongoDB-RAG-Agent implementation references)*
