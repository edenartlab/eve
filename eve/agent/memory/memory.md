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
1. **Consolidated Content**: Evolving summary, always injected into context
2. **Atomic Statements**: Growing list of discrete items (for RAG retrieval in future)

| Type | Has Consolidated? | Has Atomics? |
|------|-------------------|--------------|
| Session Episode | No (could add) | Yes (episodes) |
| User Directive | Yes (directives) | No (could add) |
| Agent Shard | Yes (suggestions) | Yes (facts) |

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
│   extraction_fields: ["consolidated", "atomic"]                     │
│   consolidated_extraction_min/max: int                              │
│   consolidated_atom_max_words: int                                  │
│   atomic_extraction_min/max: int                                    │
│   atomic_item_max_words: int                                        │
├─────────────────────────────────────────────────────────────────────┤
│ CONSOLIDATION CONFIG                                                │
│   consolidation_prompt: Optional[str]                               │
│   consolidation_trigger: int                                        │
│   consolidated_max_words: int                                       │
├─────────────────────────────────────────────────────────────────────┤
│ ATOMIC BUFFER CONFIG                                                │
│   atomic_buffer_max: int (FIFO limit)                               │
│   atomic_rag_enabled: bool (future)                                 │
├─────────────────────────────────────────────────────────────────────┤
│ STATE                                                               │
│   consolidated_content: Optional[str]                               │
│   unabsorbed_consolidated_ids: List[ObjectId]                       │
│   atomic_ids: List[ObjectId] (FIFO buffer)                          │
│   fully_formed: Optional[str]                                       │
│   last_updated_at: datetime                                         │
├─────────────────────────────────────────────────────────────────────┤
│ OPERATIONS                                                          │
│   extract(conversation_text) → Dict[str, List[MemoryAtom]]          │
│   add(atoms) → triggers consolidation if threshold met              │
│   consolidate() → merges unabsorbed into consolidated               │
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
    extraction_fields: List[str] = ["consolidated", "atomic"]
    consolidated_extraction_min: int = 0
    consolidated_extraction_max: int = 1
    consolidated_atom_max_words: int = 50
    atomic_extraction_min: int = 0
    atomic_extraction_max: int = 5
    atomic_item_max_words: int = 30

    # Consolidation config
    consolidation_prompt: Optional[str] = None
    consolidation_trigger: int = 10
    consolidated_max_words: int = 500

    # Atomic buffer config
    atomic_buffer_max: int = 100
    atomic_rag_enabled: bool = False

    # State
    consolidated_content: Optional[str] = None
    unabsorbed_consolidated_ids: List[ObjectId] = []
    atomic_ids: List[ObjectId] = []
    fully_formed: Optional[str] = None
    last_updated_at: Optional[datetime] = None

    # Optional LLM config override
    extraction_model: Optional[str] = None
    consolidation_model: Optional[str] = None


@Collection("memory_atoms")
class MemoryAtom(Document):
    """Single extracted memory unit - replaces SessionMemory"""

    stream_id: ObjectId
    atom_type: Literal["consolidated", "atomic"]
    content: str

    # Temporal (CRITICAL: every memory must have formation timestamp)
    formed_at: datetime  # When this memory was extracted/formed
    created_at: datetime = datetime.now(timezone.utc)

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
| **Session Episode** | `stream_type="session"`, `extraction_fields=["atomic"]` (or `["consolidated", "atomic"]` if adding session summaries), `consolidation_prompt=None` or session summary prompt, scope=`{session_id}` |
| **User Directive** | `stream_type="user"`, `extraction_fields=["consolidated"]`, `consolidation_prompt=USER_PROMPT`, scope=`{agent_id, user_id}` |
| **Agent Shard** | `stream_type="agent"`, `extraction_fields=["consolidated", "atomic"]`, `consolidation_prompt=AGENT_PROMPT`, scope=`{agent_id, shard_name}` |

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
        return {"consolidated": [...], "atomic": [...]}

    async def add(self, atoms: Dict[str, List[MemoryAtom]]) -> None:
        """Add atoms to stream, triggering consolidation if needed"""
        # Add to unabsorbed_consolidated_ids
        for atom in atoms.get("consolidated", []):
            atom.save()
            self.stream.unabsorbed_consolidated_ids.append(atom.id)

        # Check consolidation trigger
        if len(self.stream.unabsorbed_consolidated_ids) >= self.stream.consolidation_trigger:
            await self.consolidate()

        # Add to atomic_ids (FIFO)
        for atom in atoms.get("atomic", []):
            atom.save()
            self.stream.atomic_ids.append(atom.id)

        # FIFO eviction
        if len(self.stream.atomic_ids) > self.stream.atomic_buffer_max:
            self.stream.atomic_ids = self.stream.atomic_ids[-self.stream.atomic_buffer_max:]

        await self.regenerate_fully_formed()
        self.stream.save()

    async def consolidate(self) -> None:
        """Merge unabsorbed atoms into consolidated_content"""
        if not self.stream.consolidation_prompt:
            return

        unabsorbed = MemoryAtom.find({"_id": {"$in": self.stream.unabsorbed_consolidated_ids}})
        new_content = await llm_consolidate(
            self.stream.consolidation_prompt,
            self.stream.consolidated_content,
            unabsorbed,
            max_words=self.stream.consolidated_max_words
        )

        self.stream.consolidated_content = new_content
        self.stream.unabsorbed_consolidated_ids = []

    async def regenerate_fully_formed(self) -> None:
        """Build fully_formed string from current state"""
        parts = []

        if self.stream.atomic_ids:
            atoms = MemoryAtom.find({"_id": {"$in": self.stream.atomic_ids}})
            parts.append(self._format_atomics(atoms))

        if self.stream.consolidated_content:
            parts.append(f"## Memory:\n{self.stream.consolidated_content}")

        if self.stream.unabsorbed_consolidated_ids:
            unabsorbed = MemoryAtom.find({"_id": {"$in": self.stream.unabsorbed_consolidated_ids}})
            parts.append(self._format_unabsorbed(unabsorbed))

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
│  Session Context (query)                                                    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ 1. Always-in-Context Layer                                   │            │
│  │   ├── Consolidated memory blobs (fully_formed)              │            │
│  │   └── Recent unabsorbed atoms (awaiting consolidation)      │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ 2. RAG Augmentation Layer (MongoDB Vector Search)           │            │
│  │   ├── Query embedding generation (OpenAI text-embedding-3)  │            │
│  │   ├── Hybrid search (semantic + keyword via RRF)            │            │
│  │   └── Filter by stream scope (session/user/agent)           │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ 3. Assembled Memory Context (XML)                            │            │
│  │   ├── Static: consolidated blobs + recent atoms             │            │
│  │   └── Dynamic: RAG-retrieved relevant memories              │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.5.2 MongoDB Collections & Indexes

**`memory_atoms` Collection Schema:**
```python
{
  "_id": ObjectId,
  "stream_id": ObjectId,           # Foreign key to memory_streams
  "atom_type": "consolidated" | "atomic",
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

#### 4.5.5 Memory Flow: Extraction → Storage → RAG

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
│  STEP 2: ROUTING (Based on memory_type)                                     │
│  ──────────────────────────────────────                                      │
│  Each extracted memory has:                                                  │
│    - memory_type: session | user | collective                               │
│    - storage_type: temporary | long_term                                    │
│    - formed_at: timestamp (ALWAYS present)                                  │
│                                                                              │
│  ┌────────────────┬──────────────────────────────────────────────┐          │
│  │ Memory Type    │ Destination                                  │          │
│  ├────────────────┼──────────────────────────────────────────────┤          │
│  │ session        │ Temporary: recent context list only          │          │
│  │ user           │ Long-term: RAG-indexed + consolidation       │          │
│  │ collective     │ Long-term: RAG-indexed + consolidation       │          │
│  └────────────────┴──────────────────────────────────────────────┘          │
│       │                                                                      │
│       ▼                                                                      │
│  STEP 3: STORAGE & EMBEDDING                                                │
│  ────────────────────────────                                               │
│  For long-term memories:                                                    │
│    1. Generate embedding (text-embedding-3-small)                           │
│    2. Store in memory_atoms with embedding                                  │
│    3. Add to stream's unabsorbed list                                       │
│    4. Trigger consolidation if threshold met                                │
│                                                                              │
│  For temporary memories:                                                    │
│    1. Store in memory_atoms (no embedding)                                  │
│    2. Add to stream's recent list (FIFO)                                    │
│       │                                                                      │
│       ▼                                                                      │
│  STEP 4: CONTEXT ASSEMBLY (On each message)                                 │
│  ─────────────────────────────────────────                                  │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ Always-in-Context:                                          │            │
│  │   ├── Session: recent episodes (FIFO, no embedding)         │            │
│  │   ├── User: consolidated blob + recent directives           │            │
│  │   └── Collective: consolidated blobs + recent suggestions   │            │
│  │                                                             │            │
│  │ RAG-Augmented (query = session context summary):            │            │
│  │   ├── User memories: top-K relevant facts                  │            │
│  │   └── Collective memories: top-K relevant facts per shard  │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.5.6 Context Assembly with RAG

```python
async def assemble_memory_context(
    session: Session,
    agent: Agent,
    user: User,
    enable_rag: bool = True,
    rag_match_count: int = 5
) -> str:
    """
    Assemble complete memory context for prompt injection.

    Combines:
    1. Static: consolidated blobs + recent unabsorbed atoms
    2. Dynamic: RAG-retrieved relevant memories (when enabled)
    """

    # Get all relevant streams
    session_stream = await get_stream(stream_type="session", scope={"session_id": session.id})
    user_stream = await get_stream(stream_type="user", scope={"agent_id": agent.id, "user_id": user.id})
    agent_streams = await get_streams(stream_type="agent", scope={"agent_id": agent.id}, is_active=True)

    # Collect stream IDs for RAG query
    all_stream_ids = [user_stream.id] + [s.id for s in agent_streams]

    # Build query from recent session context
    rag_query = _build_rag_query(session)

    # Get RAG results if enabled
    rag_results = []
    if enable_rag and rag_query:
        rag_results = await search_memories(
            query=rag_query,
            stream_ids=all_stream_ids,
            match_count=rag_match_count,
            search_type="hybrid"
        )

    # Build XML context
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
    rag_results: List[MemorySearchResult]
) -> str:
    """Build XML memory context with RAG augmentation."""

    parts = ['<MemoryContext>']

    # Collective/Agent memories
    parts.append('  <CollectiveMemory>')
    for stream in agent_streams:
        parts.append(f'    <MemoryShard name="{stream.name}">')
        parts.append(f'      {stream.fully_formed}')
        parts.append('    </MemoryShard>')
    parts.append('  </CollectiveMemory>')

    # User memory
    parts.append('  <UserMemory>')
    parts.append(f'    {user_stream.fully_formed}')
    parts.append('  </UserMemory>')

    # Session context
    parts.append('  <CurrentConversation>')
    parts.append(f'    {session_stream.fully_formed}')
    parts.append('  </CurrentConversation>')

    # RAG-retrieved memories (augmentation)
    if rag_results:
        parts.append('  <RelevantMemories description="Contextually retrieved from long-term storage">')
        for result in rag_results:
            age = _format_age(result["formed_at"])
            parts.append(f'    - {result["content"]} (formed: {age})')
        parts.append('  </RelevantMemories>')

    parts.append('</MemoryContext>')
    return '\n'.join(parts)
```

### 4.6 Proposed File Structure

```
eve/agent/memory2/              # New folder alongside existing memory/
├── __init__.py                 # Public API exports
├── models.py                   # MemoryStream, MemoryAtom
├── ops.py                      # MemoryStreamOps class
├── constants.py                # Default values, LLM models
├── prompts.py                  # Default prompt templates
├── context.py                  # assemble_memory_context()
├── formation.py                # maybe_form_memories(), form_memories()
├── extraction.py               # LLM extraction logic (single batched call)
├── consolidation.py            # LLM consolidation logic
├── embedder.py                 # MemoryEmbedder class for embedding generation
├── rag.py                      # search_memories(), hybrid search, RRF
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
Extract 1-2 session episodes summarizing key events (≤50 words each).
These are temporary memories for current conversation context only.
</session_memories>
{% endif %}

{% if user_active %}
<user_memories>
Extract 0-4 user directives about preferences, instructions, or context (≤25 words each).
These will be stored long-term and indexed for RAG retrieval.
</user_memories>
{% endif %}

{% if collective_active %}
<collective_memories>
{% for shard in active_shards %}
<shard name="{shard.name}">
<context>{shard.extraction_prompt}</context>
<instructions>
  - facts: 0-3 factual observations (≤30 words, long-term RAG storage)
  - suggestions: 0-2 behavioral suggestions (≤35 words, for consolidation)
</instructions>
</shard>
{% endfor %}
</collective_memories>
{% endif %}

Return JSON with formed_at timestamp for each memory:
{
  "session": [{"content": "...", "formed_at": "ISO8601"}],
  "user": [{"content": "...", "formed_at": "ISO8601"}],
  "collective": {
    "shard_name": {
      "facts": [{"content": "...", "formed_at": "ISO8601"}],
      "suggestions": [{"content": "...", "formed_at": "ISO8601"}]
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
- `extraction_fields=["consolidated", "atomic"]`
- `consolidation_prompt` for session summary
- `atomic_ids` for recent episodes
- `consolidated_content` for evolving session summary

### 5.3 Migration Strategy

**Decision Made**: Hack migration approach:
- Jam existing `UserMemory.content` into stream's `consolidated_content`
- Jam existing `AgentMemory.content` into stream's `consolidated_content`
- Copy `AgentMemory.facts` to stream's `atomic_ids`
- Discard episodes (ephemeral anyway)

### 5.4 Prompt Storage

**Decision Made**: Full prompts stored in DB (not templates with parameters).

This provides maximum flexibility but requires careful prompt management.

---

## 6. Implementation Plan

### Phase 1: Core Models & Basic Operations
1. Create `memory2/models.py` with `MemoryStream` and `MemoryAtom` (including `formed_at` timestamp, embedding fields)
2. Create `memory2/ops.py` with `MemoryStreamOps` class
3. Create `memory2/prompts.py` with default prompts (unified extraction prompt)
4. Create `memory2/constants.py` with default values and embedding config

### Phase 2: Unified Extraction
1. Create `memory2/extraction.py` with single-call batched extraction
2. Implement prompt templating for conditional sub-prompts (session/user/collective)
3. Parse structured JSON output with `formed_at` timestamps for each memory
4. Route extracted memories to appropriate streams based on type

### Phase 3: Embedding & RAG Infrastructure
1. Create `memory2/embedder.py` with `MemoryEmbedder` class
   - Batch embedding generation using OpenAI text-embedding-3-small
   - Store embeddings as native MongoDB arrays (NOT strings)
2. Create `memory2/rag.py` with search functions:
   - `search_memories()` - main entry point
   - `_semantic_search()` - MongoDB $vectorSearch aggregation
   - `_text_search()` - MongoDB Atlas Search with fuzzy matching
   - `_reciprocal_rank_fusion()` - merge rankings with k=60
3. Create MongoDB Atlas Search indexes:
   - Vector index on `memory_atoms.embedding` (1536 dim, cosine)
   - Text index on `memory_atoms.content`

### Phase 4: Consolidation
1. Create `memory2/consolidation.py` - generic consolidation logic
2. Wire up extraction → embedding → storage → consolidation flow
3. Handle long-term vs temporary memory routing:
   - Long-term (user/collective facts): embed + index + consolidate
   - Temporary (session episodes): store without embedding, FIFO eviction

### Phase 5: Context Assembly with RAG
1. Create `memory2/context.py` with `assemble_memory_context()`
2. Implement two-layer context assembly:
   - **Always-in-Context**: consolidated blobs + recent unabsorbed atoms
   - **RAG-Augmented**: query long-term storage for relevant memories
3. Build RAG query from recent session context
4. Generate XML output with `<RelevantMemories>` section for RAG results
5. Include `formed_at` age formatting for all memories

### Phase 6: Formation Orchestration
1. Create `memory2/formation.py` with `maybe_form_memories()`, `form_memories()`
2. Determine active memory types (session, user, collective) based on config
3. Single LLM call for extraction with conditional sub-prompts
4. Route memories, generate embeddings for long-term, save atoms
5. Update stream states, trigger consolidation if thresholds met

### Phase 7: Service Layer & Integration
1. Create `memory2/service.py` facade
2. Create `memory2/backends.py` with new backend
3. Add config flag to switch between old and new memory system
4. Implement graceful fallback if RAG search fails

### Phase 8: Migration
1. Create `memory2/migration.py` script
2. Migrate `UserMemory` → user streams
3. Migrate `AgentMemory` → agent streams
4. Backfill embeddings for existing atoms (batch process)
5. Create MongoDB Atlas Search indexes
6. Validate data integrity and search functionality

### Phase 9: Testing & Optimization
1. Test hybrid search quality (semantic vs text vs hybrid)
2. Tune RAG parameters (match_count, RRF k value)
3. Monitor embedding costs and latency
4. Add caching for frequent queries if needed

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

*End of document. Last updated: 2025-12-18 (Updated with MongoDB RAG integration plan)*
