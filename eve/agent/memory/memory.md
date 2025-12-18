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
**Exports**: `GraphitiConfig`, `init_graphiti`, `MemoryBackend`, `MongoMemoryBackend`, `GraphitiMemoryBackend`, `MemoryService`, `memory_service`

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
- `GraphitiMemoryBackend` - Stub for future implementation

### `memory/graphiti.py` (96 lines)
**Purpose**: Graphiti (graph-based memory) initialization helpers
**Status**: Optional dependency, not fully integrated

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

    # Provenance
    source_session_id: ObjectId
    source_message_ids: List[ObjectId] = []
    related_users: List[ObjectId] = []
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

### 4.5 Proposed File Structure

```
eve/agent/memory2/              # New folder alongside existing memory/
├── __init__.py                 # Public API exports
├── models.py                   # MemoryStream, MemoryAtom
├── ops.py                      # MemoryStreamOps class
├── constants.py                # Default values, LLM models
├── prompts.py                  # Default prompt templates
├── context.py                  # assemble_memory_context()
├── formation.py                # maybe_form_memories(), form_memories()
├── extraction.py               # LLM extraction logic (single or batched)
├── consolidation.py            # LLM consolidation logic
├── migration.py                # One-time migration script
├── service.py                  # High-level MemoryService facade
└── backends.py                 # Backend abstraction
```

---

## 5. Open Questions & Decisions

### 5.1 LLM Call Efficiency (MAJOR DECISION NEEDED)

**Problem**: Multiple LLM calls per memory formation is expensive.

**Options:**

#### Option A: Batched Multi-Stream Extraction (Recommended for cost)
Single LLM call extracts for ALL streams at once.

```python
BATCHED_PROMPT = """
Extract memories for multiple streams:

<conversation>{conversation_text}</conversation>

<stream id="1" name="Session Episodes">
<context>{session_context}</context>
<instructions>Extract 1 episode summary (≤50 words)</instructions>
</stream>

<stream id="2" name="User Preferences">
<context>{user_context}</context>
<instructions>Extract 0-4 directives (≤25 words each)</instructions>
</stream>

<stream id="3" name="Project Alpha">
<context>{shard_context}</context>
<instructions>
  consolidated: 0-5 suggestions (≤35 words)
  atomic: 0-2 facts (≤30 words)
</instructions>
</stream>

Return JSON: {"1": {...}, "2": {...}, "3": {...}}
"""
```

**Pros**: Single call regardless of stream count, 75-90% cost reduction
**Cons**: Complex prompt, potential cross-contamination between streams

#### Option B: Two-Stage Pipeline
1. Single call extracts ALL memorable content as raw candidates
2. Route candidates to streams (can be rule-based or embedding-based)

**Pros**: Clean separation, routing can be cheaper
**Cons**: Two stages add latency, routing accuracy may suffer

#### Option C: Parallel Separate Calls (Current)
Keep separate prompts, run in parallel with `asyncio.gather`.

**Pros**: Isolated quality per stream
**Cons**: Multiple API calls, cost adds up

#### Option D: Configurable Per-Stream
Allow streams to opt into batching or request isolated extraction.

**Pros**: Flexibility
**Cons**: Complexity in orchestration

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
1. Create `memory2/models.py` with `MemoryStream` and `MemoryAtom`
2. Create `memory2/ops.py` with `MemoryStreamOps` class
3. Create `memory2/prompts.py` with default prompts
4. Create `memory2/constants.py` with default values

### Phase 2: Extraction & Consolidation
1. Create `memory2/extraction.py` - start with separate calls, optimize later
2. Create `memory2/consolidation.py` - generic consolidation logic
3. Wire up extraction → add → consolidation flow

### Phase 3: Context Assembly
1. Create `memory2/context.py` with `assemble_memory_context()`
2. Handle all stream types uniformly
3. Generate XML output compatible with current format

### Phase 4: Formation Orchestration
1. Create `memory2/formation.py` with `maybe_form_memories()`, `form_memories()`
2. Get streams for context (session, user, agent shards)
3. Extract, add, assemble in unified flow

### Phase 5: Service Layer & Integration
1. Create `memory2/service.py` facade
2. Create `memory2/backends.py` with new backend
3. Allow switching between old and new via config

### Phase 6: Migration
1. Create `memory2/migration.py` script
2. Migrate `UserMemory` → user streams
3. Migrate `AgentMemory` → agent streams
4. Validate data integrity

### Phase 7: LLM Optimization (Post-Validation)
1. Implement batched extraction (Option A)
2. Measure quality vs cost tradeoff
3. Add fallback to separate calls if batch fails

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

*End of document. Last updated: 2025-12-18*
