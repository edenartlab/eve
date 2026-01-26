# Eden Memory System: Revised Architecture Proposal

*Document created: 2025-12-21*
*Based on: memory.md (2025-12-18) with architectural revisions*
*Purpose: Simplified, consolidation-forward memory system design*

---

## Table of Contents
1. [Current System Overview](#1-current-system-overview)
2. [File-by-File Analysis](#2-file-by-file-analysis)
3. [Current Problems & Challenges](#3-current-problems--challenges)
4. [Critique of Original Proposal](#4-critique-of-original-proposal)
5. [Revised Architecture](#5-revised-architecture)
6. [Implementation Plan](#6-implementation-plan)
7. [Migration Strategy](#7-migration-strategy)

---

## 1. Current System Overview

*This section preserved from original document - accurately describes the existing system.*

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

### Memory Types Comparison (Current System)

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

---

## 2. File-by-File Analysis

*This section preserved from original document - accurately describes the existing codebase.*

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

### `memory/memory_constants.py` (318 lines)
**Purpose**: Configuration constants and prompt templates

**Key Constants:**
```python
MEMORY_LLM_MODEL_FAST = "gpt-5-mini"
MEMORY_LLM_MODEL_SLOW = "gpt-5.1"
MEMORY_FORMATION_MSG_INTERVAL = 45
MEMORY_FORMATION_TOKEN_INTERVAL = 1000
MAX_USER_MEMORIES_BEFORE_CONSOLIDATION = 5
MAX_N_EPISODES_TO_REMEMBER = 8
MAX_AGENT_MEMORIES_BEFORE_CONSOLIDATION = 16
MAX_FACTS_PER_SHARD = 100
```

### `memory/memory.py` (1226 lines)
**Purpose**: Core memory formation logic

**Entry Points:**
- `maybe_form_memories(agent_id, session, agent)` - Checks triggers, calls form_memories
- `form_memories(agent_id, session, agent, conversation_text, char_counts)` - Main memory formation

### `memory/memory_assemble_context.py` (439 lines)
**Purpose**: Assembles memory context for prompt injection

**Entry Point:**
- `assemble_memory_context(session, agent, user, force_refresh, reason, skip_save, instrumentation)` - Returns XML memory context string

### `memory/memory_cold_sessions_processor.py` (258 lines)
**Purpose**: Background processing for cold sessions (Modal deployment)

### `memory/service.py` (79 lines)
**Purpose**: High-level facade for memory operations

### `memory/backends.py` (168 lines)
**Purpose**: Backend abstraction

---

## 3. Current Problems & Challenges

*This section preserved and extended from original document.*

### 3.1 Ontology Over-Engineering
The current system has **four memory types** (episode, directive, fact, suggestion) with overlapping semantics:
- **Directive vs Suggestion**: Both are "things to remember for behavior" — why different types?
- **Fact vs Episode**: Both are "things that happened/were said" — the distinction is unclear
- **Type proliferation**: Each type has bespoke extraction, storage, and consolidation logic

This ontology adds cognitive overhead without proportional value. The LLM extracting memories must decide between fine-grained categories that are semantically similar.

### 3.2 Scope Confusion ("Shards")
The current system uses arbitrary "shards" for agent-level memory without clear semantic boundaries. Questions that arise:
- What defines a shard vs another shard?
- How do shards relate to each other?
- Who decides shard boundaries?

A cleaner model would use **explicit scope hierarchy** with clear semantics.

### 3.3 Code Duplication
- Similar logic repeated for user vs agent memory (add, consolidate, regenerate)
- `_update_user_memory()` and `_update_agent_memory()` follow same pattern
- `_consolidate_user_directives()` and `_consolidate_agent_suggestions()` are nearly identical

### 3.4 LLM Call Inefficiency
Current system requires **N+1 LLM calls** per memory formation:
- 1 call for episode + directive extraction
- N calls for N active agent shards

With 3 active shards: **4+ LLM calls per formation event**. Expensive and latency-adding.

### 3.5 RAG Not Implemented
The current system stores atomic items but has no semantic retrieval. However, the question is: **do we need full RAG, or is something simpler sufficient?**

### 3.6 Consolidation as Afterthought
Current architecture treats atomic memories as primary and consolidation as a secondary cleanup process. But consolidations are what actually go into the prompt context — they should be the primary artifact.

---

## 4. Critique of Original Proposal

The original `memory.md` proposed a MemoryStream abstraction with full RAG pipeline. While thorough, several aspects warrant reconsideration:

### 4.1 RAG Query Generation LLM is an Antipattern

**Original proposal**: "A fast, cheap LLM (gpt-4o-mini) examines the last N messages and generates a query string" on every user message.

**Problems**:
- Adds 100-300ms latency to every message, even casual ones
- Contradicts the goal of reducing LLM calls — trades N extraction calls for 1 call per message
- Industry has moved away from this pattern (Mem0, Zep tried it, abandoned it)

**Better approach**: Simple heuristics for retrieval triggers, or always-retrieve with fast vector search.

### 4.2 Observation vs Fact Distinction Adds Cognitive Load

**Original proposal**: Two atom types — "observation" (pre-consolidation buffer) and "fact" (long-term RAG).

**Problem**: The distinction is subtle and may not be meaningful to the extracting LLM. More categories = more code paths = more edge cases.

### 4.3 Full RAG Pipeline May Be Overkill

With ~100 facts per shard (FIFO), semantic vector search may be over-engineering. Many production systems do fine with:
- Keyword/regex matching
- Recency-weighted retrieval
- Simple text search

Vector search shines with large corpora (10K+ items). For small memory stores, simpler approaches may suffice.

### 4.4 Session Consolidation is Scope Creep

**Original proposal**: "Decision Made: Yes, add session consolidation."

**Problem**: This wasn't in the "Current Problems" section. Sessions are ephemeral by design. Adding consolidation for sessions increases complexity without solving a stated problem.

### 4.5 Six-Phase Implementation is Risky

The original proposal's 6-phase plan with full RAG pipeline, migration, and MongoDB Atlas setup is ~3-6 months of work. For one engineer, this is dangerously ambitious.

---

## 5. Revised Architecture

### 5.1 Core Philosophy: Consolidation-Forward

The key insight: **consolidations are the memory**. Atomic observations are receipts, not the meal.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CONSOLIDATION-FORWARD MEMORY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PRIMARY ARTIFACT: Consolidation blobs at each scope level                   │
│    • Always in context                                                       │
│    • Hierarchically generated from observations                              │
│    • The agent's actual "memory"                                             │
│                                                                              │
│  SECONDARY ARTIFACT: Observation log                                         │
│    • For audit/provenance                                                    │
│    • For specific recall ("what did I say about X?")                        │
│    • For contradiction detection                                             │
│    • Retrieved on-demand, NOT on every message                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Simplified Ontology

**Replace four types with one**: `Observation`

An observation is a timestamped piece of information extracted from conversation. No sub-categorization needed — the LLM extracts what's worth remembering, and scope determines where it lives.

| Old System | New System |
|------------|------------|
| Episode | Observation (session scope) |
| Directive | Observation (individual scope) |
| Fact | Observation (group/collective scope) |
| Suggestion | Observation (group/collective scope) |

The consolidation process handles synthesis. We don't need to pre-categorize what the LLM will do with the information.

### 5.3 Three-Level Scope Hierarchy

Replace arbitrary "shards" with a clean hierarchy matching organizational reality:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCOPE HIERARCHY                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      COLLECTIVE SCOPE                                │    │
│  │  "What the agent knows from ALL conversations"                       │    │
│  │                                                                      │    │
│  │  • One consolidation per agent                                       │    │
│  │  • Reflects across all users                                         │    │
│  │  • Updated from group-level observations                             │    │
│  │  • Example: "Users generally prefer concise responses"               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               ▲                                              │
│                               │ propagates up                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        GROUP SCOPE                                   │    │
│  │  "What the agent knows about this team/project/community"            │    │
│  │                                                                      │    │
│  │  • One consolidation per agent+group                                 │    │
│  │  • Shared context for group members                                  │    │
│  │  • Updated from individual-level observations (when shareable)       │    │
│  │  • Example: "The Eden project uses Python and MongoDB"               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               ▲                                              │
│                               │ propagates up (if appropriate)               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      INDIVIDUAL SCOPE                                │    │
│  │  "What the agent knows about this specific person"                   │    │
│  │                                                                      │    │
│  │  • One consolidation per agent+user                                  │    │
│  │  • Private to this user                                              │    │
│  │  • Updated from session observations                                 │    │
│  │  • Example: "Gene prefers direct communication, works on Eve"        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               ▲                                              │
│                               │ extracted from                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      SESSION (Ephemeral)                             │    │
│  │  Raw conversation — not persisted as memory                          │    │
│  │  Observations extracted periodically → Individual scope              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Data Models

```python
@Collection("memory_scopes")
class MemoryScope(Document):
    """
    A scope represents a consolidation boundary.
    Each scope has one consolidation blob and an observation log.
    """

    # Identity
    agent_id: ObjectId
    scope_type: Literal["individual", "group", "collective"]

    # Scope identifiers (which combination is populated depends on scope_type)
    user_id: Optional[ObjectId] = None       # For individual scope
    group_id: Optional[ObjectId] = None      # For group scope
    # collective scope: only agent_id needed

    # Configuration
    extraction_prompt: Optional[str] = None   # Custom extraction guidance
    consolidation_prompt: Optional[str] = None  # Custom consolidation guidance
    is_active: bool = True

    # Primary artifact: the consolidation
    consolidation: Optional[str] = None       # The synthesized memory blob
    consolidation_updated_at: Optional[datetime] = None

    # Secondary artifact: observation buffer (IDs of unprocessed observations)
    pending_observation_ids: List[ObjectId] = []

    # Consolidation trigger
    consolidation_threshold: int = 10  # Consolidate after N observations
    consolidation_max_words: int = 500

    # Timestamps
    created_at: datetime = datetime.now(timezone.utc)
    last_activity_at: Optional[datetime] = None


@Collection("memory_observations")
class Observation(Document):
    """
    A single extracted observation. Simple structure, no sub-categorization.
    """

    # Identity
    scope_id: ObjectId                        # Which scope this belongs to

    # Content
    content: str                              # The observation text

    # Temporal
    observed_at: datetime                     # When this was observed (from conversation)
    created_at: datetime = datetime.now(timezone.utc)

    # Provenance (for audit/debugging)
    source_session_id: ObjectId
    source_message_ids: List[ObjectId] = []
    related_user_ids: List[ObjectId] = []

    # Status
    consolidated: bool = False                # Has this been absorbed into consolidation?

    # Optional: for future retrieval (deferred, not MVP)
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
```

### 5.5 Memory Formation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REVISED MEMORY FORMATION FLOW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Chat Messages                                                               │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ maybe_form_memories() - Trigger checks (unchanged):         │            │
│  │   • Message count >= 45                                     │            │
│  │   • OR Token count >= 1000                                  │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ extract_observations() - SINGLE LLM CALL                    │            │
│  │                                                              │            │
│  │   Input: conversation_text, active_scopes                    │            │
│  │   Output: List[Observation] with scope assignments           │            │
│  │                                                              │            │
│  │   The LLM decides:                                           │            │
│  │     • What's worth remembering (content)                     │            │
│  │     • Who it's about (scope assignment)                      │            │
│  │     • No type categorization needed                          │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ save_observations()                                          │            │
│  │   • Store Observation documents                              │            │
│  │   • Add IDs to relevant scope's pending_observation_ids      │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ maybe_consolidate() - Per scope, if threshold met           │            │
│  │                                                              │            │
│  │   If len(pending_observation_ids) >= consolidation_threshold:│            │
│  │     1. Load pending observations                             │            │
│  │     2. LLM consolidation call                                │            │
│  │     3. Update scope.consolidation                            │            │
│  │     4. Mark observations as consolidated                     │            │
│  │     5. Clear pending_observation_ids                         │            │
│  │                                                              │            │
│  │   ERROR SAFETY: Never clear pending list until               │            │
│  │   consolidation is saved successfully                        │            │
│  └─────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ maybe_propagate() - Hierarchical reflection (async/deferred)│            │
│  │                                                              │            │
│  │   Individual observations may propagate to Group             │            │
│  │   Group observations may propagate to Collective             │            │
│  │                                                              │            │
│  │   Propagation criteria:                                      │            │
│  │     • Not private/personal information                       │            │
│  │     • Generalizable (applies beyond this user/group)         │            │
│  │     • High-value (LLM judges importance)                     │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.6 Context Assembly (Progressive Disclosure)

Key principle: **Consolidations are always in context. Observations are retrieved on-demand.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONTEXT ASSEMBLY MODEL                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ALWAYS IN CONTEXT (Static Layer):                                           │
│  ─────────────────────────────────                                           │
│    • Collective consolidation (if exists)                                    │
│    • Relevant group consolidation(s)                                         │
│    • Individual consolidation for current user                               │
│    • Recent pending observations (not yet consolidated)                      │
│                                                                              │
│  This is the agent's "memory" for most interactions.                         │
│  No retrieval needed — it's already there.                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │ <MemoryContext>                                             │            │
│  │   <CollectiveMemory>                                        │            │
│  │     [collective consolidation blob]                         │            │
│  │   </CollectiveMemory>                                       │            │
│  │                                                             │            │
│  │   <GroupMemory group="Eden Team">                           │            │
│  │     [group consolidation blob]                              │            │
│  │   </GroupMemory>                                            │            │
│  │                                                             │            │
│  │   <UserMemory user="gene">                                  │            │
│  │     [individual consolidation blob]                         │            │
│  │                                                             │            │
│  │     <RecentObservations>                                    │            │
│  │       - observation 1 (2 hours ago)                         │            │
│  │       - observation 2 (30 minutes ago)                      │            │
│  │     </RecentObservations>                                   │            │
│  │   </UserMemory>                                             │            │
│  │ </MemoryContext>                                            │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
│  ON-DEMAND RETRIEVAL (Dynamic Layer):                                        │
│  ────────────────────────────────────                                        │
│    • Triggered by HEURISTICS, not LLM classification                         │
│    • Searches observation log for specific recall                            │
│    • Results appended to context for that turn only                          │
│                                                                              │
│  Heuristic triggers:                                                         │
│    • Contains "remember", "recall", "what did I say about"                   │
│    • Contains question + past tense verb                                     │
│    • References proper noun not in current consolidation                     │
│    • Explicit user request for memory search                                 │
│                                                                              │
│  When triggered:                                                             │
│    1. Simple text search on observation.content (NOT vector search)          │
│    2. Filter by scope (respect privacy boundaries)                           │
│    3. Return top N matches                                                   │
│    4. Append to context as <RetrievedObservations>                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.7 Retrieval: Heuristics Over LLM

**NOT THIS** (original proposal):
```python
# Bad: LLM call on every message
query = await generate_rag_query(conversation, user_message)  # 100-300ms
if query:
    results = await vector_search(query)
```

**THIS INSTEAD**:
```python
# Good: Simple heuristics, no LLM call
def should_retrieve(user_message: str, consolidation: str) -> bool:
    """Fast heuristic check — no LLM call."""

    message_lower = user_message.lower()

    # Explicit recall requests
    recall_phrases = [
        "remember when", "what did i say", "what did i tell you",
        "do you recall", "did i mention", "what was that",
        "you told me", "i told you", "we discussed"
    ]
    if any(phrase in message_lower for phrase in recall_phrases):
        return True

    # Question about past (question word + past tense indicators)
    question_words = ["what", "when", "where", "who", "why", "how"]
    past_indicators = ["was", "were", "did", "had", "said", "told", "mentioned"]
    has_question = any(word in message_lower for word in question_words)
    has_past = any(word in message_lower for word in past_indicators)
    if has_question and has_past:
        return True

    # Proper nouns not in consolidation (potential specific recall)
    # This could be enhanced with NER, but simple capitalization check works
    words = user_message.split()
    capitalized = [w for w in words if w[0].isupper() and w.lower() not in consolidation.lower()]
    if len(capitalized) >= 2:  # Multiple unknown proper nouns
        return True

    return False


async def retrieve_observations(
    user_message: str,
    scope_ids: List[ObjectId],
    limit: int = 5
) -> List[Observation]:
    """Simple text search — no vector embeddings needed."""

    # Extract key terms from message
    # Simple approach: use significant words (skip stopwords)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "do", "did", ...}
    terms = [w.lower() for w in user_message.split() if w.lower() not in stopwords]

    # MongoDB text search (built-in, no Atlas required)
    query = {
        "scope_id": {"$in": scope_ids},
        "$text": {"$search": " ".join(terms)}
    }

    results = await Observation.find(
        query,
        sort=[("score", {"$meta": "textScore"})],
        limit=limit
    )

    return results
```

### 5.8 Extraction Prompt (Simplified)

Single LLM call, no type categorization:

```python
OBSERVATION_EXTRACTION_PROMPT = """
Extract observations from this conversation that are worth remembering.

<conversation>
{conversation_text}
</conversation>

<context>
Current user: {user_name}
User's group(s): {group_names}
</context>

<existing_memory>
{current_consolidations}
</existing_memory>

For each observation, provide:
1. content: The observation (≤50 words, factual and specific)
2. scope: Who this is about
   - "individual" = specific to this user only
   - "group:{group_name}" = relevant to a specific group
   - "collective" = applies to all users / general learning

Guidelines:
- Extract 0-5 observations (only what's genuinely worth remembering)
- Prefer specific, actionable information over vague impressions
- Don't duplicate what's already in existing memory
- For scope, default to "individual" unless clearly generalizable
- Personal preferences, instructions, corrections → individual
- Project/team facts, shared context → group
- General patterns, universal learnings → collective

Return JSON:
{
  "observations": [
    {"content": "...", "scope": "individual"},
    {"content": "...", "scope": "group:Eden Team"},
    {"content": "...", "scope": "collective"}
  ]
}
"""
```

### 5.9 Consolidation Prompt

```python
CONSOLIDATION_PROMPT = """
Update the memory consolidation by incorporating new observations.

<current_consolidation>
{current_consolidation or "No existing consolidation yet."}
</current_consolidation>

<new_observations>
{observations_text}
</new_observations>

<scope_context>
Scope type: {scope_type}
{scope_details}
</scope_context>

Instructions:
1. Synthesize the new observations into the existing consolidation
2. Resolve any contradictions (newer information takes precedence)
3. Remove information that's no longer relevant
4. Keep the consolidation focused and concise (≤{max_words} words)
5. Write in a format that's useful for an AI assistant to reference

The consolidation should read as a coherent summary, not a list of facts.
It should enable the assistant to understand and serve this {scope_type} effectively.

Return only the updated consolidation text, no JSON or formatting.
"""
```

### 5.10 Hierarchical Propagation (Deferred)

Observations at lower scopes can propagate upward when appropriate. This is a background process, not blocking.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL PROPAGATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Individual → Group propagation:                                             │
│  ────────────────────────────────                                            │
│  When: Individual scope consolidates                                         │
│  What: LLM evaluates if any observations are group-relevant                  │
│  How: Create new observations in group scope                                 │
│  Example: "Gene mentioned the project deadline is Friday"                    │
│           → Propagate to "Eden Team" group                                   │
│                                                                              │
│  Group → Collective propagation:                                             │
│  ────────────────────────────────                                            │
│  When: Group scope consolidates                                              │
│  What: LLM evaluates if any patterns are universal                           │
│  How: Create new observations in collective scope                            │
│  Example: "Eden Team prefers async communication"                            │
│           → If multiple groups show this, propagate to collective            │
│                                                                              │
│  Privacy rules:                                                              │
│  ─────────────                                                               │
│  - Personal information NEVER propagates (preferences, contact info, etc.)   │
│  - Confidential flags respected                                              │
│  - User can mark observations as "private" to prevent propagation            │
│                                                                              │
│  Implementation: Run as background job, not in request path                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Plan

### Overview: Three Phases, Not Six

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SIMPLIFIED IMPLEMENTATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Core System (MVP)                                                  │
│  ─────────────────────────────                                               │
│  • New data models (MemoryScope, Observation)                                │
│  • Single-call extraction                                                    │
│  • Consolidation at individual scope                                         │
│  • Basic context assembly (consolidations only)                              │
│                                                                              │
│  Deliverable: Working memory system, simpler than current                    │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│  Phase 2: Full Hierarchy                                                     │
│  ───────────────────────────                                                 │
│  • Group and collective scopes                                               │
│  • Hierarchical propagation                                                  │
│  • Heuristic-triggered retrieval                                             │
│                                                                              │
│  Deliverable: Complete scope hierarchy with retrieval                        │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│  Phase 3: Enhancements (If Needed)                                           │
│  ─────────────────────────────────                                           │
│  • Vector embeddings for retrieval (only if heuristics insufficient)         │
│  • Advanced propagation rules                                                │
│  • Performance optimization                                                  │
│                                                                              │
│  Deliverable: Optimized system based on real usage data                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Core System (MVP)

**Goal**: Replace current memory formation with simplified consolidation-forward system.

**Files to Create**:
```
eve/agent/memory2/
├── __init__.py              # Public exports
├── models.py                # MemoryScope, Observation
├── constants.py             # Thresholds, limits
├── prompts.py               # Extraction, consolidation prompts
├── extraction.py            # Single-call observation extraction
├── consolidation.py         # Consolidation logic
├── formation.py             # maybe_form_memories(), form_memories()
├── context.py               # assemble_memory_context()
└── service.py               # MemoryService facade
```

**Deliverables**:

1. **`models.py`**:
   ```python
   class MemoryScope(Document):
       agent_id: ObjectId
       scope_type: Literal["individual", "group", "collective"]
       user_id: Optional[ObjectId] = None
       group_id: Optional[ObjectId] = None
       consolidation: Optional[str] = None
       pending_observation_ids: List[ObjectId] = []
       consolidation_threshold: int = 10
       consolidation_max_words: int = 500
       # ...

   class Observation(Document):
       scope_id: ObjectId
       content: str
       observed_at: datetime
       source_session_id: ObjectId
       consolidated: bool = False
       # ...
   ```

2. **`extraction.py`** — Single LLM call:
   ```python
   async def extract_observations(
       conversation_text: str,
       user: User,
       groups: List[Group],
       existing_consolidations: Dict[str, str]
   ) -> List[Observation]:
       """Single LLM call to extract all observations with scope assignments."""
   ```

3. **`consolidation.py`** — Error-safe consolidation:
   ```python
   async def consolidate_scope(scope: MemoryScope) -> None:
       """
       Merge pending observations into consolidation.
       CRITICAL: Never clear pending_observation_ids until consolidation saved.
       """
   ```

4. **`context.py`** — Simple assembly:
   ```python
   async def assemble_memory_context(
       session: Session,
       agent: Agent,
       user: User
   ) -> str:
       """Assemble XML context from consolidations (static layer only)."""
   ```

**Testing Checkpoint**:
- [ ] Can create MemoryScope and Observation documents
- [ ] Single LLM call extracts observations with scope assignments
- [ ] Consolidation merges observations correctly
- [ ] Context assembly returns valid XML with consolidations
- [ ] Error-safe: failed consolidation doesn't lose observations

### Phase 2: Full Hierarchy

**Goal**: Add group/collective scopes and heuristic-triggered retrieval.

**Files to Modify/Create**:
```
eve/agent/memory2/
├── (modify) context.py      # Add heuristic retrieval
├── (modify) formation.py    # Add propagation triggers
├── retrieval.py             # Heuristic triggers, text search
└── propagation.py           # Hierarchical propagation logic
```

**Deliverables**:

1. **`retrieval.py`** — Heuristic-triggered search:
   ```python
   def should_retrieve(user_message: str, consolidation: str) -> bool:
       """Fast heuristic check — no LLM call."""

   async def retrieve_observations(
       user_message: str,
       scope_ids: List[ObjectId],
       limit: int = 5
   ) -> List[Observation]:
       """Simple text search on observations."""
   ```

2. **`propagation.py`** — Hierarchical flow:
   ```python
   async def maybe_propagate_to_group(
       individual_scope: MemoryScope,
       observations: List[Observation]
   ) -> None:
       """Evaluate and propagate individual observations to group scope."""

   async def maybe_propagate_to_collective(
       group_scope: MemoryScope,
       observations: List[Observation]
   ) -> None:
       """Evaluate and propagate group observations to collective scope."""
   ```

3. **Updated `context.py`**:
   ```python
   async def assemble_memory_context(
       session: Session,
       agent: Agent,
       user: User,
       user_message: str  # Needed for retrieval heuristics
   ) -> str:
       """
       Assemble context with:
       1. Static layer: consolidations (always present)
       2. Dynamic layer: retrieved observations (if heuristics trigger)
       """
   ```

**Testing Checkpoint**:
- [ ] Group and collective scopes work correctly
- [ ] Heuristic triggers fire appropriately (not too often, not too rarely)
- [ ] Text search retrieves relevant observations
- [ ] Propagation respects privacy boundaries
- [ ] Full context assembly includes all layers

### Phase 3: Enhancements (If Needed)

**Goal**: Add optimizations based on real usage data.

**Only implement if Phase 2 reveals deficiencies**:
- Vector embeddings for retrieval (if text search insufficient)
- Importance scoring on observations
- Memory decay / eviction policies
- Performance optimization (caching, batching)

**Decision criteria for vector search**:
- Text search retrieval quality < 80% relevance
- Users frequently not finding expected memories
- Observation corpus grows beyond 1000 per scope

---

## 7. Migration Strategy

### Approach: Conservative, Reversible

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MIGRATION MAPPING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  UserMemory → MemoryScope (individual)                                       │
│  ────────────────────────────────────                                        │
│  • UserMemory.content → scope.consolidation                                  │
│  • UserMemory.agent_id → scope.agent_id                                      │
│  • UserMemory.user_id → scope.user_id                                        │
│  • scope_type = "individual"                                                 │
│                                                                              │
│  AgentMemory → MemoryScope (collective)                                      │
│  ───────────────────────────────────────                                     │
│  • Merge all shard contents into single collective consolidation             │
│  • AgentMemory.content (all shards) → LLM merge → scope.consolidation        │
│  • scope_type = "collective"                                                 │
│  • Discard shard-level granularity (was arbitrary anyway)                    │
│                                                                              │
│  SessionMemory → Observation                                                 │
│  ───────────────────────────────                                             │
│  • SessionMemory.content → observation.content                               │
│  • SessionMemory.memory_type ignored (no type in new system)                 │
│  • Assign to appropriate scope based on original type:                       │
│      - "directive" → individual scope                                        │
│      - "episode" → discard (ephemeral)                                       │
│      - "fact"/"suggestion" → collective scope                                │
│                                                                              │
│  Groups: Create as needed                                                    │
│  ────────────────────────────                                                │
│  • If group concept doesn't exist in current system, create "default" group  │
│  • Or create groups based on existing segmentation logic                     │
│                                                                              │
│  SAFETY:                                                                     │
│  • Keep old collections intact                                               │
│  • Run migration to new collections only                                     │
│  • Feature flag to switch between old/new systems                            │
│  • Ability to rollback by flipping flag                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Migration Script Outline

```python
async def migrate_memory_system():
    """
    One-time migration from old memory system to new.
    Safe: old collections preserved, feature flag controls routing.
    """

    # 1. Migrate UserMemory → individual MemoryScopes
    async for user_memory in UserMemory.find_all():
        scope = MemoryScope(
            agent_id=user_memory.agent_id,
            scope_type="individual",
            user_id=user_memory.user_id,
            consolidation=user_memory.content,  # Direct copy
            consolidation_updated_at=user_memory.last_updated_at,
        )
        await scope.save()

    # 2. Migrate AgentMemory shards → collective MemoryScope
    # Group all shards by agent_id, merge into single consolidation
    agent_ids = await AgentMemory.distinct("agent_id")
    for agent_id in agent_ids:
        shards = await AgentMemory.find({"agent_id": agent_id, "is_active": True})

        # Merge shard contents (could use LLM for intelligent merge)
        merged_content = "\n\n".join([
            f"## {shard.shard_name}\n{shard.content}"
            for shard in shards if shard.content
        ])

        scope = MemoryScope(
            agent_id=agent_id,
            scope_type="collective",
            consolidation=merged_content,
        )
        await scope.save()

    # 3. Migrate relevant SessionMemory → Observations
    # Only migrate directives (user-specific), skip ephemeral episodes
    async for session_memory in SessionMemory.find({"memory_type": "directive"}):
        # Find the individual scope for this user
        scope = await MemoryScope.find_one({
            "agent_id": session_memory.agent_id,
            "scope_type": "individual",
            "user_id": {"$in": session_memory.related_users}
        })

        if scope:
            observation = Observation(
                scope_id=scope.id,
                content=session_memory.content,
                observed_at=session_memory.created_at,
                source_session_id=session_memory.source_session_id,
                consolidated=True,  # Already absorbed into old consolidation
            )
            await observation.save()

    # 4. Validate
    await validate_migration()


async def validate_migration():
    """Verify data integrity after migration."""

    old_user_count = await UserMemory.count()
    new_individual_count = await MemoryScope.count({"scope_type": "individual"})
    assert old_user_count == new_individual_count, "User memory count mismatch"

    # ... additional validation checks
```

---

## Appendix A: Comparison with Original Proposal

| Aspect | Original Proposal | Revised Architecture |
|--------|-------------------|---------------------|
| **Memory ontology** | 4 types (episode, directive, fact, suggestion) | 1 type (observation) with scope |
| **Primary artifact** | Atomic memories with RAG | Consolidation blobs |
| **Secondary artifact** | Consolidations | Observation log |
| **Scope model** | Arbitrary shards | Hierarchy (individual → group → collective) |
| **Retrieval trigger** | LLM call on every message | Simple heuristics |
| **Retrieval method** | Vector search + text search + RRF | Text search only (vector deferred) |
| **LLM calls for extraction** | N+1 (per shard) | 1 (single call) |
| **LLM calls for retrieval** | 1 per message | 0 (heuristics only) |
| **Implementation phases** | 6 phases | 3 phases |
| **MongoDB requirements** | Atlas vector search indexes | Standard text indexes only |
| **Complexity** | High (full RAG pipeline) | Low (consolidations + text search) |

## Appendix B: When to Add Vector Search

Vector search should be **deferred** until empirical evidence shows it's needed. Indicators:

1. **Retrieval quality issues**:
   - Users frequently say "I told you about X" but agent doesn't recall
   - Text search returns irrelevant results
   - Heuristics trigger but retrieved observations aren't helpful

2. **Scale issues**:
   - Observation corpus exceeds 1000 per scope
   - Text search becomes slow (>100ms)

3. **Semantic matching needs**:
   - Users describe things differently than stored ("my dietary restrictions" vs "I'm vegetarian")
   - Synonym/paraphrase matching needed

If these issues emerge, Phase 3 adds embeddings to the existing Observation model and introduces hybrid search. The architecture supports this extension without redesign.

## Appendix C: Open Questions

1. **Group definition**: How are groups created and managed? User-defined? Inferred? Admin-configured?

2. **Propagation frequency**: How often should propagation run? Every consolidation? Daily batch?

3. **Privacy controls**: Should users be able to mark observations as "never propagate"?

4. **Consolidation triggers**: Is count-based (N observations) the right trigger, or should it be time-based or content-based?

5. **Cross-agent memory**: Should collective memory be per-agent or truly global? Current design is per-agent.

---

*End of document. Last updated: 2025-12-21*
