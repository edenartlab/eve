# Eden Memory System: Redesign Proposal v2

*Document updated: 2025-12-26*
*Purpose: Comprehensive reference for memory system refactoring*

---

## Current System Reference (Production Context)

> **Note**: This section summarizes the current working implementation documented in `memory_current.md`. Understanding this context is essential for a successful migration to the new architecture.

### What's Currently Working in Production

**Memory Types (Current → New Mapping)**:
| Current Type | Storage | New Equivalent |
|--------------|---------|----------------|
| `episode` | SessionMemory (FIFO, 8 items) | Session reflections |
| `directive` | SessionMemory → UserMemory consolidation | User reflections |
| `fact` | SessionMemory (FIFO, 100 items per shard) | Facts (RAG) |
| `suggestion` | SessionMemory → AgentMemory consolidation | Agent reflections |

**Production Trigger Thresholds** (currently working well):
```python
MEMORY_FORMATION_MSG_INTERVAL = 45      # Messages between formations
MEMORY_FORMATION_TOKEN_INTERVAL = 1000  # ~1k tokens triggers formation
NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = 4  # Minimum before attempting
CONSIDER_COLD_AFTER_MINUTES = 10        # Session inactivity threshold
```

**Production Word Limits** (tuned values):
```python
SESSION_EPISODE_MEMORY_MAX_WORDS = 50   # Per episode item
SESSION_DIRECTIVE_MEMORY_MAX_WORDS = 25 # Per directive item
SESSION_SUGGESTION_MEMORY_MAX_WORDS = 35 # Per suggestion item
SESSION_FACT_MEMORY_MAX_WORDS = 30      # Per fact item
USER_MEMORY_BLOB_MAX_WORDS = 400        # User consolidated blob
AGENT_MEMORY_BLOB_MAX_WORDS = 1000      # Agent shard consolidated blob
```

**Existing Collections** (for migration awareness):
- `memory_sessions` - Raw extracted memories (SessionMemory model)
- `memory_user` - Consolidated user blobs (UserMemory model)
- `memory_agent` - Agent shards with consolidated blobs (AgentMemory model)

**Key Production Patterns to Preserve**:
1. **Async memory formation** - Never blocks user-facing responses
2. **Cold session processing** - Background job processes inactive sessions (10+ min idle with ≥4 messages)
3. **Character weighting** - `messages_to_text()` weights sources: USER=1.0, TOOL=0.5, AGENT=0.2, OTHER=0.5
4. **Sync across sessions** - Memory context synced every 5 minutes across active sessions
5. **Session caching** - `memory_context` cached on session object, refreshed on staleness check

**Current LLM Models**:
```python
MEMORY_LLM_MODEL_FAST = "gpt-5-mini"    # For extraction
MEMORY_LLM_MODEL_SLOW = "gpt-5.1"       # For consolidation
```

**Consolidation Triggers** (current values, consider for new system):
```python
MAX_USER_MEMORIES_BEFORE_CONSOLIDATION = 5   # Directives before consolidation
MAX_AGENT_MEMORIES_BEFORE_CONSOLIDATION = 16 # Suggestions before consolidation
MAX_N_EPISODES_TO_REMEMBER = 8              # Session episode FIFO limit
MAX_FACTS_PER_SHARD = 100                   # Facts FIFO limit (per shard)
```

**Known Issues Being Addressed by New Architecture**:
1. Multiple LLM calls per formation (1 + N for N shards) → New: 2 sequential calls max
2. No semantic deduplication for facts → New: mem0-inspired ADD/UPDATE/DELETE/NONE
3. No session-level consolidation → New: Session reflections consolidate too
4. Hardcoded extraction prompts for user memory → New: Unified reflection extraction
5. No vector/RAG retrieval → New: MongoDB Atlas vector search

---

## Table of Contents
1. [Core Concepts](#1-core-concepts)
2. [Memory Types: Facts vs Reflections](#2-memory-types-facts-vs-reflections)
3. [Memory Scope](#3-memory-scope)
4. [Extraction Architecture](#4-extraction-architecture)
5. [Storage Architecture](#5-storage-architecture)
6. [Facts Management (Deduplication & Conflict Resolution)](#6-facts-management-deduplication--conflict-resolution)
7. [Always-in-Context Memory System](#7-always-in-context-memory-system)
8. [RAG System (Standalone)](#8-rag-system-standalone)
9. [Data Models](#9-data-models)
10. [Implementation Plan](#10-implementation-plan)
11. [MongoDB Atlas Setup](#11-mongodb-atlas-setup)

---

## 1. Core Concepts

### 1.1 Two Independent Memory Systems

The memory architecture consists of two **completely independent** systems that can be enabled/disabled separately:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO INDEPENDENT MEMORY SYSTEMS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐  │
│  │  ALWAYS-IN-CONTEXT MEMORY       │   │     RAG MEMORY (Facts)          │  │
│  │  (Reflections)                  │   │     (Standalone, Toggle-able)   │  │
│  ├─────────────────────────────────┤   ├─────────────────────────────────┤  │
│  │                                 │   │                                 │  │
│  │  • Always injected into contex  │   │  • Stored in vector database    │  │
│  │  • Contains agent ideas/plans   │   │  • Retrieved via tool call      │  │
│  │  • Evolves over time            │   │  • Only when explicitly needed  │  │
│  │  • Buffer → Consolidation       │   │  • Infinitely scalable          │  │
│  │                                 │   │                                 │  │
│  │  Memory Type: "reflection"      │   │  Memory Type: "fact"            │  │
│  │                                 │   │                                 │  │
│  └─────────────────────────────────┘   └─────────────────────────────────┘  │
│                                                                             │
│            CAN BE ENABLED/DISABLED INDEPENDENTLY                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Terminology

| Term | Definition |
|------|------------|
| **Fact** | Atomic, objective statement extracted from conversation. Stored in vector DB for RAG retrieval. Stands alone without context. |
| **Reflection** | Interpreted memory that evolves the agent's character. Always in context. Formed with awareness of existing memory state. Can disappear over time (when overwritten by a consolidation).|
| **Scope** | Where a memory lives / can be fetched: `session`, `user`, `agent` |
| **Consolidation** | Process of merging buffered reflections into a condensed blob. Keeps always-in-context memory size capped |
| **Always-in-Context** | Memory content injected into agent context, regardless of RAG query / toolcall. |

---

## 2. Memory Types: Facts vs Reflections

### 2.1 Facts

**Purpose**: Store atomic, factual information for semantic retrieval.

**Characteristics**:
- Objective statements that stand alone
- No dependency on existing memory context for extraction
- Stored in vector database with embeddings
- Retrieved via RAG when query is relevant
- Never automatically in context (only via explicit retrieval)
- Infinitely scalable storage

**Examples**:
- "Xander's birthday is March 15th"
- "Gene loves hockey"
- "The project deadline is January 30th"
- "User works at Acme Corp as a senior engineer"

### 2.2 Reflections

**Purpose**: Form and evolve the agent's persona/behavior/plans/projects through interpreted observations. (Acts like a scratchpad for the agent.)

**Characteristics**:
- Subjective, interpreted memories that evolve over time
- Extracted **with awareness of existing reflections memory**
- Always in context (influences every response)
- Buffer accumulates → triggers consolidation
- Can track preferences, ongoing projects, ideas, plans, roadmap, ...

**Examples**:
- "User seems frustrated with slow responses - prioritize conciseness"
- "Conversation style has shifted to more technical depth"
- "User appreciates when I proactively suggest alternatives"
- "We are currently organising a collective dinner for Friday Nov 4th"
- "The remaining todo's are: ..."

---

## 3. Memory Scope

### 3.1 Scope Definition

Memory scope determines visibility. **Facts and reflections have different valid scopes:**

```python
# FACTS: Only user/agent scope (no session - see Section 4.2 for rationale)
fact_scope: List[Literal["user", "agent"]]

# REFLECTIONS: All three scopes (session reflections consolidate like others)
reflection_scope: List[Literal["session", "user", "agent"]]
```

| Scope | Visibility | Facts | Reflections |
|-------|------------|-------|-------------|
| `session` | Only during this conversation session | ❌ No | ✅ Yes |
| `user` | Across all sessions with this user | ✅ Yes | ✅ Yes |
| `agent` | All conversations with all users | ✅ Yes | ✅ Yes |

Scope hierarchy: `agent` supersedes `user` supersedes `session`.

### 3.2 Scope Examples

**Facts** (user/agent scope only - stored in vector DB for RAG):
```python
# User-specific fact
{"content": "Gene's birthday is March 15th", "scope": ["user"]}

# Agent-wide fact
{"content": "Product launch is scheduled for Q2", "scope": ["agent"]}

# Both scopes (user fact that's also relevant agent-wide)
{"content": "Xander is a team lead designing the memory system", "scope": ["agent"]}
```

**Reflections** (all scopes - always in context, consolidate over time):
```python
# Session reflection: rolling summary of what's happening in the current session:
{"content": "We are debugging the login bug, tried 3 approaches so far, next steps: ...", "scope": ["session"]}
{"content": "We are creating a short story about dolphins", "scope": ["session"]}

# User reflection: how to interact with this user
{"content": "User (Xander) prefers concise responses in Spanish", "scope": ["user"]}
{"content": "User (Gene) always wants to give explicit confirmation before running video toolcalls", "scope": ["user"]}

# Agent reflection: agent updates
{"content": "Memory system v2 is the current priority project", "scope": ["agent"]}
{"content": "Eden is struggling finding product-market-fit and needs better marketing approach", "scope": ["agent"]}
```

### 3.3 Scope Filtering

```python
# For always-in-context memory (reflections):
#   - Include agent-level consolidated blob + recent unabsorbed
#   - Include user-level consolidated blob + recent unabsorbed (for current user)
#   - Include session-level consolidated blob + recent unabsorbed (for current session)

# For RAG retrieval (facts only):
#   - Include all agent-scoped facts
#   - Include user-scoped facts for current user
#   - NO session-scoped facts exist (by design)
```

---

## 4. Extraction Architecture

### 4.1 Sequential LLM Calls (Facts → Reflections)

**Key insight**: Facts and reflections have fundamentally different context requirements. Reflections benefit from knowing what facts were just extracted to avoid redundancy. Since memory formation is async, the sequential dependency has no impact on user-facing latency.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTRACTION ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Conversation Messages                                                      │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LLM CALL 1: FACTS (Fast/Cheap Model)                                │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  INPUT:                                                              │   │
│  │    • Conversation text                                               │   │
│  │    • (no memory context needed)                                      │   │
│  │                                                                      │   │
│  │  OUTPUT:                                                             │   │
│  │    • List of facts with scope: "user" and/or "agent"                 │   │
│  │    • NO session-scoped facts (see 4.2)                               │   │
│  │                                                                      │   │
│  │  COST: Low | LATENCY: Fast                                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         │ newly_formed_facts passed to next call                            │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LLM CALL 2: REFLECTIONS (Full Context Model)                        │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  INPUT:                                                              │   │
│  │    • Conversation text                                               │   │
│  │    • Current consolidated memory blobs (agent, user, session)        │   │
│  │    • Recent unabsorbed reflections                                   │   │
│  │    • NEWLY FORMED FACTS from Call 1  ← enables deduplication         │   │
│  │                                                                      │   │
│  │  OUTPUT (hierarchical extraction order - see 4.4):                   │   │
│  │    1. Agent reflections   → extracted first (broadest scope)         │   │
│  │    2. User reflections    → only what's NOT in agent reflections     │   │
│  │    3. Session reflections → only what's NOT in agent/user            │   │
│  │                                                                      │   │
│  │  COST: Medium | LATENCY: Medium                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ├───────────────────────────────────┬───────────────────────────┐   │
│         ▼                                   ▼                           │   │
│  ┌────────────────────────┐      ┌────────────────────────────────────┐ │   │
│  │   Vector Store         │      │   Reflection Buffer                │ │   │
│  │   (RAG-indexed facts)  │      │   (→ Consolidation at ALL 3        │ │   │
│  │                        │      │    scope levels: agent/user/sess)  │ │   │
│  └────────────────────────┘      └────────────────────────────────────┘ │   │
│                                                                         │   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Why No Session-Scoped Facts?

Facts are stored in the vector database for RAG retrieval. Session-scoped facts would:
- Bloat the vector DB with stale data (never retrieved after session ends)
- Increase RAG latency/cost as DB grows
- Be redundant with session reflections

**Decision**: Facts only have `user` and/or `agent` scope. Session-level context is handled entirely by session reflections, which consolidate into a rolling summary of what happened.

### 4.3 Why Sequential Calls?

| Aspect | Facts | Reflections |
|--------|-------|-------------|
| **Context needed** | Conversation only | Conversation + memory + newly formed facts |
| **Model** | Fast/cheap (gemini-3-flash-preview) | Full capability (gemini-3-flash-preview) |
| **Purpose** | Extract objective data | Evolve persona |
| **Prompt complexity** | Simple extraction | Complex (aware of state + deduplication) |
| **Scope** | user/agent only | session/user/agent |

**Benefits of sequential extraction**:
1. Facts extraction uses cheaper, faster model
2. Reflections know what facts were captured → avoids redundancy
3. Clear separation of concerns
4. Reflections focus on *interpretive* insights beyond raw facts
5. Memory formation is async → no impact on user latency

### 4.4 Extraction Prompts

**Fact Extraction Prompt** (no memory context needed):
```
You are extracting factual information from a conversation.

<conversation>
{conversation_text}
</conversation>

Extract atomic facts - objective statements that stand alone.
For each fact, determine its scope:
- "user": Relevant across all conversations with this specific user
- "agent": Relevant to all conversations for this agent (across all users)

Note: Do NOT extract session-only facts. Session context is handled by reflections.

Return JSON:
{
  "facts": [
    {"content": "...", "scope": ["user"]},
    {"content": "...", "scope": ["agent"]},
    {"content": "...", "scope": ["user", "agent"]}
  ]
}

Guidelines:
- Facts should be objective, verifiable statements
- Maximum 30 words per fact
- Only extract genuinely new information
- Return empty list if no facts found
```

**Reflection Extraction Prompt** (needs memory context + newly formed facts):
```
You are updating an agent's memory based on a conversation.

<current_agent_memory>
{consolidated_agent_blob}
Recent: {recent_agent_reflections}
</current_agent_memory>

<current_user_memory>
{consolidated_user_blob}
Recent: {recent_user_reflections}
</current_user_memory>

<current_session_memory>
{consolidated_session_blob}
Recent: {recent_session_reflections}
</current_session_memory>

<newly_formed_facts>
{facts_from_call_1}
</newly_formed_facts>

<conversation>
{conversation_text}
</conversation>

Extract reflections in HIERARCHICAL ORDER. Each level should only contain
information NOT already captured at a higher level or in the newly formed facts.

## EXTRACTION ORDER (follow strictly):

### 1. AGENT REFLECTIONS (extract first - broadest scope)
Things the agent should always have in context across ALL conversations:
- Updates to the agent's persona, character, or communication style
- Ongoing projects, plans, ideas, roadmaps
- Domain knowledge and insights that apply universally
- Learnings that shape how the agent operates

### 2. USER REFLECTIONS (extract second - exclude what's in agent reflections)
Things specific to THIS USER that aren't agent-wide:
- User preferences and directives ("always respond in Spanish")
- User's expertise level and communication style preferences
- User-specific facts that should always be in context (favorite color, etc.)
- How to interact with this specific user

### 3. SESSION REFLECTIONS (extract last - exclude what's in agent/user reflections)
Things specific to THIS SESSION that aren't captured above:
- Current status of ongoing tasks/toolcalls in this session
- Session-specific instructions given at the start
- What has happened so far (rolling summary for long sessions)
- Temporary context that will expire with the session

Return JSON:
{
  "agent_reflections": [
    {"content": "..."}
  ],
  "user_reflections": [
    {"content": "..."}
  ],
  "session_reflections": [
    {"content": "..."}
  ]
}

Guidelines:
- Reflections are subjective, interpreted insights (not raw facts)
- Do NOT duplicate information already in newly_formed_facts
- Each reflection max 35 words
- Lower scopes should ONLY contain info not captured at higher scopes
- Return empty arrays if no meaningful reflections at that level
```

---

## 5. Storage Architecture

### 5.1 Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         FACTS STORAGE                                  │ │
│  │                    (Vector DB - MongoDB Atlas)                         │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  Collection: memory_facts                                              │ │
│  │                                                                        │ │
│  │  {                                                                     │ │
│  │    "_id": ObjectId,                                                    │ │
│  │    "content": "User prefers dark mode",                                │ │
│  │    "embedding": [0.123, -0.456, ...],  // 1536-dim vector              │ │
│  │    "scope": ["user"],                                                  │ │
│  │    "agent_id": ObjectId,                                               │ │
│  │    "user_id": ObjectId,           // if scope includes "user"          │ │
│  │    "session_id": ObjectId,        // if scope includes "session"       │ │
│  │    "formed_at": datetime,                                              │ │
│  │    "hash": "md5_hash",            // deduplication                     │ │
│  │    "access_count": 0,             // RAG retrieval tracking            │ │
│  │    "last_accessed_at": null       // When this memory was last queried │ │
│  │  }                                                                     │ │
│  │                                                                        │ │
│  │  Indexes: vector_index, scope filters                                  │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      REFLECTIONS STORAGE                               │ │
│  │                (Buffer + Consolidated Blobs)                           │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  Collection: memory_reflections (buffer)                               │ │
│  │  {                                                                     │ │
│  │    "_id": ObjectId,                                                    │ │
│  │    "content": "User prefers concise responses",                        │ │
│  │    "scope": ["user"],                                                  │ │
│  │    "agent_id": ObjectId,                                               │ │
│  │    "user_id": ObjectId,                                                │ │
│  │    "session_id": ObjectId,                                             │ │
│  │    "formed_at": datetime,                                              │ │
│  │    "absorbed": false              // true after consolidation          │ │
│  │  }                                                                     │ │
│  │                                                                        │ │
│  │  Collection: memory_consolidated                                       │ │
│  │  {                                                                     │ │
│  │    "_id": ObjectId,                                                    │ │
│  │    "scope_key": {"agent_id": ..., "user_id": ...},                     │ │
│  │    "consolidated_content": "Blob of merged reflections...",            │ │
│  │    "unabsorbed_ids": [ObjectId, ...],                                  │ │
│  │    "last_consolidated_at": datetime                                    │ │
│  │  }                                                                     │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Fact Storage (mem0-inspired)

Following mem0's approach for vector storage:

```python
# Payload structure for facts (adapted from mem0)
fact_document = {
    "_id": ObjectId(),
    "content": "The actual fact text",
    "hash": md5(content),              # For deduplication
    "embedding": [float, ...],         # 1536-dim vector

    # Scope
    "scope": ["user", "agent"],        # List of applicable scopes
    "agent_id": ObjectId,
    "user_id": ObjectId | None,        # Set if "user" in scope
    "session_id": ObjectId | None,     # Set if "session" in scope

    # Temporal
    "formed_at": datetime,
    "created_at": datetime,

    # Access tracking (for future importance scoring)
    "access_count": 0,
    "last_accessed_at": None
}
```

### 5.3 Reflection Storage

```python
# Individual reflection (buffer)
reflection_document = {
    "_id": ObjectId(),
    "content": "The reflection text",
    "scope": ["user"],
    "agent_id": ObjectId,
    "user_id": ObjectId | None,
    "session_id": ObjectId | None,
    "formed_at": datetime,
    "absorbed": False                  # True after consolidation
}

# Consolidated blob
consolidated_document = {
    "_id": ObjectId(),
    "scope_key": {
        "agent_id": ObjectId,
        "user_id": ObjectId | None,    # None for agent-level consolidation
        "session_id": ObjectId | None  # None for user/agent level
    },
    "consolidated_content": "Merged reflection blob...",
    "unabsorbed_ids": [ObjectId, ...], # Reflections awaiting consolidation
    "last_consolidated_at": datetime,
    "word_limit": 400                  # Max words for this blob
}
```

---

## 6. Facts Management (Deduplication & Conflict Resolution)

### 6.1 The Problem

Simple hash-based deduplication is insufficient for fact management:

| Problem | Example | Hash Dedup Fails Because |
|---------|---------|--------------------------|
| **Semantic duplicates** | "John likes pizza" vs "John enjoys pizza" | Different text, same meaning |
| **Contradictions** | "John likes pizza" → "John hates pizza" | Both stored, conflicting info |
| **Partial updates** | "John works at Acme" → "John works at Acme as CTO" | Old fact becomes outdated |
| **Fact evolution** | "Project deadline is Jan 15" → "Project deadline moved to Feb 1" | Old deadline still in memory |

### 6.2 Solution: Memory Update Decision (mem0-inspired)

After extracting facts, we add an LLM-based decision step that compares new facts against existing memories and decides: **ADD**, **UPDATE**, **DELETE**, or **NONE**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FACTS MANAGEMENT PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Extracted Facts (from LLM Call 1)                                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  FOR EACH EXTRACTED FACT:                                               ││
│  │                                                                         ││
│  │  1. EMBED & SEARCH                                                      ││
│  │     • Embed the new fact                                                ││
│  │     • Vector search for top-K similar existing facts (same scope)       ││
│  │     • Threshold: similarity > 0.7 (retrieve candidates for comparison)  ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  2. MEMORY UPDATE DECISION (LLM Call - batched for efficiency)          ││
│  │                                                                         ││
│  │  INPUT:                                                                 ││
│  │    • New facts (list)                                                   ││
│  │    • Retrieved similar existing facts (with IDs)                        ││
│  │                                                                         ││
│  │  OUTPUT (for each fact):                                                ││
│  │    • event: "ADD" | "UPDATE" | "DELETE" | "NONE"                        ││
│  │    • id: existing fact ID (for UPDATE/DELETE/NONE)                      ││
│  │    • text: final fact text (may be merged/refined)                      ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  3. EXECUTE OPERATIONS                                                  ││
│  │                                                                         ││
│  │  ADD:    Create new vector entry (embed + store)                        ││
│  │  UPDATE: Update existing entry (re-embed + update payload)              ││
│  │  DELETE: Remove existing entry from vector store                        ││
│  │  NONE:   Skip (fact already exists unchanged)                           ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Memory Update Decision Prompt

```
You are managing a fact database. Compare new facts against existing memories
and decide what action to take for each.

<new_facts>
{list of newly extracted facts}
</new_facts>

<existing_memories>
{list of similar existing facts with their IDs, retrieved via vector search}
</existing_memories>

For each new fact, determine the appropriate action:

1. **ADD**: The fact is genuinely new information not captured by any existing memory
2. **UPDATE**: The fact updates/enhances/corrects an existing memory (return the merged/updated text)
3. **DELETE**: The fact contradicts an existing memory in a way that invalidates it (e.g., preference reversal)
4. **NONE**: The fact is already captured by an existing memory (no change needed)

Return JSON:
{
  "decisions": [
    {
      "new_fact": "John now works at TechCorp",
      "event": "UPDATE",
      "existing_id": "mem_123",
      "existing_text": "John works at Acme Corp",
      "final_text": "John works at TechCorp",
      "reasoning": "Job change - updating employer"
    },
    {
      "new_fact": "John likes pizza",
      "event": "NONE",
      "existing_id": "mem_456",
      "existing_text": "John enjoys pizza",
      "reasoning": "Semantically identical - already captured"
    },
    {
      "new_fact": "John's birthday is March 15th",
      "event": "ADD",
      "final_text": "John's birthday is March 15th",
      "reasoning": "New information not in existing memories"
    },
    {
      "new_fact": "John dislikes pizza",
      "event": "DELETE",
      "existing_id": "mem_456",
      "existing_text": "John enjoys pizza",
      "final_text": "John dislikes pizza",
      "reasoning": "Preference reversal - delete old, add new contradicting fact"
    }
  ]
}

Guidelines:
- UPDATE when new fact adds detail or corrects existing (merge intelligently)
- DELETE + ADD when there's a direct contradiction (preference changes, corrections)
- NONE for semantic duplicates (different wording, same meaning)
- ADD only for genuinely new information
- When in doubt between UPDATE and ADD, prefer UPDATE to avoid duplicates
```

### 6.4 Implementation

```python
async def process_extracted_facts(
    extracted_facts: List[str],
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str]
) -> List[FactOperation]:
    """
    Process extracted facts through the deduplication pipeline.
    Returns list of operations to execute (ADD/UPDATE/DELETE).
    """

    if not extracted_facts:
        return []

    # Step 1: For each fact, find similar existing facts
    fact_candidates = []
    for fact in extracted_facts:
        similar = await search_similar_facts(
            query=fact,
            agent_id=agent_id,
            user_id=user_id,
            scope_filter=scope_filter,
            threshold=0.7,  # Similarity threshold for candidate retrieval
            limit=5
        )
        fact_candidates.append({
            "new_fact": fact,
            "similar_existing": similar
        })

    # Step 2: Batch LLM call to decide operations
    decisions = await llm_memory_update_decision(fact_candidates)

    # Step 3: Execute operations
    operations = []
    for decision in decisions:
        if decision["event"] == "ADD":
            op = await add_fact(decision["final_text"], agent_id, user_id, scope_filter)
            operations.append(op)
        elif decision["event"] == "UPDATE":
            op = await update_fact(decision["existing_id"], decision["final_text"])
            operations.append(op)
        elif decision["event"] == "DELETE":
            await delete_fact(decision["existing_id"])
            # After DELETE, add the new contradicting fact
            if decision.get("final_text"):
                op = await add_fact(decision["final_text"], agent_id, user_id, scope_filter)
                operations.append(op)
        # NONE: no operation needed

    return operations


async def search_similar_facts(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str],
    threshold: float = 0.7,
    limit: int = 5
) -> List[dict]:
    """
    Search for semantically similar existing facts.
    Used for deduplication candidate retrieval.
    """
    embedding = await openai.embed(query, model="text-embedding-3-small")

    pre_filter = _build_scope_filter(agent_id, user_id, scope_filter)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "fact_vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": limit * 10,
                "limit": limit,
                "filter": pre_filter
            }
        },
        {
            "$addFields": {
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$match": {
                "score": {"$gte": threshold}  # Only return sufficiently similar facts
            }
        },
        {
            "$project": {
                "_id": 1,
                "content": 1,
                "score": 1,
                "scope": 1
            }
        }
    ]

    return await db.memory_facts.aggregate(pipeline).to_list()
```

### 6.5 Updated Extraction Architecture (with Facts Management)

The complete flow now includes the memory update decision step:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   COMPLETE EXTRACTION ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Conversation Messages                                                      │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LLM CALL 1: FACT EXTRACTION (Fast/Cheap Model)                      │   │
│  │  • Extract atomic facts from conversation                            │   │
│  │  • No memory context needed                                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         │ extracted_facts                                                   │
│         │                                                                   │
│         ├───────────────────────────────────────────────────────────────┐   │
│         │                                                               │   │
│         ▼                                                               │   │
│  ┌──────────────────────────────────────────────────────────────────┐   │   │
│  │  VECTOR SEARCH: Find similar existing facts                      │   │   │
│  │  • Embed each extracted fact                                     │   │   │
│  │  • Search for candidates (similarity > 0.7)                      │   │   │
│  └──────────────────────────────────────────────────────────────────┘   │   │
│         │                                                               │   │
│         ▼                                                               │   │
│  ┌──────────────────────────────────────────────────────────────────┐   │   │
│  │  LLM CALL 1.5: MEMORY UPDATE DECISION                            │   │   │
│  │  • Compare new facts vs existing                                 │   │   │
│  │  • Decide: ADD / UPDATE / DELETE / NONE                          │   │   │
│  │  • Handles duplicates, contradictions, updates                   │   │   │
│  └──────────────────────────────────────────────────────────────────┘   │   │
│         │                                                               │   │
│         ▼                                                               │   │
│  ┌──────────────────────────────────────────────────────────────────┐   │   │
│  │  EXECUTE OPERATIONS                                              │   │   │
│  │  ADD → embed + store new fact                                    │   │   │
│  │  UPDATE → re-embed + update existing                             │   │   │
│  │  DELETE → remove from vector store                               │   │   │
│  │  NONE → skip                                                     │   │   │
│  └──────────────────────────────────────────────────────────────────┘   │   │
│                                                                         │   │
│         ◄────────────────────────────────────────────────────(passed)───┘   │
│         │ newly_formed_facts (after dedup)                                  │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LLM CALL 2: REFLECTIONS EXTRACTION                                  │   │
│  │  • Conversation + memory context + newly formed facts                │   │
│  │  • Hierarchical: agent → user → session                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌────────────────────────┐      ┌────────────────────────────────────┐     │
│  │   Vector Store         │      │   Reflection Buffer                │     │
│  │   (deduplicated facts) │      │   (→ Consolidation)                │     │
│  └────────────────────────┘      └────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.6 Extended Fact Model (for tracking updates)

```python
@Collection("memory_facts")
class Fact(Document):
    # ... existing fields ...

    # Update tracking (new fields)
    version: int = 1                       # Incremented on UPDATE
    previous_content: Optional[str] = None # Content before last update
    updated_at: Optional[datetime] = None  # When last updated
    update_history: List[dict] = []        # Optional: full history
    # Example: [{"content": "old text", "updated_at": datetime, "reason": "job change"}]
```

### 6.7 Cost & Performance Considerations

| Aspect | Impact | Mitigation |
|--------|--------|------------|
| **Extra LLM call** | +1 LLM call per extraction | Use fast model, batch all facts in one call |
| **Vector searches** | N searches for N extracted facts | Batch embed all facts, run searches in parallel |
| **Latency** | Additional ~500ms-1s | Memory formation is async, no user-facing impact |
| **False positives** | May incorrectly merge distinct facts | Tune similarity threshold (0.7 default), add domain context to prompt |

**Optimization**: For high-volume scenarios, skip the update decision call if no similar facts found (similarity < 0.7 for all candidates) - direct ADD.

---

## 7. Always-in-Context Memory System

### 7.1 Overview

The always-in-context system manages reflections that shape the agent's persona.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALWAYS-IN-CONTEXT MEMORY FLOW                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Reflections Extracted                                                      │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  BUFFER LAYER                                                           ││
│  │  (Recent, unabsorbed reflections)                                       ││
│  │                                                                         ││
│  │  When buffer size >= N (e.g., 5 reflections)                            ││
│  │         │                                                               ││
│  │         ▼                                                               ││
│  │  ┌────────────────────────────────────────────────────────────────────┐ ││
│  │  │  CONSOLIDATION (LLM Call)                                          │ ││
│  │  │                                                                    │ ││
│  │  │  INPUT:                                                            │ ││
│  │  │    • Current consolidated blob                                     │ ││
│  │  │    • Buffered reflections                                          │ ││
│  │  │                                                                    │ ││
│  │  │  OUTPUT:                                                           │ ││
│  │  │    • Updated consolidated blob (max N words)                       │ ││
│  │  └────────────────────────────────────────────────────────────────────┘ ││
│  │         │                                                               ││
│  │         ▼                                                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  CONSOLIDATED LAYER                                                     ││
│  │  (Persistent, word-limited summary)                                     ││
│  │                                                                         ││
│  │  "This user prefers direct communication. They work in data science     ││
│  │   and appreciate technical depth. Conversations tend toward problem-    ││
│  │   solving mode. Be concise, skip pleasantries, focus on solutions..."   ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  CONTEXT ASSEMBLY                                                       ││
│  │                                                                         ││
│  │  <MemoryContext>                                                        ││
│  │    <AgentMemory>                                                        ││
│  │      {agent-level consolidated blob}                                    ││
│  │      Recent: {unabsorbed agent reflections}                             ││
│  │    </AgentMemory>                                                       ││
│  │                                                                         ││
│  │    <UserMemory>                                                         ││
│  │      {user-level consolidated blob}                                     ││
│  │      Recent: {unabsorbed user reflections}                              ││
│  │    </UserMemory>                                                        ││
│  │                                                                         ││
│  │    <SessionMemory>                                                      ││
│  │      {session-level consolidated blob}                                  ││
│  │      Recent: {unabsorbed session reflections}                           ││
│  │    </SessionMemory>                                                     ││
│  │  </MemoryContext>                                                       ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Consolidation

**All three scope levels consolidate.** Consolidation is triggered when buffer exceeds threshold:

```python
CONSOLIDATION_THRESHOLDS = {
    "agent": 8,    # Agent-wide reflections (persona, projects, ideas)
    "user": 4,     # User-specific reflections (preferences, directives)
    "session": 4   # Session reflections (rolling summary of what happened)
}

CONSOLIDATED_WORD_LIMITS = {
    "agent": 1000,  # Largest - agent's full persona/project state
    "user": 400,    # Medium - user preferences and interaction style
    "session": 400  # Rolling summary of session events and status
}
```

**Scope-specific consolidation purposes**:

| Scope | Consolidation Purpose |
|-------|----------------------|
| `agent` | Evolving persona, ongoing projects, domain insights, learnings |
| `user` | User preferences, directives, interaction style |
| `session` | Rolling summary of what happened, task status, temporary context |

**Consolidation Prompt** (same structure, different focus per scope):
```
You are consolidating {scope_type} memory reflections.

<current_consolidated_memory>
{existing_blob or "None yet"}
</current_consolidated_memory>

<new_reflections>
{list of buffered reflections}
</new_reflections>

Merge the new reflections into the existing memory, creating an updated summary.

{scope_specific_instructions}

General guidelines:
- Preserve important existing insights
- Integrate new information
- Resolve any contradictions (newer info takes precedence)
- Maximum {word_limit} words

Return ONLY the updated consolidated memory text.
```

**Scope-specific instructions**:
- **Agent**: Focus on persona evolution, project status, domain learnings. What should the agent always know?
- **User**: Focus on preferences, directives, interaction style. How should the agent interact with this user?
- **Session**: Focus on what happened, current task status, temporary instructions. What's the rolling context for this conversation?

### 7.3 Context Assembly

```python
async def assemble_always_in_context_memory(
    agent_id: ObjectId,
    user_id: ObjectId,
    session_id: ObjectId
) -> str:
    """
    Assembles the always-in-context memory for prompt injection.
    This gets injected into agent context on EVERY message, regardless of RAG.
    The final memory_xml can be cached inside the session object to avoid running this for every message
    """

    # Get consolidated blobs
    agent_blob = await get_consolidated("agent", agent_id=agent_id)
    user_blob = await get_consolidated("user", agent_id=agent_id, user_id=user_id)
    session_blob = await get_consolidated("session", session_id = session_id)

    # Get recent unabsorbed reflections
    agent_reflections = await get_unabsorbed("agent", agent_id=agent_id)
    user_reflections = await get_unabsorbed("user", agent_id=agent_id, user_id=user_id)
    session_reflections = await get_unabsorbed("session", session_id=session_id)

    return _build_memory_xml(
        agent_blob, agent_reflections,
        user_blob, user_reflections,
        session_blob, session_reflections
    )
```

---

## 8. RAG System (Standalone)

### 8.1 Independence

**Critical**: The RAG system is **completely independent** from the always-in-context system.

```python
# Configuration
RAG_ENABLED = True  # Can be toggled independently
ALWAYS_IN_CONTEXT_ENABLED = True  # Can be toggled independently

# Both can be on, both can be off, or any combination
```

### 8.2 RAG Implementation Options

The RAG retrieval can be implemented in two ways (TBD):

**Option A: Tool Call in Main Agent**
```python
# Agent has access to a memory search tool
tools = [
    {
        "name": "search_memory",
        "description": "Search long-term memory for facts about users, projects, etc.",
        "parameters": {
            "query": "string - what to search for",
            "scope": "filter for current agent_id + user_id scopes"
        }
    }
]
```

**Option B: Memory Subagent**
```python
# A background subagent decides when to query and what to retrieve
# Results are injected into main agent's context
async def memory_subagent_retrieve(
    conversation: List[Message],
    user_message: str
) -> List[Fact]:
    # Analyzes conversation, generates query, retrieves facts
    # Invisible to main agent
```

### 8.3 RAG Pipeline (from mem0)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG RETRIEVAL PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query (from tool call or subagent)                                         │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  1. EMBED QUERY                                                         ││
│  │     embedding = openai.embed(query, model="text-embedding-3-small")     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  2. HYBRID SEARCH WITH PRE-FILTERING (parallel)                         ││
│  │                                                                         ││
│  │     Scope filter applied INSIDE each search (not post-filter):          ││
│  │     • Include all agent-scoped facts (agent_id match)                   ││
│  │     • Include user-scoped facts for current user (user_id match)        ││
│  │                                                                         ││
│  │     ┌──────────────────────┐    ┌──────────────────────┐                ││
│  │     │  Semantic Search     │    │  Text Search         │                ││
│  │     │  ($vectorSearch)     │    │  ($search)           │                ││
│  │     │                      │    │                      │                ││
│  │     │  filter: {scope}     │    │  filter: {scope}     │                ││
│  │     │  + cosine similarity │    │  + keyword/fuzzy     │                ││
│  │     └──────────────────────┘    └──────────────────────┘                ││
│  │                │                         │                              ││
│  │                └────────────┬────────────┘                              ││
│  │                             │                                           ││
│  │                             ▼                                           ││
│  │     ┌──────────────────────────────────────────────────────┐            ││
│  │     │  Reciprocal Rank Fusion (RRF)                        │            ││
│  │     │  score(d) = Σ(1 / (k + rank)), k=60                  │            ││
│  │     └──────────────────────────────────────────────────────┘            ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  3. UPDATE ACCESS TRACKING                                              ││
│  │     access_count++                                                      ││
│  │     last_accessed_at = now()                                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                   │
│         ▼                                                                   │
│  Return top-K facts                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why pre-filtering is more efficient**: MongoDB Atlas Vector Search supports a `filter` parameter
that narrows the search space *before* computing similarity scores. This means:
- Fewer vectors to compare → faster query execution
- Lower computation cost
- Top-K results come from the relevant scope only (no risk of losing good matches to post-filtering)

### 8.4 RAG Search Implementation

```python
async def search_facts(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    scope_filter: List[str] = ["user", "agent"],
    match_count: int = 10,
    search_type: str = "hybrid"  # "hybrid", "semantic", or "text"
) -> List[Fact]:
    """
    Search facts in vector store with PRE-FILTERING.

    Scope filter is applied INSIDE the vector/text search (not post-filter).
    This is more efficient: narrows search space before computing similarities.
    """

    # Build pre-filter for MongoDB Atlas (applied inside $vectorSearch/$search)
    pre_filter = _build_scope_filter(agent_id, user_id, scope_filter)

    if search_type == "hybrid":
        semantic_results, text_results = await asyncio.gather(
            _semantic_search(query, pre_filter, match_count * 2),
            _text_search(query, pre_filter, match_count * 2)
        )
        results = _reciprocal_rank_fusion([semantic_results, text_results], k=60, limit=match_count)
    elif search_type == "semantic":
        results = await _semantic_search(query, pre_filter, match_count)
    else:
        results = await _text_search(query, pre_filter, match_count)

    # Update access tracking
    await _update_access_tracking([r["_id"] for r in results])

    return results


def _build_scope_filter(
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str]
) -> dict:
    """Build MongoDB filter for pre-filtering in vector/text search."""
    conditions = []
    if "agent" in scope_filter:
        conditions.append({"scope": "agent", "agent_id": agent_id})
    if "user" in scope_filter and user_id:
        conditions.append({"scope": "user", "user_id": user_id})
    return {"$or": conditions} if conditions else {}


async def _semantic_search(query: str, pre_filter: dict, limit: int) -> List[dict]:
    """Vector search with pre-filtering (filter applied BEFORE similarity computation)."""
    embedding = await openai.embed(query, model="text-embedding-3-small")

    pipeline = [
        {
            "$vectorSearch": {
                "index": "fact_vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": limit * 10,  # Over-fetch for better recall
                "limit": limit,
                "filter": pre_filter  # <-- PRE-FILTER: narrows search space first
            }
        },
        {"$project": {"embedding": 0}}  # Exclude large embedding from results
    ]
    return await db.memory_facts.aggregate(pipeline).to_list()


async def _text_search(query: str, pre_filter: dict, limit: int) -> List[dict]:
    """Text search with pre-filtering."""
    pipeline = [
        {
            "$search": {
                "index": "fact_text_index",
                "compound": {
                    "must": [{"text": {"query": query, "path": "content"}}],
                    "filter": [pre_filter]  # <-- PRE-FILTER: applied before scoring
                }
            }
        },
        {"$limit": limit}
    ]
    return await db.memory_facts.aggregate(pipeline).to_list()


def _reciprocal_rank_fusion(
    result_lists: List[List[dict]],
    k: int = 60,
    limit: int = 10
) -> List[dict]:
    """
    Merge ranked lists using RRF.
    Formula: RRF_score(d) = Σ(1 / (k + rank))
    """
    rrf_scores = {}
    doc_map = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = str(doc["_id"])
            rrf_score = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_ids[:limit]]
```

---

## 9. Data Models

### 9.1 Fact Model

```python
@Collection("memory_facts")
class Fact(Document):
    """
    Atomic factual memory stored in vector DB.
    Retrieved via RAG when relevant.

    NOTE: Facts do NOT have session scope (see Section 4.2).
    Session-level context is handled entirely by session reflections.
    """

    # Content
    content: str                           # The fact text
    hash: str                              # MD5 hash for deduplication

    # Embedding
    embedding: List[float]                 # 1536-dim vector
    embedding_model: str = "text-embedding-3-small"

    # Scope (NO session scope - only user/agent)
    scope: List[Literal["user", "agent"]]  # At least one required
    agent_id: ObjectId
    user_id: Optional[ObjectId] = None     # Required if "user" in scope

    # Temporal
    formed_at: datetime                    # When extracted from conversation
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Source (provenance - which session created this fact)
    source_session_id: ObjectId
    source_message_ids: List[ObjectId] = []

    # Access tracking
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None


# Indexes
FACT_INDEXES = [
    # Vector search index (created in Atlas UI)
    {
        "name": "fact_vector_index",
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"},
            {"type": "filter", "path": "agent_id"},
            {"type": "filter", "path": "user_id"},
            {"type": "filter", "path": "scope"}
        ]
    },
    # Text search index
    {
        "name": "fact_text_index",
        "mappings": {
            "fields": {"content": {"type": "string", "analyzer": "lucene.standard"}}
        }
    },
    # Deduplication index
    {"hash": 1, "agent_id": 1, "unique": True}
]
```

### 9.2 Reflection Model

```python
@Collection("memory_reflections")
class Reflection(Document):
    """
    Interpreted memory that evolves agent persona.
    Buffered until consolidated.
    """

    # Content
    content: str

    # Scope
    scope: List[Literal["session", "user", "agent"]]
    agent_id: ObjectId
    user_id: Optional[ObjectId] = None
    session_id: Optional[ObjectId] = None

    # Temporal
    formed_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Source
    source_session_id: ObjectId
    source_message_ids: List[ObjectId] = []

    # Consolidation state
    absorbed: bool = False                 # True after consolidated
    absorbed_at: Optional[datetime] = None
    consolidated_into: Optional[ObjectId] = None  # FK to ConsolidatedMemory
```

### 9.3 Consolidated Memory Model

```python
@Collection("memory_consolidated")
class ConsolidatedMemory(Document):
    """
    Merged reflection blob for a specific scope.
    Always in context.
    """

    # Scope key (unique per combination)
    scope_type: Literal["agent", "user", "session"]
    agent_id: ObjectId
    user_id: Optional[ObjectId] = None     # For user-level consolidation
    session_id: Optional[ObjectId] = None  # For session-level consolidation

    # Content
    consolidated_content: str              # The merged blob
    word_limit: int                        # Max words for this blob

    # Buffer tracking
    unabsorbed_ids: List[ObjectId] = []    # Reflections awaiting consolidation

    # Temporal
    last_consolidated_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Unique index on scope combination
CONSOLIDATED_INDEXES = [
    {"agent_id": 1, "user_id": 1, "session_id": 1, "scope_type": 1, "unique": True}
]
```

---

## 10. Implementation Plan

The implementation is split into two major phases:
- **Phase 1 (1.1-1.4)**: Reflections, Consolidation & Always-in-Context Memory
- **Phase 2 (2.1-2.4)**: Facts & RAG System

This ordering prioritizes the always-in-context memory system first, allowing us to validate reflections and consolidation before adding the complexity of facts, embeddings, and RAG retrieval.

---

### Phase 1: Reflections, Consolidation & Always-in-Context Memory

#### Phase 1.1: Core Models & Reflection Extraction

**Goal**: Create core models and reflection extraction with memory context awareness.

**Files**:
```
eve/agent/memory2/
├── __init__.py
├── models.py              # Reflection, ConsolidatedMemory (Fact model stubbed)
├── constants.py           # Thresholds, word limits
├── reflection_extraction.py   # LLM call for reflections (needs memory context)
└── reflection_storage.py      # Store reflections in buffer
```

**Deliverables**:
- Reflection model
- ConsolidatedMemory model
- Reflection extraction prompt (includes consolidated blob + recent reflections)
- Hierarchical extraction: agent → user → session
- Buffer management (unabsorbed reflections)

#### Phase 1.2: Consolidation

**Goal**: Implement consolidation when buffer threshold exceeded.

**Files**:
```
├── consolidation.py       # Consolidation LLM call
└── consolidated_storage.py  # ConsolidatedMemory management
```

**Deliverables**:
- Consolidation prompt and LLM call
- Buffer threshold checks (agent: 10, user: 5, session: 8)
- Word limit enforcement (agent: 1000, user: 400, session: 300)
- Error-safe consolidation (never lose reflections)
- Mark reflections as absorbed after consolidation

#### Phase 1.3: Always-in-Context Assembly

**Goal**: Assemble memory context for prompt injection.

**Files**:
```
├── context_assembly.py    # assemble_always_in_context_memory()
```

**Deliverables**:
- XML context builder
- Agent/user/session blob assembly
- Recent unabsorbed reflections inclusion
- Cache-friendly design (can cache in session object)

#### Phase 1.4: Reflection Integration & Testing

**Goal**: Wire reflections into the agent system and validate end-to-end.

**Files**:
```
├── formation.py           # maybe_form_memories() - reflection extraction orchestration
├── service.py             # High-level MemoryService facade (reflection methods)
```

**Deliverables**:
- Memory formation trigger (async, after agent response)
- Context injection into agent prompts
- End-to-end testing of reflection → buffer → consolidation → context flow
- Validation that consolidation preserves important information

---

### Phase 2: Facts & RAG System

#### Phase 2.1: Fact Extraction & Storage

**Goal**: Add fact extraction and vector storage (no RAG retrieval yet).

**Files**:
```
├── models.py              # Add Fact model (update existing)
├── fact_extraction.py     # LLM call for facts (no memory context needed)
└── fact_storage.py        # Store facts with embeddings
```

**Deliverables**:
- Fact model with embedding support
- Fact extraction prompt and LLM call (fast/cheap model)
- Embedding generation (text-embedding-3-small)
- Basic storage (hash check for exact duplicates)
- Scope assignment: user and/or agent (no session scope)

#### Phase 2.2: Facts Management (Deduplication & Conflict Resolution)

**Goal**: Add semantic deduplication and conflict resolution for facts (mem0-inspired).

**Files**:
```
├── fact_management.py     # process_extracted_facts(), similarity search, LLM decision
```

**Deliverables**:
- Similarity search for candidate retrieval (threshold 0.7)
- Memory Update Decision prompt and LLM call
- ADD/UPDATE/DELETE/NONE operation execution
- Version tracking for updated facts
- Optimization: skip LLM call when no similar facts found

#### Phase 2.3: RAG Retrieval

**Goal**: Implement RAG search pipeline (standalone, toggle-able).

**Files**:
```
├── rag.py                 # search_facts(), hybrid search, RRF
└── rag_tool.py            # Tool definition for agent (Option A)
```

**Deliverables**:
- Semantic search ($vectorSearch with pre-filtering)
- Text search (Atlas Search)
- Reciprocal Rank Fusion (RRF)
- Access tracking updates
- Tool call interface for agent

#### Phase 2.4: Full Integration & Migration

**Goal**: Wire facts into the memory formation pipeline, complete integration.

**Files**:
```
├── formation.py           # Update: orchestrate facts → reflections sequentially
└── migration.py           # One-time migration script
```

**Deliverables**:
- Sequential extraction: facts first, then reflections (with newly formed facts passed)
- Pass newly_formed_facts to reflection extraction (deduplication)
- RAG toggle (independent of always-in-context)
- Migration from old memory system
- Full end-to-end validation

---

## 11. MongoDB Atlas Setup

### 11.1 Collections

| Collection | Purpose |
|------------|---------|
| `memory_facts` | Facts with embeddings (RAG) |
| `memory_reflections` | Reflection buffer |
| `memory_consolidated` | Consolidated blobs |

### 11.2 Vector Search Index

Create in Atlas UI for `memory_facts`:

```json
{
  "name": "fact_vector_index",
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {"type": "filter", "path": "agent_id"},
    {"type": "filter", "path": "user_id"},
    {"type": "filter", "path": "scope"}
  ]
}
```

Note: No `session_id` filter needed - facts don't have session scope.

### 11.3 Text Search Index

Create in Atlas UI for `memory_facts`:

```json
{
  "name": "fact_text_index",
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

### 11.4 Cost Estimates

| Item | Cost |
|------|------|
| Embeddings | ~$0.02 per 1M tokens (~$0.001 per 1000 facts) |
| MongoDB Atlas | M0 (free) supports vector search |
| LLM calls (facts) | gemini-3-flash-preview (configurable) |
| LLM calls (reflections) | gemini-3-flash-preview (configurable) |

---

## Summary of Key Decisions

1. **Two memory types**: Facts (RAG) and Reflections (always-in-context)
2. **Sequential extraction**: Facts extracted first, then reflections (with newly formed facts passed to avoid redundancy)
3. **Facts have no session scope**: Only `user` and/or `agent` scope. Session context is handled entirely by session reflections.
4. **Hierarchical reflection extraction**: Agent → User → Session (each level only captures what's not in higher levels)
5. **All three reflection scopes consolidate**: Agent, user, AND session reflections all have buffer → consolidation flow
6. **RAG is standalone**: Can be enabled/disabled independently
7. **mem0-inspired RAG**: Vector storage, hybrid search, RRF fusion
8. **Facts management via LLM decision** (mem0-inspired): After extraction, an LLM compares new facts against semantically similar existing facts and decides ADD/UPDATE/DELETE/NONE - handles duplicates, contradictions, and fact evolution