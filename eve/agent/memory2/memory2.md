# Eden Memory System v2

*Last updated: 2026-01-21*

---

## Overview

The memory system consists of two **independent** subsystems that can be enabled/disabled separately:

| System | Purpose | Storage | Retrieval |
|--------|---------|---------|-----------|
| **Always-in-Context (Reflections)** | Agent persona, user preferences, session state | MongoDB | Injected into every prompt |
| **RAG (Facts)** | Searchable factual data | MongoDB with vector embeddings | Via tool call or FIFO (temporary) |

---

## Core Concepts

| Term | Definition |
|------|------------|
| **Fact** | Atomic, objective statement stored for retrieval. Stands alone without context. Scopes: `user`, `agent` |
| **Reflection** | Interpreted memory that evolves the agent's behavior. Always in context. Scopes: `session`, `user`, `agent` |
| **Consolidation** | LLM-driven process of merging buffered reflections into a condensed blob |
| **Scope** | Where memory applies: `session` (this conversation), `user` (this user), `agent` (all users) |

---

## Memory Scopes

| Scope | Visibility | Facts | Reflections |
|-------|------------|-------|-------------|
| `session` | Current conversation only | ❌ No | ✅ Always active |
| `user` | All sessions with this user | ✅ Yes | ✅ If enabled |
| `agent` | All conversations, all users | ✅ Yes | ✅ If enabled |

**Important**: Session reflections are **always formed**, regardless of user/agent toggles. This ensures every session maintains a rolling summary of context.

### Multi-User Sessions (Group Chats)

When `len(session.users) > 1`, user-scoped memory is **automatically disabled** to prevent memory leakage between users. Only session and agent scoped memories are active.

---

## Architecture

### Extraction Pipeline

Memory formation runs asynchronously after agent responses:

```
Conversation Messages
        │
        ▼
┌─────────────────────────────────────┐
│  LLM CALL 1: Fact Extraction        │  (gemini-3-flash-preview)
│  - No memory context needed         │
│  - Scopes: user, agent              │
└─────────────────────────────────────┘
        │
        │  extracted_facts (passed to next call)
        ▼
┌─────────────────────────────────────┐
│  LLM CALL 2: Reflection Extraction  │  (gemini-3-flash-preview)
│  - Includes memory context          │
│  - Includes newly formed facts      │
│  - Hierarchical: agent → user → session
└─────────────────────────────────────┘
        │
        ├──────────────────────┬────────────────────┐
        ▼                      ▼                    ▼
   Vector Store          Reflection Buffer     Consolidation
   (facts w/ embeddings)  (unabsorbed)         (if threshold met)
```

### Context Assembly

Memory context is injected into every agent prompt as XML:

```xml
<MemoryContext>
  <AgentMemory>
    <Consolidated>...</Consolidated>
    <RecentReflections>...</RecentReflections>
  </AgentMemory>
  <UserMemory>...</UserMemory>
  <SessionMemory>...</SessionMemory>
  <Facts>...</Facts>  <!-- FIFO mode only -->
</MemoryContext>
```

Memory context is cached on the session with a 5-minute TTL and persisted to the database.

---

## Consolidation

When the buffer of unabsorbed reflections exceeds a threshold, an LLM merges them into a condensed blob.

**Thresholds** (production):
- Agent: 8 reflections
- User: 4 reflections
- Session: 4 reflections

**Word Limits**:
- Agent: 1000 words
- User: 300 words
- Session: 200 words

Consolidations for different scopes run in **parallel** since they are independent.

---

## Facts Management

Facts are stored with embeddings for semantic search. The system supports:

1. **Hash-based deduplication** - Exact duplicate prevention
2. **LLM-based deduplication** (when RAG enabled) - Semantic comparison deciding ADD/UPDATE/DELETE/NONE

### Current State: FIFO Mode

RAG is not yet fully enabled. Facts are currently:
- Extracted and embedded (same as full RAG)
- Stored directly (skipping deduplication LLM)
- Retrieved via FIFO query (50 most recent facts)

When RAG is enabled:
1. Set `RAG_ENABLED = True` in constants.py
2. Set `FACTS_FIFO_ENABLED = False`
3. Ensure MongoDB Atlas vector index is configured

---

## Configuration

Key settings in `constants.py`:

```python
# Models
MEMORY_LLM_MODEL_FAST = "gemini-3-flash-preview"
MEMORY_LLM_MODEL_SLOW = "gemini-3-flash-preview"

# Formation triggers
MEMORY_FORMATION_MSG_INTERVAL = 45   # Messages between formations
MEMORY_FORMATION_TOKEN_INTERVAL = 1500  # Weighted tokens trigger
NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = 4

# Feature toggles
RAG_ENABLED = False
FACTS_FIFO_ENABLED = True
ALWAYS_IN_CONTEXT_ENABLED = True
```

Agent-level toggles:
- `agent.user_memory_enabled` - Enable user-scoped memories
- `agent.agent_memory_enabled` - Enable agent-scoped memories

---

## Data Models

### Collections

| Collection | Purpose |
|------------|---------|
| `memory2_facts` | Facts with embeddings |
| `memory2_reflections` | Reflection buffer |
| `memory2_consolidated` | Consolidated blobs |

### Fact

```python
content: str
scope: List["user" | "agent"]
agent_id: ObjectId
user_id: Optional[ObjectId]  # if "user" in scope
embedding: List[float]       # 1536-dim vector
hash: str                    # MD5 for dedup
version: int                 # Incremented on UPDATE
access_count: int            # RAG retrieval tracking
```

### Reflection

```python
content: str
scope: "session" | "user" | "agent"
agent_id: ObjectId
user_id: Optional[ObjectId]
session_id: ObjectId
absorbed: bool               # True after consolidation
```

### ConsolidatedMemory

```python
scope_type: "agent" | "user" | "session"
agent_id: ObjectId
user_id: Optional[ObjectId]
session_id: Optional[ObjectId]
consolidated_content: str
word_limit: int
unabsorbed_ids: List[ObjectId]
```

---

## File Structure

```
eve/agent/memory2/
├── __init__.py              # Public exports
├── constants.py             # Config, thresholds, prompts
├── models.py                # Fact, Reflection, ConsolidatedMemory
├── formation.py             # Memory formation orchestration
├── context_assembly.py      # Always-in-context assembly
├── consolidation.py         # Reflection consolidation
├── fact_extraction.py       # LLM call 1
├── fact_storage.py          # Embedding, storage
├── fact_management.py       # Deduplication pipeline
├── reflection_extraction.py # LLM call 2
├── reflection_storage.py    # Buffer management
├── rag.py                   # Vector search, RRF fusion
├── rag_tool.py              # Agent tool interface
├── service.py               # High-level MemoryService facade
└── memory2_cold_sessions_processor.py  # Background processing
```

---

## Usage

```python
from eve.agent.memory2 import MemoryService

# Create service
service = MemoryService(agent_id)

# Get memory context for prompts
context = await service.get_memory_context(session, user_id)

# Form memories from conversation (called after agent response)
await service.maybe_form_memories(session, messages, user_id)

# Search facts via RAG (when enabled)
facts = await service.search_facts("user preferences", user_id)
```

---

## Key Design Decisions

1. **Two independent systems**: RAG and always-in-context can be enabled separately
2. **Sequential extraction**: Facts first, then reflections (with fact awareness for deduplication)
3. **No session-scoped facts**: Session context is handled entirely by session reflections
4. **Session always active**: Session reflections are always formed regardless of toggles
5. **Parallel consolidation**: Different scopes consolidate independently and in parallel
6. **Multi-user protection**: User memory automatically disabled in group chats
7. **FIFO as stepping stone**: Facts are embedded but retrieved via FIFO until RAG is fully enabled
