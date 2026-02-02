# Eden Memory System v2

*Last updated: 2026-01-27*

---

## Overview

The Memory System v2 provides long-term memory capabilities for Eden agents. It consists of two **independent** subsystems that work together to give agents both persistent context awareness and searchable knowledge retrieval:

| Subsystem | Purpose | Storage | Retrieval Method |
|-----------|---------|---------|------------------|
| **Always-in-Context (Reflections)** | Agent persona, user preferences, session state | MongoDB | Automatically injected into every prompt |
| **Facts (FIFO + RAG)** | Searchable factual data | MongoDB with vector embeddings | Recent facts in context via FIFO + explicit search via `search_facts` tool |

### Key Design Principle

The system separates **behavioral context** (reflections - what the agent should know and how it should behave) from **retrievable knowledge** (facts - specific data points that can be searched when needed).

---

## Core Concepts

| Term | Definition |
|------|------------|
| **Fact** | Atomic, objective statement stored for retrieval. Stands alone without context. Searchable via hybrid RAG. Scopes: `user`, `agent` |
| **Reflection** | Interpreted memory that evolves the agent's behavior. Always in context. Scopes: `session`, `user`, `agent` |
| **Consolidation** | LLM-driven process of merging buffered reflections into a condensed blob |
| **Scope** | Where memory applies: `session` (this conversation), `user` (this user), `agent` (all users) |
| **FIFO** | Recent facts automatically included in context (limited by age and count) |
| **RAG** | Retrieval-Augmented Generation - semantic + keyword search for facts |

---

## Memory Scopes

### Scope Definitions

| Scope | Visibility | Facts | Reflections |
|-------|------------|-------|-------------|
| `session` | Current conversation only | ❌ No | ✅ Always active |
| `user` | All sessions with this user | ✅ Yes | ✅ If enabled |
| `agent` | All conversations, all users | ✅ Yes | ✅ If enabled |

### Important Rules

1. **Session reflections are ALWAYS formed**, regardless of user/agent toggles. This ensures every session maintains a rolling summary of context.

2. **Facts have no session scope** - session-level context is handled entirely by session reflections.

3. **Multi-user sessions (group chats)**: When `len(session.users) > 1`, user-scoped memory is **automatically disabled** to prevent memory leakage between users. Only session and agent scoped memories are active.

---

## Architecture

### High-Level Flow

```
                        ┌───────────────────────────────────┐
                        │         AGENT PROMPT              │
                        │                                   │
                        │  ┌────────────────────────────┐   │
                        │  │ <MemoryContext>            │   │
                        │  │   <AgentMemory>...</>      │   │
                        │  │   <UserMemory>...</>       │   │  ← Always-in-Context
                        │  │   <SessionMemory>...</>    │   │    (Reflections + FIFO Facts)
                        │  │   <Facts>...</>            │   │
                        │  │ </MemoryContext>           │   │
                        │  └────────────────────────────┘   │
                        │                                   │
                        │ + Agent can call search_facts     │  ← RAG Retrieval (on-demand)
                        │   tool to query long-term memory  │
                        └───────────────────────────────────┘
```

### Memory Formation Pipeline

Memory formation runs asynchronously after agent responses:

```
Conversation Messages
        │
        ▼
┌─────────────────────────────────────┐
│  LLM CALL 1: Fact Extraction        │  (gemini-3-flash-preview)
│  - Fast model, no memory context    │
│  - Extracts: user + agent scoped    │
│  - Output: atomic factual statements│
└─────────────────────────────────────┘
        │
        │  extracted_facts
        ▼
┌─────────────────────────────────────┐
│  DEDUPLICATION (if enabled)         │  (parallel vector search + LLM)
│  - Batch embed all new facts        │
│  - Vector search for similar (||)   │  ← Parallelized
│  - LLM decides: ADD/UPDATE/DELETE   │  ← Single batched call
│  - Execute decisions                │
└─────────────────────────────────────┘
        │
        │  deduplicated_facts (passed to next call + stored)
        ▼
┌─────────────────────────────────────┐
│  LLM CALL 2: Reflection Extraction  │  (gemini-3-flash/pro based on tier)
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

### Fact Deduplication

When `FACTS_DEDUP_ENABLED=True` (default), new facts go through a deduplication pipeline:

1. **Batch Embedding**: All new facts are embedded in a single API call
2. **Parallel Vector Search**: Each fact's embedding is searched against existing facts concurrently using `asyncio.gather`
3. **Batched LLM Decision**: Facts with similar matches are sent to an LLM in a single call to decide:
   - `ADD`: Store as new fact (no similar facts found, or genuinely new info)
   - `UPDATE`: Merge with existing fact (enhances/corrects existing)
   - `DELETE`: Remove contradicting fact and add new (preference reversal)
   - `NONE`: Skip (semantic duplicate already exists)
4. **Execute**: Decisions are applied to the database

This prevents duplicate facts ("John's email is john@example.com" stored multiple times) and handles contradictions ("John likes pizza" → "John dislikes pizza").

### Context Assembly

Memory context is injected into every agent prompt as XML:

```xml
<MemoryContext>
  <AgentMemory>
    <Consolidated>VERSION: 3
    [CURRENT PROJECTS]
    - Mars Festival: Phase 2, budget $50k
    ...</Consolidated>
    <RecentReflections>
    - User mentioned new deadline for Phase 3
    ...</RecentReflections>
  </AgentMemory>
  <UserMemory>
    <Consolidated>VERSION: 2
    [PREFERENCES]
    - Prefers concise responses
    ...</Consolidated>
  </UserMemory>
  <SessionMemory>
    <Consolidated>VERSION: 1
    [CURRENT GOAL]
    - Debugging the login issue
    ...</Consolidated>
    <RecentReflections>
    - Tried approach #3, still failing
    ...</RecentReflections>
  </SessionMemory>
  <Facts>
    - [agent] Mars Festival budget is $50,000 (3d ago)
    - [user] Alice's email is alice@example.com (1h ago)
    - [agent] API endpoint is api.eden.art/v2 (5d ago)
  </Facts>
</MemoryContext>
```

**Caching**: Memory context is cached on the session with a 5-minute TTL and persisted to the database.

---

## Facts System

### Dual Retrieval Strategy

Facts are accessible through two complementary mechanisms:

#### 1. FIFO (Always-in-Context)

Recent facts are automatically included in the agent's always-on context (do not need to be retrieved):

- **Age Filter**: Only facts within `FACTS_FIFO_MAX_AGE_HOURS` (default: 7 days)
- **Count Limit**: Maximum `FACTS_FIFO_LIMIT` facts (default: 40)
- **Ordering**: Most recent first
- **Format**: Includes scope indicator and temporal age suffix

```
<Facts>
- [agent] Project deadline is January 30th (2d ago)
- [user] Alice prefers Python over JavaScript (5h ago)
</Facts>
```

#### 2. RAG Tool (`search_facts`)

Agents with memory enabled automatically receive the `search_facts` tool for explicit memory searches that lets them access all FACTS ever created:

**Tool Definition:**
```yaml
name: search_facts
description: Search long-term memory using query(s)
parameters:
  query:
    type: array
    items: string
    max_length: 3
    description: List of search queries (run in parallel)
  top_k:
    type: integer
    default: 10
    description: Number of facts to retrieve per query
```

**Search Method**: Hybrid retrieval combining:
- **Semantic Search**: MongoDB Atlas Vector Search using OpenAI embeddings
- **Text Search**: MongoDB Atlas Search with fuzzy matching
- **Result Fusion**: Reciprocal Rank Fusion (RRF) to merge and re-rank results

**When the Agent Should Use It**:
- When the answer to a specific question is not in current memory context
- Recalling specific information (emails, URLs, specifications)
- Finding stored data from conversations days/weeks ago

### Fact Extraction Criteria

Facts must be:
1. **Searchable**: Answers a specific question (Who, What, Where, When)
2. **Specific**: Contains names, numbers, dates, or specific entities
3. **Enduring**: Information likely to remain true for at least a month
4. **Cold Storage Appropriate**: OK to "forget" until specifically searched for

**DO NOT extract as facts:**
- Preferences & behavior ("Alice likes concise responses") → These go to reflections
- Current context ("We are debugging X") → Session reflection
- Opinions ("Bob thinks we should use React") → Reflection if relevant
- Ephemeral state ("There's a bug in production right now") → Session reflection

---

## Reflections System

### Extraction Hierarchy

Reflections are extracted in a hierarchical manner to avoid redundancy:

1. **Agent Reflections** (broadest scope - extracted first)
   - Projects & milestones
   - Learnings about users/community
   - World state changes

2. **User Reflections** (only what's NOT captured at agent level)
   - Behavioral rules and preferences
   - Skills, interests, goals
   - Personal project tracking

3. **Session Reflections** (only what's NOT captured above)
   - High-level goals for this conversation
   - Assets to pin (URLs, images)
   - Corrections and preferences for this session

### Consolidation

When the buffer of unabsorbed reflections exceeds a threshold, an LLM merges them into a condensed blob.

**Thresholds** (production):
| Scope | Threshold | Word Limit |
|-------|-----------|------------|
| Agent | 10 reflections | 1200 words |
| User | 4 reflections | 300 words |
| Session | 4 reflections | 200 words |

**Consolidation runs in parallel** across scopes since they are independent.

**LLM Model Selection**:
- Free users: `gemini-3-flash-preview`
- Premium users (tier ≥ 1) or preview flag: `gemini-3-pro-preview`

---

## Configuration

### Feature Toggles (constants.py)

```python
# Feature Toggles
ALWAYS_IN_CONTEXT_ENABLED = True  # Master toggle for memory system

# Facts FIFO Configuration
FACTS_FIFO_ENABLED = True          # Enable FIFO facts in context
FACTS_FIFO_LIMIT = 40              # Max recent facts to include
FACTS_FIFO_MAX_AGE_HOURS = 24 * 7  # Only facts from last 7 days

# Facts Deduplication
FACTS_DEDUP_ENABLED = True         # Enable semantic deduplication pipeline
FACTS_DEDUP_SIMILARITY_LIMIT = 5   # Max similar facts to retrieve per new fact
SIMILARITY_THRESHOLD = 0.7         # Min cosine similarity for dedup match

# Models
MEMORY_LLM_MODEL_FAST = "gemini-3-flash-preview"  # Fact extraction + dedup decisions
# Reflection extraction/consolidation: tier-based (flash for free, pro for premium)

# Formation Triggers
MEMORY_FORMATION_MSG_INTERVAL = 45      # Messages between formations
MEMORY_FORMATION_TOKEN_INTERVAL = 1500  # Weighted tokens trigger
NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = 4

# RAG Thresholds
RAG_SEMANTIC_SCORE_THRESHOLD = 0.65  # Min vector similarity
RAG_TEXT_SCORE_THRESHOLD = 1.5       # Min text search score
RAG_RRF_SCORE_THRESHOLD = 0.015      # Min RRF fusion score
```

### Agent-Level Toggles

On the `Agent` model:
- `agent.user_memory_enabled`: Enable user-scoped memories
- `agent.agent_memory_enabled`: Enable agent-scoped memories

When either is enabled, the agent automatically receives:
1. Memory context injection into all prompts
2. The `search_facts` tool for RAG retrieval

---

## Data Models

### MongoDB Collections

| Collection | Purpose |
|------------|---------|
| `memory2_facts` | Facts with embeddings |
| `memory2_reflections` | Reflection buffer |
| `memory2_consolidated` | Consolidated blobs |

### Fact Schema

```python
content: str                        # The fact text
hash: str                           # MD5 for deduplication
embedding: List[float]              # 1536-dim vector (text-embedding-3-small)
scope: "user" | "agent"             # Single scope value
agent_id: ObjectId
user_id: Optional[ObjectId]         # If scope == "user"
session_id: Optional[ObjectId]      # Source session
formed_at: datetime
access_count: int                   # RAG retrieval tracking
version: int                        # Incremented on UPDATE
```

### Reflection Schema

```python
content: str
scope: "session" | "user" | "agent"
agent_id: ObjectId
user_id: Optional[ObjectId]         # If scope == "user"
session_id: ObjectId                # Always populated (source session)
absorbed: bool                      # True after consolidation
formed_at: datetime
```

### ConsolidatedMemory Schema

```python
scope: "agent" | "user" | "session"
agent_id: ObjectId
user_id: Optional[ObjectId]         # For user scope
session_id: Optional[ObjectId]      # For session scope
consolidated_content: str           # The merged blob
word_limit: int                     # Max words for this blob
unabsorbed_ids: List[ObjectId]      # Pending reflection IDs
last_consolidated_at: datetime
```

---

## File Structure

```
eve/agent/memory2/
├── __init__.py              # Public exports
├── constants.py             # Config, thresholds, prompts, Memory2Config
├── models.py                # Fact, Reflection, ConsolidatedMemory
├── service.py               # High-level MemoryService facade
├── backend.py               # Memory2Backend for integration
│
├── formation.py             # Memory formation orchestration
├── context_assembly.py      # Always-in-context assembly
├── consolidation.py         # Reflection consolidation
│
├── fact_extraction.py       # LLM call for fact extraction
├── fact_storage.py          # Embedding generation, storage
├── fact_management.py       # Deduplication pipeline
│
├── reflection_extraction.py # LLM call for reflection extraction
├── reflection_storage.py    # Buffer management
│
├── rag.py                   # Vector/text search, RRF fusion
├── rag_tool.py              # DEPRECATED - tool now in agent stack
├── utils.py                 # Helper utilities
│
├── memory2_cold_sessions_processor.py  # Background processing
└── scripts/                 # Migration and maintenance scripts
```

### Related Files

```
eve/tools/retrieval/search_facts/
├── api.yaml                 # Tool definition
└── handler.py               # search_facts tool handler
```

---

## Usage

### Basic Integration

```python
from eve.agent.memory2 import MemoryService

# Create service
service = MemoryService(agent_id)

# Get memory context for prompts (cached, auto-refreshed)
context = await service.get_memory_context(session, user_id)

# Form memories from conversation (called after agent response)
await service.maybe_form_memories(session, messages, user_id)
```

### Using the Backend (Recommended)

```python
from eve.agent.memory2.backend import memory2_backend

# Assemble memory context
memory_xml = await memory2_backend.assemble_memory_context(
    session=session,
    agent=agent,
    user=user,
    force_refresh=False,
)

# Conditionally form memories (respects thresholds)
formed = await memory2_backend.maybe_form_memories(
    agent_id=agent.id,
    session=session,
    agent=agent,
)
```

### Session Lifecycle

```python
# On session start
await service.on_session_start(session, user_id)

# On session end (forces final formation + consolidation)
await service.on_session_end(session, messages, user_id)

# For cold session processing (background job)
await service.process_cold_session(session, user_id)
```

---

## Key Design Decisions

1. **Two independent systems**: Reflections (always-in-context) and Facts (FIFO + RAG) can function independently

2. **Sequential extraction**: Facts first (fast model), then reflections with fact awareness (avoids redundancy)

3. **No session-scoped facts**: Session context is handled entirely by session reflections

4. **Session always active**: Session reflections are always formed regardless of memory toggles

5. **Parallel consolidation**: Different scopes consolidate independently and concurrently

6. **Multi-user protection**: User memory automatically disabled in group chats

7. **FIFO + RAG hybrid**: Recent facts always in context, older facts searchable via tool

8. **Tier-based model selection**: Premium users get higher quality models for reflection extraction

9. **Automatic tool provisioning**: Agents with memory enabled automatically receive `search_facts` tool

---

## Prompt Engineering Notes

### Fact Extraction Prompt

The fact extraction prompt emphasizes:
- Facts are for **retrieval only** (not always in context)
- Must be answers to specific questions
- Must be enduring (true for 30+ days)
- Excludes preferences, opinions, ephemeral state
- Maximum 30 words per fact

### Reflection Extraction Prompt

The reflection extraction prompt:
- Shows current memory state to avoid redundancy
- Includes newly formed facts to prevent overlap
- Uses hierarchical extraction (agent → user → session)
- Emphasizes behavioral relevance
- Maximum 35 words per reflection

### Consolidation Prompt

The consolidation prompt:
- Increments version numbers
- Preserves section structure
- Resolves conflicts (new info wins)
- Performs garbage collection on outdated info
- Respects word limits
- Never invents information

---

## Monitoring & Debugging

### Memory Stats

```python
stats = service.get_memory_stats(agent_id, user_id, session_id)
# Returns word counts, reflection counts, timestamps per scope
```

### Consolidation Status

```python
status = service.get_consolidation_status(agent_id, user_id, session_id)
# Returns buffer sizes, thresholds, whether consolidation needed
```

### Debug Mode for search_facts

Set `debug: true` in tool call to get detailed timing and scoring info for each search stage.

---

## Migration Notes

If upgrading from memory v1:
- Facts collection changed from `memories` to `memory2_facts`
- Reflections are a new concept (no direct v1 equivalent)
- Consolidated memories replace the old simple memory blobs
- RAG is now accessed via `search_facts` tool, not proactive injection
