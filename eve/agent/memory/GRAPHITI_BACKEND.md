# Graphiti Memory Backend Overview

This note explains how the proof-of-concept Graphiti backend is wired into Eden’s
memory system, how it relates to the existing Mongo abstractions, and which
knobs control when memories are extracted and injected into prompts.

## Where memory is used inside the session loop

1. **Prompt assembly** (`eve/agent/session/session.py:build_system_message`)
   calls `memory_service.assemble_memory_context(...)` to build the XML block
   that is injected into the system prompt. This stage combines:
   - User directives (Mongo).
   - Collective shard blobs (Mongo).
   - The most recent episodes for the session, now sourced from Graphiti when
     the Graphiti backend is active.

2. **Online formation** happens through
   `memory_service.maybe_form_memories(...)`, which is scheduled from the
   conversation loop (see `Session._dispatch_messages` around line ~1560).
   - `should_form_memories` in `memory.py` enforces the message/token gates.
     The defaults (`MEMORY_FORMATION_MSG_INTERVAL`, `MEMORY_FORMATION_TOKEN_INTERVAL`,
     `NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES`) live in `memory_constants.py`.

3. **Cold session sweep** (`memory_cold_sessions_processor.py`) periodically
   calls `memory_service.form_memories(...)` for stalled sessions so that
   “stale” conversations still emit memories. This path now also respects the
   agent-specific backend selection described below.

Together these hooks mean the “input” to memory is either the live session
loop (message based) or the cold-session processor, and the “output” is the XML
block injected from `build_system_message`. Both entry points now go through
`MemoryService`, which decides whether to use the legacy Mongo backend or the
Graphiti backend on a per-agent basis.

## Mapping Mongo concepts to Graphiti

| Mongo concept                          | Purpose                                                    | Graphiti surface                                    |
|---------------------------------------|------------------------------------------------------------|-----------------------------------------------------|
| **Episode memories** (`SessionMemory`) | Recent session summaries used for prompt continuity        | Stored as Graphiti Episodic nodes (MENTIONS edges)  |
| **Directives** (`UserMemory`)          | Persistent user-specific rules/preferences                 | Remain in Mongo; surfaced in context alongside episodes |
| **Facts/Suggestions** (`AgentMemory`)  | Collective shard state and proposals                       | Remain in Mongo; Graphiti POC does not mutate shards |
| **Memory context XML**                 | Prompt-ready view of user/collective/episode data          | Episode section now hydrated via `Graphiti.retrieve_episodes` |

Only the episodic slice was written to Graphiti in the original POC. The latest
backend now mirrors *all* memory types inside Graphiti using explicit namespaces.

## Namespace Strategy

| Namespace                               | Contents                                                               |
|-----------------------------------------|------------------------------------------------------------------------|
| `session:<session_id>`                  | Episodic summaries extracted from the current conversation window      |
| `user:<agent_id>:<user_id>`             | User directives + metadata (per agent/user pair)                       |
| `shard:<agent_id>:<shard_id>`           | Collective shard facts, consolidated text, and recent suggestions      |

Every call to `graphiti.add_episode` includes a `group_id` set to one of the
namespaces above. The `episode_body` is a JSON payload of the form:

```json
{
  "memory_type": "directive",
  "content": "Always favor neon gradients.",
  "metadata": {
    "agent_id": "...",
    "user_id": "...",
    "session_id": "...",
    "message_ids": ["..."],
    "sequence": 0
  }
}
```

Retrieval uses the same namespace via `group_id`, so a single Falkor instance
can safely host multiple agents, users, and sessions without data leakage.

## End-to-end flow (Graphiti backend)

1. **Trigger**  
   `GraphitiMemoryBackend.maybe_form_memories` still defers to
   `should_form_memories`, so the existing “every ~25 messages / stale session”
   cadence is preserved.

2. **Extraction & storage**  
   `_extract_all_memories` (the same helper used by Mongo) returns episodes,
   directives, facts, and suggestions. `_persist_graphiti_memories` stores each
   item into its namespace as JSON episodes, while `_save_all_memories` keeps the
   Mongo collections in sync for UI/backfill compatibility.

3. **Prompt assembly**  
   `GraphitiMemoryBackend.assemble_memory_context` reads back the namespaced
   episodes directly from Graphiti to build the three sections of the memory XML.
   If a namespace is empty (e.g., an agent just opted in), the code falls back to
   the Mongo helpers so nothing regresses.

## Configuring which backend to use

- Agents can opt into Graphiti by setting
  `agent.agent_extras.experimental_memory_backend = "graphiti"`.
- `MemoryService` keeps the Mongo backend as the default but now supports
  registering named backends. At import time it registers a `"graphiti"`
  backend (when `graphiti_core` is available) via `GraphitiMemoryBackend`.
- Every call to `assemble_memory_context`, `maybe_form_memories`, and
  `form_memories` resolves the backend based on the agent’s extras (falling
  back to Mongo if the backend is missing or fails to initialize).
- Background tasks that only know an `agent_id` will lazily fetch the agent to
  honor the override.

This makes the entry/exit points explicit and lets us experiment with the
Graphiti backend on a single agent without disturbing others. Once Graphiti
data looks good for an agent we can gradually stop reading from Mongo for that
agent.

## Tuning triggers and stale-session behavior

- Message/token thresholds live in `memory_constants.py`:
  `MEMORY_FORMATION_MSG_INTERVAL`, `MEMORY_FORMATION_TOKEN_INTERVAL`, and
  `NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES`.
- The background sweep cadence is controlled via
  `memory_cold_sessions_processor.py` (`CLEANUP_COLD_SESSIONS_EVERY_MINUTES`)
  and the “consider cold after” window in `memory_constants.py`
  (`CONSIDER_COLD_AFTER_MINUTES`).
- Because `MemoryService` now routes per agent, changing these constants or
  swapping the backend can be safely tested on an isolated agent before being
  rolled out more broadly.

## Next steps

1. Gradually migrate UI + analytics paths to use Graphiti namespaces directly.
2. Add namespace-specific maintenance tasks (e.g., pruning or aggregating old
   directive episodes) now that the graph is the source of truth.
