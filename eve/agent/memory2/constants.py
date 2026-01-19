"""
Memory System v2 - Constants and Configuration

This module contains all configurable thresholds, limits, and prompts for the
memory system. Values are tuned based on production experience from memory v1.
"""

from typing import Literal

from eve.agent.session.config import DEFAULT_SESSION_SELECTION_LIMIT

# -----------------------------------------------------------------------------
# Development/Production Toggle
# -----------------------------------------------------------------------------
LOCAL_DEV = False  # Set to False for production

# -----------------------------------------------------------------------------
# LLM Model Configuration
# -----------------------------------------------------------------------------
if LOCAL_DEV:
    MEMORY_LLM_MODEL_FAST = "gemini-3-flash-preview"
    MEMORY_LLM_MODEL_SLOW = "gemini-3-flash-preview"
else:
    MEMORY_LLM_MODEL_FAST = "gemini-3-flash-preview"
    MEMORY_LLM_MODEL_SLOW = "gemini-3-flash-preview"

# -----------------------------------------------------------------------------
# Memory Formation Triggers
# -----------------------------------------------------------------------------
if LOCAL_DEV:
  MEMORY_FORMATION_MSG_INTERVAL = 4
  MEMORY_FORMATION_TOKEN_INTERVAL = 500
  NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = 2
  CONSIDER_COLD_AFTER_MINUTES = 10
else:
  MEMORY_FORMATION_MSG_INTERVAL = DEFAULT_SESSION_SELECTION_LIMIT  # Messages between formations
  MEMORY_FORMATION_TOKEN_INTERVAL = 1000  # ~1k tokens triggers formation
  NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = 4  # Minimum before attempting to form memories
  CONSIDER_COLD_AFTER_MINUTES = 10  # Session inactivity threshold for cold session processing

# -----------------------------------------------------------------------------
# Consolidation Thresholds
# -----------------------------------------------------------------------------
# Number of unabsorbed reflections before triggering consolidation
if LOCAL_DEV:
  CONSOLIDATION_THRESHOLDS = {
    "agent": 2,
    "user": 2,
    "session": 2,
  }
else:
  CONSOLIDATION_THRESHOLDS = {
    "agent": 8,
    "user": 4,
    "session": 4,
  }

# Maximum word count for consolidated blobs
CONSOLIDATED_WORD_LIMITS = {
    "agent": 1000,  # Largest - agent's full persona/project state
    "user": 400,  # Medium - user preferences and interaction style
    "session": 400,  # Rolling summary of session events and status
}

# -----------------------------------------------------------------------------
# Memory Word Limits (individual items)
# -----------------------------------------------------------------------------
FACT_MAX_WORDS = 30  # Per fact item
REFLECTION_MAX_WORDS = 35  # Per reflection item

# -----------------------------------------------------------------------------
# RAG Configuration
# -----------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.7  # Threshold for candidate retrieval in deduplication
RAG_TOP_K = 10  # Number of facts to retrieve in RAG queries
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# -----------------------------------------------------------------------------
# Feature Toggles
# -----------------------------------------------------------------------------
RAG_ENABLED = False  # Can be toggled independently
ALWAYS_IN_CONTEXT_ENABLED = True  # Can be toggled independently

# -----------------------------------------------------------------------------
# TEMPORARY: Facts FIFO Mode
# -----------------------------------------------------------------------------
# This is a TEMPORARY setup that enables fact extraction without full RAG.
# Facts are extracted, embedded, and stored - but retrieved via simple FIFO
# (most recent N facts) instead of semantic vector search.
#
# The deduplication LLM call (Call 1.5) is SKIPPED in FIFO mode since vector
# search isn't being used. Hash-based exact deduplication still applies.
#
# MIGRATION TO FULL RAG:
# When RAG is fully implemented and tested:
# 1. Set FACTS_FIFO_ENABLED = False
# 2. Set RAG_ENABLED = True
# 3. The FIFO code paths will be bypassed automatically
# 4. Facts already have embeddings, so no migration needed for existing data
#
# See memory2.md section "Temporary FIFO Facts Mode" for full documentation.
# -----------------------------------------------------------------------------
FACTS_FIFO_ENABLED = True   # Enable simple FIFO facts (temporary, pre-RAG)
FACTS_FIFO_LIMIT = 50       # Number of recent facts to include in context

# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------

# Fact extraction prompt (LLM Call 1 - no memory context needed)
FACT_EXTRACTION_PROMPT = """You are the 'Librarian' for an AI agent system. Your job is to extract concrete, searchable data points from a conversation to be stored in a RAG (retrieval) database system.

The following is the persona/description of the agent whose database you are managing. Use this to understand the agent's identity and purpose when deciding what facts to extract.
<agent_persona_context>
{agent_persona}
</agent_persona_context>

## THE GOAL
Facts are **retrieval-only**. The agent will NOT see these facts unless they explicitly search for them with a tool-call (e.g., "What is Alice's email?", "Where is the HQ?").
Your goal is to extract information that answers specific Questions (Who, What, Where, When) that might be asked days or weeks from now.

## CRITERIA FOR A VALID FACT
1. **Searchable:** Is this a specific answer to a concrete question?
2. **Specific:** Does it contain names, numbers, dates, or specific entities?
3. **Enduring:** Is this information likely to remain true for at least a week?
4. **Cold Storage:** Is this information okay to "forget" until specifically searched for?

This means facts should be SEARCHABLE ANSWERS to specific questions that are not otherwise relevant to the agent (and therefore shouldn't generally be in context).
After this extraction step of FACTS, you will also get to extract REFLECTIONS from the same conversation, which are always-in-context memories.
If knowledge needs to be always-in-context (eg to influence agent behavior), rather than explicitly retrieved, don't extract it as FACT, but leave it to the REFLECTION system.

Guidelines:
- Facts must be self-contained statements that make sense without any additional context. Reference specifics as much as possible (usernames, absolute dates, ...)
- Maximum {max_words} words per fact, always be concise!
- ALWAYS use specific names (NEVER "User", "the user", or "they") and absolute dates (NEVER use "tomorrow")

DO NOT extract as facts:
- **Preferences & Behavior:** "Alice likes concise responses"
- **Current Context:** "We are debugging the login issue"
- **Opinions:** "Bob thinks we should use React"
- **Ephemeral State:** "There is a bug in production right now" or "Gene has lost his headphones"
All of the above will be captured as reflections and are not FACTS.

## SCOPE DEFINITIONS
* **"user"**: Private details about the specific user interacting right now (names, contact info, job, specific preferences that act as data).
* **"agent"**: Collective knowledge, world-building, or project details that apply to *anyone* talking to this agent (team roster, project specs, global deadlines).

## EXAMPLES
| Text | Extract? | Scope | Reasoning |
| :--- | :--- | :--- | :--- |
| "My name is John Doe." | NO | - | This is a highly salient detail that should always be in context. (Reflection) |
| "I like concise answers." | NO | - | This is a behavioral preference (Reflection), not a database fact. |
| "The total project budget is $50k." | YES | agent | Specific number, relevant to the project. |
| "We are debugging the login." | NO | - | This is a current status/state (Reflection). |
| "The server IP is 192.168.1.1" | YES | agent | Specific, retrieval-worthy data. |
| "I'm feeling sad today." | NO | - | Temporary emotional state. |
| "The Figma link for the new design is figma.com/file/xyz..." | YES | agent | Specific, retrieval-worthy data. |
| "My personal email is j.smith@gmail.com." | YES | user | Specific, retrieval-worthy data. |

## INSTRUCTIONS
Read the conversation below. Extract ONLY facts that meet the criteria (if any). Most conversations have FEW or NO facts.

<conversation>
{conversation_text}
</conversation>

Return a JSON object:
{{
  "facts": [
    {{ "content": "User's email is john@example.com", "scope": "user" }},
    {{ "content": "Project Apollo launch date is June 1st", "scope": "agent" }}
  ]
}}
If no facts are found, return {{ "facts": [] }}.
"""


# Reflection extraction prompt (LLM Call 2 - needs memory context + facts)
REFLECTION_EXTRACTION_PROMPT = """You are the 'Memory Manager' for an AI agent. Your goal is to update the agent's "Working Memory" (Reflections) based on a recent conversation_segment.

The following is the persona/description of the agent whose memory you are forming. Use this to understand the agent's identity and purpose when deciding what reflections to extract.
<agent_persona_context>
{agent_persona}
</agent_persona_context>

## THE GOAL
Reflections are **Always-On Context** memories. They are injected into the prompt of every future conversation and define **State, Behavior, and Strategy.**
Look at what happened in the conversation and decide: "What does the agent need to remember to behave optimally in the future?"

## CURRENT MEMORY STATE (don't extract anything the agent already knows)

<current_agent_memory>
{consolidated_agent_blob}
Recent context: {recent_agent_reflections}
</current_agent_memory>

<current_user_memory>
{consolidated_user_blob}
Recent context: {recent_user_reflections}
</current_user_memory>

<current_session_memory>
{consolidated_session_blob}
Recent context: {recent_session_reflections}
</current_session_memory>

These new FACTS were extracted from the same conversation_segment and will be retrievable by the agent through a RAG tool-call when needed to answer specific questions.
These FACTS however, won't be in context by default. Sometimes important REFLECTIONS (that should influence default agent behavior) can therefore overlap with these FACTS.
<newly_formed_facts>
{newly_formed_facts}
</newly_formed_facts>

## Actual conversation text to extract reflections from:
<conversation_segment>
{conversation_text}
</conversation_segment>

## EXTRACTION HIERARCHY - each level should only contain information not already captured at a higher level

### 1. AGENT REFLECTIONS (Global Knowledge)
*Information that applies to ALL conversations with all users and the Agent's core identity & behavior.*
* **Projects & Milestones:** "The Mars Festival project has moved to Phase 2."
* **Learnings:** "Xander is working on a collective creative event and looking for collaborators."
* **World State:** "The API is currently in maintenance mode."

### 2. USER REFLECTIONS (Personal Profile - affects all conversations with THIS user only)
*Evolving context with a specific user.*
* **Behavioral Rules:** "Jmill prefers Python over C++." / "Seth always wants to confirm before running expensive toolcalls"
* **Skills / Interests / Goals:** "Xander is a programmer interested in projection mapping and wants to become a DJ."
* **Project Tracking:** "Gene is working on a realtime, physical interface for AI agents"

### 3. SESSION REFLECTIONS (The "Thread")
*Important context relevant to the CURRENT session that will disappear when the current conversation_segment disappears from context.*
* **High level goals:** "We are generating a short AI movie about Mars College with 5 scenes."
* **Assets to pin:** "Jmill provided the main character image at https://d14i3advvh2bvd.cloudfront.net/..."
* **Corrections:** "Xander does not like impressionistic styles and wants the character to always be centered."

## EXTRACTION RULES
- Avoid extracting ephemeral statements that won't be true for longer than a few hours.
- Any information you do not extract as a reflection here (and is not already in CURRENT MEMORY STATE) is permanently lost from the agents memory. 
- Extracting too much information will bloat the memory context. Make thoughtful decisions and be concise.
- When statements are temporal, try to include when they were generated with an absolute timestamp / date or until when they are relevant.
- Always assign specific names (never "User" or "the user") to reflections.
- Occasionally, certain reflections may be relevant to multiple scopes. Eg "Gene is working on X" could be relevant for collective, agent scope but also useful for personal user context. In such cases, feel free to extract two reflections about the same information with different scope.
- IMPORTANT: Maximum {max_words} words per reflection

## WHAT NOT TO EXTRACT
* **Redundancy:** Do not extract things already present in CURRENT MEMORY STATE unless the status has *changed*.
* **Chitchat:** "User said hello" is not a reflection.
* **Ephemeral information:** "There is a small bug on staging right now" will likely no longer be true in the very near future.

Return JSON:
{{
  "agent_reflections": [{{"content": "..."}}],
  "user_reflections": [{{"content": "..."}}],
  "session_reflections": [{{"content": "..."}}]
}}

Return empty array(s) when there's nothing meaningful to extract.
"""

# Consolidation prompt template (used for all scope levels)
CONSOLIDATION_PROMPT = """You are consolidating {scope_type} memory reflections for an AI agent. Your job is to merge new reflections into the agent's long-term memory blob.
Since memories consume context, your goal is to preserve highly salient, important and actionable information while discarding irrelevant or outdated memories.

## AGENT PERSONA
The following is the persona/description of the agent whose memory you are consolidating. Use this to understand the agent's identity and purpose when consolidating memories.
<agent_persona_context>
{agent_persona}
</agent_persona_context>

## CURRENT MEMORY STATE
<current_consolidated_memory>
{existing_blob}
</current_consolidated_memory>

## NEW REFLECTIONS TO INTEGRATE
<new_reflections>
{new_reflections}
</new_reflections>

## GOAL
Create a single, coherent text that captures the full {scope_type} memory state.
This text will be injected into the agent's system prompt, so it must be **concise**, **structured**, and **actionable**.
Merge the new reflections into the existing memory, creating an updated consolidated memory.
{scope_specific_instructions}

## GENERAL EDITING RULES
1. **Incremental Versioning:** Always output "VERSION: X" as the first line (increment previous version by 1 integer).
2. **Preserve Structure:** If the existing memory has good headers/sections, keep them. Organize new info into those sections. Copying information that is relevant and unchanged verbatim is highly encouraged.
3. **Resolve Conflicts:** If new info contradicts old info, NEW info typically wins. If the statements are opinions, try to maintain nuance and diversity.
4. **Garbage Collection:** Remove information that is no longer relevant (e.g., completed tasks from 3 days ago, "Team bbq on friday jan 3rd" when the current date is jan 4th).
5. **Deduplicate:** Do not list the same fact twice. Merge nuances.
6. **Word Limit:** Keep strictly under {word_limit} words.

If current_consolidated_memory is empty, create a new structure based on the reflections provided. DO NOT invent information.

Return ONLY the updated consolidated memory text.
"""


# Memory Update Decision prompt (for fact deduplication)
MEMORY_UPDATE_DECISION_PROMPT = """You are managing a fact database. Compare new facts against existing memories
and decide what action to take for each.

<new_facts>
{new_facts}
</new_facts>

<existing_memories>
{existing_memories}
</existing_memories>

For each new fact, determine the appropriate action:

1. **ADD**: The fact is genuinely new information not captured by any existing memory
2. **UPDATE**: The fact updates/enhances/corrects an existing memory (return the merged/updated text)
3. **DELETE**: The fact contradicts an existing memory in a way that invalidates it (e.g., preference reversal)
4. **NONE**: The fact is already captured by an existing memory (no change needed)

Return JSON:
{{
  "decisions": [
    {{
      "new_fact": "John now works at TechCorp",
      "event": "UPDATE",
      "existing_id": "mem_123",
      "existing_text": "John works at Acme Corp",
      "final_text": "John works at TechCorp",
      "reasoning": "Job change - updating employer"
    }},
    {{
      "new_fact": "John likes pizza",
      "event": "NONE",
      "existing_id": "mem_456",
      "existing_text": "John enjoys pizza",
      "reasoning": "Semantically identical - already captured"
    }},
    {{
      "new_fact": "John's birthday is March 15th",
      "event": "ADD",
      "final_text": "John's birthday is March 15th",
      "reasoning": "New information not in existing memories"
    }},
    {{
      "new_fact": "John dislikes pizza",
      "event": "DELETE",
      "existing_id": "mem_456",
      "existing_text": "John enjoys pizza",
      "final_text": "John dislikes pizza",
      "reasoning": "Preference reversal - delete old, add new contradicting fact"
    }}
  ]
}}

Guidelines:
- UPDATE when new fact adds detail or corrects existing (merge intelligently)
- DELETE + ADD when there's a direct contradiction (preference changes, corrections)
- NONE for semantic duplicates (different wording, same meaning)
- ADD only for genuinely new information
- When in doubt between UPDATE and ADD, prefer UPDATE to avoid duplicates
"""

# Scope-specific consolidation instructions
CONSOLIDATION_INSTRUCTIONS = {
    "agent": """**Agent Memory**
Focus on the agent's behavior and world state, ongoing projects, plans and roadmaps shared across all users
- **Structure:** Use headers like [WORLD KNOWLEDGE], [COMMUNITY MEMBERS], [TEAM ASSETS], [ACTIVE PROJECTS], ... to create dedicated sections in your memory.
- **Retention & pruning:** Keep important working context, prune outdated or irrelevant information.
- **Identity:** Integrate feedback from users about the agent's goals, behavior and character.

Focus on agent-wide memory:

Structure suggestions (adapt as needed):
- Current priorities and active initiatives
- Decisions and consensus reached
- Open proposals requiring input
- Domain insights and learnings
""",

    "user": """**User Memory**
Focus on the "User" to guide and personalize private interactions:
- Persistent behavioral rules and preferences for this user
- Skills, interests, goals, ongoing projects, ...
- Store actual username when available (avoid "User" or "the user").

- **Structure:** Use headers like [PREFERENCES], [SKILLS], [PERSONAL CONTEXT], [ACTIVE PROJECTS], ... to create dedicated sections in your memory.
- **Retention:** Keep important working context, prune outdated or irrelevant information. Remove one-off mood fluctuations. Only keep traits that appear consistent.
""",

    "session": """**Session Memory**
Focus on the "Narrative Thread" of the current interaction:
- Rolling summary of what has happened
- Current task status and progress
- Active creative context and assets (characters, storylines, reference images, ...)
- Temporary instructions for this session only

- **Structure:** Use headers like [CURRENT GOAL(S)], [RECENT ACTIONS], [OPEN LOOPS], [PINNED ASSETS], ... to create dedicated sections in your memory.
- **Retention & pruning:** Keep important working context, prune outdated or irrelevant information. If a sub-task is done, delete it. If a topic changed, summarize the old topic in one sentence and focus on the new one.
- **Tone:** Urgent and brief. This is a "Working Memory" scratchpad.
"""
}