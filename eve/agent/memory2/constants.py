"""
Memory System v2 - Constants and Configuration

This module contains all configurable thresholds, limits, and prompts for the
memory system. Values are tuned based on production experience from memory v1.

Prompts are structured as modular chunks that can be conditionally assembled
based on which memory scopes are enabled (user_memory_enabled, agent_memory_enabled).
Session memory is always active when any memory is enabled.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

from eve.agent.session.config import DEFAULT_SESSION_SELECTION_LIMIT


# -----------------------------------------------------------------------------
# Memory2 Configuration
# -----------------------------------------------------------------------------
@dataclass
class Memory2Config:
    """
    Configuration for memory2 system based on agent settings.

    Controls which memory scopes are enabled for both extraction and assembly.

    IMPORTANT: Session memory is ALWAYS active, regardless of user/agent toggles.
    This ensures session reflections are always formed and injected into context.

    IMPORTANT: User memory is automatically disabled for multi-user sessions (group chats)
    to prevent memory leakage between users. In multi-user sessions, only session and
    agent scoped memories are formed.
    """
    user_enabled: bool = False
    agent_enabled: bool = False
    is_multi_user: bool = False  # True when session has multiple users

    @property
    def fact_scopes(self) -> List[str]:
        """
        Get enabled scopes for fact extraction (user/agent only, no session facts).

        User scope is disabled for multi-user sessions to prevent leakage.
        """
        scopes = []
        # User facts disabled in multi-user sessions
        if self.user_enabled and not self.is_multi_user:
            scopes.append("user")
        if self.agent_enabled:
            scopes.append("agent")
        return scopes

    @property
    def reflection_scopes(self) -> List[str]:
        """
        Get enabled scopes for reflection extraction.

        Session is ALWAYS included - session reflections are never disabled.
        User scope is disabled for multi-user sessions to prevent leakage.
        """
        scopes = ["session"]  # Session ALWAYS active
        # User reflections disabled in multi-user sessions
        if self.user_enabled and not self.is_multi_user:
            scopes.append("user")
        if self.agent_enabled:
            scopes.append("agent")
        return scopes

    @property
    def any_enabled(self) -> bool:
        """Check if user or agent memory is enabled (for fact extraction)."""
        return self.user_enabled or self.agent_enabled

    @property
    def session_always_active(self) -> bool:
        """Session memory is always active - this always returns True."""
        return True

    @classmethod
    def from_agent(cls, agent, session=None) -> "Memory2Config":
        """
        Create config from an Agent object, optionally with session context.

        Args:
            agent: Agent object with memory settings
            session: Optional session object to detect multi-user context

        Returns:
            Memory2Config with appropriate settings
        """
        is_multi_user = is_multi_user_session(session) if session else False

        return cls(
            user_enabled=getattr(agent, "user_memory_enabled", False),
            agent_enabled=getattr(agent, "agent_memory_enabled", False),
            is_multi_user=is_multi_user,
        )

    @classmethod
    def from_agent_id(cls, agent_id, session=None) -> "Memory2Config":
        """
        Create config by loading agent from database, optionally with session context.

        Args:
            agent_id: Agent ObjectId
            session: Optional session object to detect multi-user context

        Returns:
            Memory2Config with appropriate settings
        """
        try:
            from eve.agent.agent import Agent
            agent = Agent.from_mongo(agent_id)
            if agent:
                return cls.from_agent(agent, session=session)
        except Exception:
            pass
        return cls()  # Default: all disabled


def is_multi_user_session(session) -> bool:
    """
    Check if a session is a multi-user session (group chat).

    Multi-user sessions have more than one user in the users list.
    In multi-user sessions, user-scoped memories should be disabled
    to prevent memory leakage between users.

    Args:
        session: Session object with users attribute (List[ObjectId] in MongoDB)

    Returns:
        True if session has multiple users, False otherwise
    """
    if session is None:
        return False

    users = getattr(session, "users", None) or []
    return len(users) > 1

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
  MEMORY_FORMATION_TOKEN_INTERVAL = 1500  # ~1k tokens triggers formation
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
    "user": 300,  # Medium - user preferences and interaction style
    "session": 200,  # Rolling summary of session events and status
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
# Prompt Templates - Modular Chunks
# -----------------------------------------------------------------------------
# Prompts are broken into modular chunks that can be conditionally assembled
# based on enabled memory scopes (user_memory_enabled, agent_memory_enabled).


# =============================================================================
# FACT EXTRACTION PROMPT CHUNKS
# =============================================================================

FACT_PROMPT_HEADER = """You are the 'Librarian' for an AI agent system. Your job is to extract concrete, searchable data points from a conversation to be stored in a RAG (retrieval) database system.

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
- ALWAYS assign specific usernames (NEVER "User", "the user", or "they") and absolute dates (NEVER use "tomorrow")

DO NOT extract as facts:
- **Preferences & Behavior:** "Alice likes concise responses"
- **Current Context:** "We are debugging the login issue"
- **Opinions:** "Bob thinks we should use React"
- **Ephemeral State:** "There is a bug in production right now" or "Gene has lost his headphones"
All of the above will be captured as reflections and are not FACTS."""

# Scope definition chunks - conditionally included
FACT_SCOPE_USER = """* **"user"**: Private details about the specific user interacting right now (names, contact info, job, specific preferences that act as data)."""

FACT_SCOPE_AGENT = """* **"agent"**: Collective knowledge, world-building, or project details that apply to *anyone* talking to this agent (team roster, project specs, global deadlines)."""

# Example chunks - conditionally included based on enabled scopes
FACT_EXAMPLES_HEADER = """## EXAMPLES
| Text | Extract? | Scope | Reasoning |
| :--- | :--- | :--- | :--- |
| "My name is John Doe." | NO | - | This is a highly salient detail that should always be in context. (Reflection) |
| "I like concise answers." | NO | - | This is a behavioral preference (Reflection), not a database fact. |
| "We are debugging the login." | NO | - | This is a current status/state (Reflection). |
| "I'm feeling sad today." | NO | - | Temporary emotional state. |
| "I'm currently camping in Paris." | NO | - | Temporary state. |
| "Jordan requested an image of a buff golden retriever." | NO | - | This is an ephemeral event, not actionable information. |"""

FACT_EXAMPLES_USER = """| "My personal email is j.smith@gmail.com." | YES | user | Specific, retrieval-worthy data. |"""

FACT_EXAMPLES_AGENT = """| "The total project budget is $50k." | YES | agent | Specific number, relevant to the project. |
| "The server IP is 192.168.1.1" | YES | agent | Specific, retrieval-worthy data. |
| "The Figma link for the new design is figma.com/file/xyz..." | YES | agent | Specific, retrieval-worthy data. |"""

FACT_PROMPT_FOOTER = """
## INSTRUCTIONS
Read the conversation below. Extract ONLY facts that meet the criteria (if any). Most conversations have FEW or NO facts.

<conversation>
{conversation_text}
</conversation>

Return a JSON object:
{{
  "facts": [
{json_examples}
  ]
}}
If no facts are found, return {{ "facts": [] }}.
"""

FACT_JSON_EXAMPLE_USER = '    {{ "content": "User\'s email is john@example.com", "scope": "user" }}'
FACT_JSON_EXAMPLE_AGENT = '    {{ "content": "Project Apollo launch date is June 1st", "scope": "agent" }}'


def build_fact_extraction_prompt(
    conversation_text: str,
    agent_persona: str,
    enabled_scopes: List[str],
    max_words: int = FACT_MAX_WORDS,
) -> str:
    """
    Build fact extraction prompt with only enabled scopes.

    Args:
        conversation_text: The conversation to extract facts from
        agent_persona: Agent persona/description
        enabled_scopes: List of enabled scopes ["user", "agent"] or subset
        max_words: Maximum words per fact

    Returns:
        Complete prompt string
    """
    if not enabled_scopes:
        return ""

    parts = [FACT_PROMPT_HEADER.format(agent_persona=agent_persona, max_words=max_words)]

    # Add scope definitions
    scope_defs = []
    if "user" in enabled_scopes:
        scope_defs.append(FACT_SCOPE_USER)
    if "agent" in enabled_scopes:
        scope_defs.append(FACT_SCOPE_AGENT)

    if scope_defs:
        parts.append("\n\n## SCOPE DEFINITIONS")
        parts.append("\n".join(scope_defs))

    # Add examples
    parts.append("\n\n" + FACT_EXAMPLES_HEADER)
    if "agent" in enabled_scopes:
        parts.append(FACT_EXAMPLES_AGENT)
    if "user" in enabled_scopes:
        parts.append(FACT_EXAMPLES_USER)

    # Add footer with JSON examples
    json_examples = []
    if "user" in enabled_scopes:
        json_examples.append(FACT_JSON_EXAMPLE_USER)
    if "agent" in enabled_scopes:
        json_examples.append(FACT_JSON_EXAMPLE_AGENT)

    parts.append(FACT_PROMPT_FOOTER.format(
        conversation_text=conversation_text,
        json_examples=",\n".join(json_examples) if json_examples else ""
    ))

    return "\n".join(parts)

# =============================================================================
# REFLECTION EXTRACTION PROMPT CHUNKS
# =============================================================================

REFLECTION_PROMPT_HEADER = """You are the 'Memory Manager' for an AI agent. Your goal is to update the agent's "Working Memory" (Reflections) based on a recent conversation_segment.

The following is the persona/description of the agent whose memory you are forming. Use this to understand the agent's identity and purpose when deciding what reflections to extract.
<agent_persona_context>
{agent_persona}
</agent_persona_context>

## THE GOAL
Reflections are **Always-On Context** memories. They are injected into the prompt of every future conversation and define **State, Behavior, and Strategy.**
Look at what happened in the conversation and decide: "What does the agent need to remember to behave optimally in the future?"

## CURRENT MEMORY STATE (don't extract anything the agent already knows)
"""

# Memory context chunks - conditionally included
REFLECTION_MEMORY_AGENT = """
<current_agent_memory>
{consolidated_agent_blob}
Recent context: {recent_agent_reflections}
</current_agent_memory>
"""

REFLECTION_MEMORY_USER = """
<current_user_memory>
{consolidated_user_blob}
Recent context: {recent_user_reflections}
</current_user_memory>
"""

REFLECTION_MEMORY_SESSION = """
<current_session_memory>
{consolidated_session_blob}
Recent context: {recent_session_reflections}
</current_session_memory>
"""

REFLECTION_FACTS_SECTION = """
These new FACTS were extracted from the same conversation_segment and will be retrievable by the agent through a RAG tool-call when needed to answer specific questions.
These FACTS however, won't be in context by default. Sometimes important REFLECTIONS (that should influence default agent behavior) can therefore overlap with these FACTS.
<newly_formed_facts>
{newly_formed_facts}
</newly_formed_facts>
"""

REFLECTION_CONVERSATION_SECTION = """
## Actual conversation text to extract reflections from:
<conversation_segment>
{conversation_text}
</conversation_segment>

## EXTRACTION HIERARCHY - each level should only contain information not already captured at a higher level
"""

# Hierarchy chunks - conditionally included based on enabled scopes
REFLECTION_HIERARCHY_AGENT = """
### AGENT REFLECTIONS (Global Knowledge)
*Information that applies to ALL conversations with all users and the Agent's core identity & behavior.*
* **Projects & Milestones:** "The Mars Festival project has moved to Phase 2."
* **Learnings:** "Xander is working on a collective creative event and looking for collaborators."
* **World State:** "The API is currently in maintenance mode."
"""

REFLECTION_HIERARCHY_USER = """
### USER REFLECTIONS (Personal Profile - affects all conversations with THIS user only)
*Evolving context with a specific user.*
* **Behavioral Rules:** "Jmill prefers Python over C++." / "Seth always wants to confirm before running expensive toolcalls"
* **Skills / Interests / Goals:** "Xander is a programmer interested in projection mapping and wants to become a DJ."
* **Project Tracking:** "Gene is working on a realtime, physical interface for AI agents"
"""

REFLECTION_HIERARCHY_SESSION = """
### SESSION REFLECTIONS (The "Thread")
*Important context relevant to the CURRENT session that will disappear when the current conversation_segment disappears from context.*
* **High level goals:** "We are generating a short AI movie about Mars College with 5 scenes."
* **Assets to pin:** "Jmill provided the main character image at https://d14i3advvh2bvd.cloudfront.net/..."
* **Corrections:** "Xander does not like impressionistic styles and wants the character to always be centered."
"""

REFLECTION_PROMPT_RULES = """
## EXTRACTION RULES
- Avoid extracting ephemeral statements that won't be true for longer than a few hours.
- Any information you do not extract as a reflection here (and is not already in CURRENT MEMORY STATE) is permanently lost from the agents memory.
- Extracting too much information will bloat the memory context. Make thoughtful decisions, extract only salient information and be concise.
- When statements are temporal, try to include when they were generated with an absolute timestamp / date or until when they are relevant.
- ALWAYS assign specific usernames (NEVER "User", "the user", or "they") and absolute dates (NEVER use "tomorrow")
- Occasionally, certain reflections may be relevant to multiple scopes. Eg "Gene is working on X" could be relevant for collective, agent scope but also useful for personal user context. In such cases, feel free to extract two reflections about the same information with different scope.
- IMPORTANT: Maximum {max_words} words per reflection

## WHAT NOT TO EXTRACT
* **Redundancy:** Do not extract things already present in CURRENT MEMORY STATE unless the status has *changed*.
* **Chitchat:** "User said hello" is not a reflection.
* **Ephemeral information:** "There is a small bug on staging right now" will likely no longer be true in the very near future.

Return JSON:
{{
{json_fields}
}}

Return empty array(s) when there's nothing meaningful to extract.
"""


def build_reflection_extraction_prompt(
    conversation_text: str,
    agent_persona: str,
    memory_context: dict,
    newly_formed_facts: str,
    enabled_scopes: List[str],
    max_words: int = REFLECTION_MAX_WORDS,
) -> str:
    """
    Build reflection extraction prompt with only enabled scopes.

    Args:
        conversation_text: The conversation to extract reflections from
        agent_persona: Agent persona/description
        memory_context: Dict with keys like "agent_blob", "agent_recent", etc.
        newly_formed_facts: Formatted string of newly formed facts
        enabled_scopes: List of enabled scopes ["session", "user", "agent"]
        max_words: Maximum words per reflection

    Returns:
        Complete prompt string
    """
    parts = [REFLECTION_PROMPT_HEADER.format(agent_persona=agent_persona)]

    # Add memory context sections for enabled scopes
    if "agent" in enabled_scopes:
        parts.append(REFLECTION_MEMORY_AGENT.format(
            consolidated_agent_blob=memory_context.get("agent_blob") or "None yet",
            recent_agent_reflections=memory_context.get("agent_recent") or "None",
        ))

    if "user" in enabled_scopes:
        parts.append(REFLECTION_MEMORY_USER.format(
            consolidated_user_blob=memory_context.get("user_blob") or "None yet",
            recent_user_reflections=memory_context.get("user_recent") or "None",
        ))

    # Session is always included
    if "session" in enabled_scopes:
        parts.append(REFLECTION_MEMORY_SESSION.format(
            consolidated_session_blob=memory_context.get("session_blob") or "None yet",
            recent_session_reflections=memory_context.get("session_recent") or "None",
        ))

    # Add facts section
    parts.append(REFLECTION_FACTS_SECTION.format(newly_formed_facts=newly_formed_facts))

    # Add conversation section
    parts.append(REFLECTION_CONVERSATION_SECTION.format(conversation_text=conversation_text))

    # Add hierarchy sections for enabled scopes (order: agent -> user -> session)
    hierarchy_num = 1
    if "agent" in enabled_scopes:
        parts.append(f"\n### {hierarchy_num}." + REFLECTION_HIERARCHY_AGENT.lstrip("\n### "))
        hierarchy_num += 1

    if "user" in enabled_scopes:
        parts.append(f"\n### {hierarchy_num}." + REFLECTION_HIERARCHY_USER.lstrip("\n### "))
        hierarchy_num += 1

    if "session" in enabled_scopes:
        parts.append(f"\n### {hierarchy_num}." + REFLECTION_HIERARCHY_SESSION.lstrip("\n### "))

    # Build JSON fields based on enabled scopes
    json_fields = []
    if "agent" in enabled_scopes:
        json_fields.append('  "agent_reflections": [{{"content": "..."}}]')
    if "user" in enabled_scopes:
        json_fields.append('  "user_reflections": [{{"content": "..."}}]')
    if "session" in enabled_scopes:
        json_fields.append('  "session_reflections": [{{"content": "..."}}]')

    parts.append(REFLECTION_PROMPT_RULES.format(
        max_words=max_words,
        json_fields=",\n".join(json_fields),
    ))

    return "".join(parts)

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