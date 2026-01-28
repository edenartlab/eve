"""
Memory System v2 - Constants and Configuration (Refactored)

This module contains all configurable thresholds, limits, and prompts for the
memory system. Values are tuned based on production experience from memory v1.

REFACTORED APPROACH:
- Prompts are now defined as complete, readable templates with section markers
- Section markers use {# SECTION:name #} ... {# END:name #} syntax
- A post-processor removes disabled sections and cleans up whitespace
- This makes prompts readable as complete documents while preserving conditional behavior

Session memory is always active when any memory is enabled.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from eve.agent.session.config import DEFAULT_SESSION_SELECTION_LIMIT

# =============================================================================
# Development/Production Toggle
# =============================================================================
LOCAL_DEV = False  # Set to False for production


# =============================================================================
# LLM Model Configuration
# =============================================================================
# Fast model is always flash - used for fact extraction
MEMORY_LLM_MODEL_FAST = "gemini-3-flash-preview"

# Model names for slow model (reflection extraction, consolidation)
_MEMORY_LLM_MODEL_SLOW_FREE = "gemini-3-flash-preview"
_MEMORY_LLM_MODEL_SLOW_PREMIUM = "gemini-3-pro-preview"

# =============================================================================
# Memory Formation Triggers
# =============================================================================
if LOCAL_DEV:
    MEMORY_FORMATION_MSG_INTERVAL = 4
    MEMORY_FORMATION_TOKEN_INTERVAL = 500
    NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = 2
    CONSIDER_COLD_AFTER_MINUTES = 10
else:
    MEMORY_FORMATION_MSG_INTERVAL = (
        DEFAULT_SESSION_SELECTION_LIMIT  # Messages between formations
    )
    MEMORY_FORMATION_TOKEN_INTERVAL = 1500  # ~1.5k tokens triggers formation
    NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = (
        4  # Minimum before attempting to form memories
    )
    CONSIDER_COLD_AFTER_MINUTES = (
        10  # Session inactivity threshold for cold session processing
    )

# =============================================================================
# Consolidation Thresholds
# =============================================================================
# Number of unabsorbed reflections before triggering consolidation
if LOCAL_DEV:
    CONSOLIDATION_THRESHOLDS = {
        "agent": 2,
        "user": 2,
        "session": 2,
    }
else:
    CONSOLIDATION_THRESHOLDS = {
        "agent": 10,
        "user": 4,
        "session": 4,
    }

# Maximum word count for consolidated blobs
CONSOLIDATED_WORD_LIMITS = {
    "agent": 1200,  # Largest - agent's full persona/project state
    "user": 300,  # Medium - user preferences and interaction style
    "session": 2000,  # Rich session state - accumulates for long sessions (games, complex tasks)
}

# =============================================================================
# Memory Word Limits (individual items)
# =============================================================================
FACT_MAX_WORDS = 30  # Per fact item
REFLECTION_MAX_WORDS = 35  # Per reflection item


# =============================================================================
# Embedding Configuration (for fact storage)
# =============================================================================
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Threshold for FACTS deduplication:
SIMILARITY_THRESHOLD = 0.7

# RAG Retrieval Thresholds
RAG_SEMANTIC_SCORE_THRESHOLD = 0.65  # Min vectorSearchScore (cosine similarity, 0-1)
RAG_TEXT_SCORE_THRESHOLD = 1.5  # Min searchScore (BM25-based, unbounded)
# RRF threshold: With k=60, a single-source rank-0 result scores 1/60 = 0.0167
# Setting threshold to 0.015 allows high-ranking single-source results through
# (important when semantic finds synonyms that text search misses, e.g. "pottery" -> "ceramics")
RAG_RRF_SCORE_THRESHOLD = 0.015  # Min RRF score after fusion (don't change this!)

# =============================================================================
# Feature Toggles
# =============================================================================
ALWAYS_IN_CONTEXT_ENABLED = True  # Can be toggled independently

# =============================================================================
# Facts FIFO Mode
# =============================================================================
# Facts are extracted, embedded, and stored - retrieved via simple FIFO
# (most recent N facts within age limit) for always-in-context injection.
#
# RAG retrieval is implemented as a separate tool call in the agent stack,
# allowing agents to explicitly search their memory when needed.
#
# FIFO filtering:
# - Only facts within FACTS_FIFO_MAX_AGE_HOURS are included
# - Limited to FACTS_FIFO_LIMIT facts (oldest dropped if exceeded)
# =============================================================================
FACTS_FIFO_ENABLED = True  # Enable FIFO facts in context
FACTS_FIFO_LIMIT = 40  # Max number of recent facts to include
FACTS_FIFO_MAX_AGE_HOURS = 24 * 7  # Only include facts from last 48 hours

# =============================================================================
# FACT EXTRACTION PROMPT TEMPLATE
# =============================================================================
# This is the complete fact extraction prompt with section markers.
# Sections marked with {# SECTION:user #} are included only when user scope is enabled.
# Sections marked with {# SECTION:agent #} are included only when agent scope is enabled.

FACT_EXTRACTION_TEMPLATE = """You are the 'Librarian' for an AI agent system. Your job is to extract concrete, searchable data points from a conversation to be stored in a RAG (retrieval) database system.

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
3. **Enduring:** Is this information highly likely to remain true for at least a month?
4. **Cold Storage:** Is this information okay to "forget" until specifically searched for?

This means facts should be SEARCHABLE ANSWERS to specific questions that are not otherwise relevant to the agent (and therefore shouldn't generally be in context).
After this extraction step of FACTS, you will also get to extract REFLECTIONS from the same conversation, which are always-in-context memories.
If knowledge needs to be always in context (to consistently influence agent behavior), rather than occasionally retrieved, don't extract it as FACT, but leave it to the REFLECTION system.

DO NOT extract as facts:
- **Preferences & Behavior:** "Alice likes concise responses"
- **Current Context:** "We are debugging the login issue"
- **Opinions:** "Bob thinks we should use React"
- **Ephemeral State:** "There is a bug in production right now" or "Gene has lost his headphones"

All of the above will be captured as REFLECTIONS and are thus not FACTS.

## SCOPE DEFINITIONS
{# SECTION:user #}
* **"user"**: Private details about the specific user interacting right now (names, contact info, job, specific preferences that act as data).
{# END:user #}
{# SECTION:agent #}
* **"agent"**: Collective knowledge, world-building, or project details that apply to *anyone* talking to this agent (team roster, project specs, global deadlines).
{# END:agent #}

## EXAMPLES
| Text | Extract? | Scope | Reasoning |
| :--- | :--- | :--- | :--- |
| "My name is John Doe." | NO | - | This is a highly salient detail that should always be in context. (Reflection) |
| "I like concise answers." | NO | - | This is a behavioral preference (Reflection), not a database fact. |
| "We are debugging the login." | NO | - | This is a current status/state (Reflection). |
| "I'm feeling sad today." | NO | - | Temporary emotional state. |
| "I'm currently camping in Paris." | NO | - | Temporary state. |
| "Jordan requested an image of a buff golden retriever." | NO | - | This is an ephemeral event, not actionable information. |
{# SECTION:agent #}
| "The total project budget is $50k." | YES | agent | Specific number, relevant to the project. |
| "The server IP is 192.168.1.1" | YES | agent | Specific, retrieval-worthy data. |
| "The Figma link for the new design is figma.com/file/xyz..." | YES | agent | Specific, retrieval-worthy data. |
{# END:agent #}
{# SECTION:user #}
| "My personal email is j.smith@gmail.com." | YES | user | Specific, retrieval-worthy data. |
{# END:user #}

## INSTRUCTIONS
- Facts must be self-contained statements that make sense without any additional context.
- **CRITICAL - ABSOLUTE TIMESTAMPS:** NEVER store relative time references like "tomorrow", "next week", "in 3 days", "this Friday". ALWAYS convert to absolute dates/times (e.g., "January 30th 2025 at 2pm UTC"). Use the UTC timestamps from the conversation to calculate exact dates. If a user says "the event is tomorrow at 3pm" and their message timestamp is Jan 28th 2025 14:00 UTC, store: "Event on January 29th 2025" (adjust for user's likely timezone if context suggests they're not in UTC).
- ALWAYS assign specific usernames (NEVER "User", "the user", or "they").
- Maximum {max_words} words per fact, always be concise!
- Prioritize information explicitly stated by users. Agent messages may contain assumptions, suggestions, or inferences that haven't been verified—only extract agent-stated information if the user confirmed or agreed with it.

Read the conversation_segment below. Extract ONLY facts that meet the criteria (if any). Most conversations have FEW or NO facts.

<conversation_segment>
{conversation_text}
</conversation_segment>

Return a JSON object:
{{
  "facts": [
{json_examples}
  ]
}}
If no facts are found, return {{ "facts": [] }}.
"""


# =============================================================================
# REFLECTION EXTRACTION PROMPT TEMPLATE
# =============================================================================
# This is the complete reflection extraction prompt with section markers.
# Memory sections are injected as pre-formatted content.
# Hierarchy sections are conditionally included based on enabled scopes.

REFLECTION_EXTRACTION_TEMPLATE = """You are the 'Memory Manager' for an AI agent. Your goal is to update the agent's "Working Memory" (Reflections) based on a recent conversation_segment.

The following is the persona/description of the agent whose memory you are forming. Use this to understand the agent's identity and purpose when deciding what reflections to extract.
<agent_persona_context>
{agent_persona}
</agent_persona_context>

## THE GOAL
Reflections are **Always-On Context** memories. They are injected into the prompt of every future conversation and define **State, Behavior, and Strategy.**
Look at what happened in the conversation and decide: "What does the agent need to remember to behave optimally in the future?"

## CURRENT MEMORY STATE (don't extract anything the agent already knows)
{memory_sections}{# SECTION:has_facts #}These new FACTS were extracted from the same conversation_segment and will be retrievable by the agent through a RAG tool-call when needed to answer specific questions.
These FACTS however, won't be in context by default. Sometimes important REFLECTIONS (that should influence default agent behavior) can therefore overlap with these FACTS.
<newly_formed_facts>
{newly_formed_facts}
</newly_formed_facts>

{# END:has_facts #}## EXTRACTION HIERARCHY - each level should only contain information not already captured at a higher level
{# SECTION:agent #}

### {agent_hierarchy_num}.AGENT REFLECTIONS (Global Knowledge)
*Information that applies to ALL conversations with all users and the Agent's core identity & behavior.*
* **Projects & Milestones:** "The Mars Festival project has moved to Phase 2."
* **Learnings:** "Xander is working on a collective creative event and looking for collaborators."
* **World State:** "The API is currently in maintenance mode."
{# END:agent #}
{# SECTION:user #}

### {user_hierarchy_num}.USER REFLECTIONS (Personal Profile - affects all conversations with THIS user only)
*Evolving context with a specific user.*
* **Behavioral Rules:** "Jmill prefers Python over C++." / "Seth always wants to confirm before running expensive toolcalls"
* **Skills / Interests / Goals:** "Xander is a programmer interested in projection mapping and wants to become a DJ."
* **Project Tracking:** "Gene is working on a realtime, physical interface for AI agents"
{# END:user #}
{# SECTION:session #}

### {session_hierarchy_num}.SESSION REFLECTIONS (The "Thread")
*Important context relevant to the CURRENT session that will disappear when the current conversation_segment disappears from context.*
* **High level goals:** "This session is creating a 5-scene AI-generated short film about Mars College, an off-grid desert community focused on AI and art. Current phase: scene 3 of 5."
* **Assets to pin (MUST include URLs):** "Main character reference image (desert wanderer in Byzantine mosaic style): https://d14i3advvh2bvd.cloudfront.net/abc123.png" / "Project brief document: https://docs.google.com/doc/d/xyz789"
* **Corrections & preferences:** "Gene (session owner) rejected impressionistic styles for this project; prefers bold, graphic aesthetics. All characters must be centered in frame."
* **Game rules & state:** "Playing Werewolf with 8 players: Alice, Bob, Charlie, David, Eve, Frank, Grace, Henry. Secret roles: Alice and Bob are werewolves. Current phase: Day 2 voting. Villagers suspect Charlie."
* **Session-specific knowledge:** "The group established naming conventions: 'blue team' = Alice, Charlie, Eve; 'red team' = Bob, David, Frank. Current score: Blue 3, Red 2."
* **Strategic context:** "After losing rounds 1 and 2, the group adopted a defensive strategy focused on protecting the crystal artifact. Key constraint: crystal cannot be moved until round 5."
{# END:session #}

## EXTRACTION RULES
- **SELF-CONTAINED CONTEXT:** Each reflection must be fully understandable on its own. The LLM reading these memories is stateless and sees them for the first time with no prior context. Never use shorthand like "the project" or "the image" - always specify "the 'Inheritance and Rights' project" or "the Byzantine mosaic reference image at [URL]". A reflection that says "Shifted to aesthetic chaos" is useless without explaining what shifted, from what, and what "aesthetic chaos" means in this context. Write as if briefing someone who knows nothing about this session.
- **PINNED ASSETS MUST INCLUDE URLs:** When pinning assets (images, documents, videos, links), ALWAYS include the full URL. "Main character image" is useless; "Main character image: https://cdn.example.com/char.png" is actionable. URLs are the whole point of pinning assets.
- **CRITICAL - ABSOLUTE TIMESTAMPS:** NEVER store relative time references. Convert ALL temporal references to absolute dates/times:
  - BAD: "Team BBQ tomorrow at 2pm" / "Meeting next Monday" / "Deadline in 3 days"
  - GOOD: "Team BBQ on January 30th 2025 at 2pm" / "Meeting on February 3rd 2025" / "Deadline: February 1st 2025"
  - Use the UTC timestamps from conversation messages to calculate exact dates. The agent's system shows current UTC time.
  - Be aware that users may reference times in their local timezone while the system operates in UTC. If a user in California says "3pm tomorrow", that's likely 3pm PST, not UTC. When timezone is unclear, note the time as stated and add "(user's local time)" or convert to UTC if you can infer their timezone.
- Avoid extracting ephemeral statements that won't be true for longer than a few hours.
- Any information you do not extract as a reflection here (and is not already in CURRENT MEMORY STATE) is permanently lost from the agents memory.
- Extracting too much information will bloat the memory context. Make thoughtful decisions, extract only salient information and be concise.
- ALWAYS assign specific usernames (NEVER "User", "the user", or "they").
- Occasionally, certain reflections may be relevant to multiple scopes. Eg "Gene is working on X" could be relevant for collective, agent scope but also useful for personal user context. In such cases, feel free to extract two reflections about the same information with different scope.
- IMPORTANT: Maximum {max_words} words per reflection
- Give more weight to information explicitly stated by users. Agent messages may contain assumptions, guesses, or hallucinations—only extract agent-stated information if the user confirmed or acknowledged it.

## WHAT NOT TO EXTRACT
* **Redundancy:** Do not extract things already present in CURRENT MEMORY STATE unless the status has *changed*.
* **Chitchat:** "User said hello" is not a reflection.
* **Ephemeral information:** "There is a small bug on staging right now" will likely no longer be true in the very near future.

## Actual conversation text to extract reflections from:
<conversation_segment>
{conversation_text}
</conversation_segment>

Return JSON:
{{
{json_fields}
}}

Return empty array(s) when there's nothing meaningful to extract.
"""


# =============================================================================
# CONSOLIDATION PROMPT (unchanged - already follows good pattern)
# =============================================================================
CONSOLIDATION_PROMPT = """You are consolidating {scope} memory reflections for an AI agent. Your job is to merge new reflections into the agent's long-term memory blob.
Since memories consume context, your goal is to preserve highly salient, important and actionable information while discarding irrelevant or outdated memories.

**CURRENT UTC TIMESTAMP: {current_utc_timestamp}**
Use this timestamp to determine which events, deadlines, and temporal information are now in the past and should be removed.

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
Create a single, coherent text that captures the full {scope} memory state.
This text will be injected into the agent's system prompt, so it must be **concise**, **structured**, and **actionable**.
Merge the new reflections into the existing memory, creating an updated consolidated memory.
{scope_specific_instructions}

## GENERAL EDITING RULES
1. **Incremental Versioning:** Always output "VERSION: X" as the first line (increment previous version by 1 integer).
2. **Preserve Structure:** If the existing memory has good headers/sections, keep them. Organize new info into those sections. Copying information that is relevant and unchanged verbatim is highly encouraged.
3. **Resolve Conflicts:** If new info contradicts old info, NEW info typically wins. If the statements are opinions, try to maintain nuance and diversity.
4. **Garbage Collection - CRITICAL:** Actively remove outdated information:
   - **Past events:** Delete any events, meetings, deadlines, or appointments that have already occurred. Compare dates in memories against the current system time (which is in UTC). If "Team BBQ on January 15th" and current date is January 16th or later, DELETE it entirely—don't keep it as "Team BBQ happened on January 15th" unless the outcome is specifically relevant.
   - **Stale temporal context:** Remove "upcoming" language for past events. If memory says "upcoming launch on March 1st" and it's now March 5th, either delete or update to reflect what happened.
   - **Completed tasks:** Remove completed tasks unless their outcomes have ongoing relevance.
   - **Expired deadlines:** Delete deadline references once the date has passed.
5. **Deduplicate:** Do not list the same fact twice. Merge nuances.
6. **Word Limit:** The current_consolidated_memory is ~{current_word_count} words. Keep the new version strictly under {word_limit} words.

If current_consolidated_memory is empty, create a new structure based on the reflections provided. DO NOT invent information.

Return ONLY the updated consolidated memory text.
"""


# =============================================================================
# MEMORY UPDATE DECISION PROMPT (unchanged - for fact deduplication)
# =============================================================================
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


# =============================================================================
# SCOPE-SPECIFIC CONSOLIDATION INSTRUCTIONS (unchanged)
# =============================================================================
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
- Game state, rules, and strategic context (for games or interactive sessions)

- **Structure:** Use headers like [CURRENT GOAL(S)], [GAME STATE], [RULES & CONSTRAINTS], [KEY CONTEXT], [RECENT ACTIONS], [OPEN LOOPS], [PINNED ASSETS], ... to create dedicated sections in your memory.
- **SELF-CONTAINED CONTEXT:** Write as if briefing someone who knows nothing about this session. The LLM reading this memory is stateless—it has no prior context. Never use shorthand like "the project" without first establishing what the project is, its purpose, and key details. Every section should be understandable without reference to anything outside this memory blob.
- **PINNED ASSETS MUST INCLUDE URLs:** The [PINNED ASSETS] section exists specifically to preserve actionable links. Every pinned asset MUST include its full URL. "Reference image for main character" is useless; "Reference image for main character (Byzantine mosaic style): https://cdn.example.com/byzantine-char.png" is actionable. If an asset doesn't have a URL, it probably shouldn't be pinned.
- **Retention & pruning:** Keep information that provides understanding of the current state. Preserve rules, constraints, and context that remain relevant. History of past events should be retained insofar as it helps understand why things are the way they are now. Completed sub-tasks can be pruned, but significant decisions or turning points should be summarized if they affect current context.
- **Tone:** Be thorough for complex sessions. This memory can grow large (up to 2000 words) for sessions with rich state like games, multi-step projects, or ongoing collaborations. Start concise and accumulate detail as the session develops complexity.
""",
}


def get_memory_llm_model_slow(
    subscription_tier: Optional[int] = None,
    has_preview_flag: bool = False,
) -> str:
    """
    Get the slow LLM model based on environment and subscription tier.

    The slow model is used for reflection extraction and consolidation.
    Premium users (subscriptionTier >= 1) or users with the "preview" feature flag
    get the pro model for better quality.
    Free users and local development use the flash model for cost savings.

    Args:
        subscription_tier: User's subscription tier (0 or None = free, >= 1 = premium)
        has_preview_flag: True if the owner has the "preview" feature flag

    Returns:
        Model name: "gemini-3-flash-preview" for LOCAL_DEV or free users,
                   "gemini-3-pro-preview" for premium users or preview flag holders
    """
    # Local development always uses fast/cheap model
    if LOCAL_DEV:
        return _MEMORY_LLM_MODEL_SLOW_FREE

    # Preview flag holders get pro model (equivalent to premium)
    if has_preview_flag:
        return _MEMORY_LLM_MODEL_SLOW_PREMIUM

    # Free users (tier < 1) or unknown tier get flash model
    if subscription_tier is None or subscription_tier < 1:
        return _MEMORY_LLM_MODEL_SLOW_FREE

    # Premium users get pro model
    return _MEMORY_LLM_MODEL_SLOW_PREMIUM


# =============================================================================
# Memory2 Configuration (unchanged from original)
# =============================================================================
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
    subscription_tier: Optional[int] = (
        None  # Owner's subscription tier for model selection
    )
    has_preview_flag: bool = False  # True if owner has "preview" feature flag

    @property
    def slow_model(self) -> str:
        """Get the slow LLM model based on owner's subscription tier or preview flag."""
        return get_memory_llm_model_slow(self.subscription_tier, self.has_preview_flag)

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

        # Get owner's subscription tier and preview flag for model selection
        subscription_tier, has_preview_flag = _get_owner_premium_status(agent)

        return cls(
            user_enabled=getattr(agent, "user_memory_enabled", False),
            agent_enabled=getattr(agent, "agent_memory_enabled", False),
            is_multi_user=is_multi_user,
            subscription_tier=subscription_tier,
            has_preview_flag=has_preview_flag,
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


def _get_owner_premium_status(agent) -> Tuple[Optional[int], bool]:
    """
    Get the premium status of the agent's owner.

    Used to determine which LLM model to use for memory operations.
    Premium users (tier >= 1) or users with the "preview" feature flag
    get the pro model, free users get flash.

    Args:
        agent: Agent object with owner attribute

    Returns:
        Tuple of (subscription_tier, has_preview_flag):
        - subscription_tier: Owner's subscription tier, or None if not found
        - has_preview_flag: True if owner has "preview" in featureFlags
    """
    try:
        owner_id = getattr(agent, "owner", None)
        if owner_id is None:
            return None, False

        # Import here to avoid circular imports
        from eve.user import User

        owner = User.from_mongo(owner_id)
        if owner is None:
            return None, False

        subscription_tier = getattr(owner, "subscriptionTier", None)
        feature_flags = getattr(owner, "featureFlags", None) or []
        has_preview_flag = "preview" in feature_flags

        return subscription_tier, has_preview_flag

    except Exception:
        # Fail silently - default to free tier model if we can't determine
        return None, False


# =============================================================================
# TEMPLATE POST-PROCESSOR
# =============================================================================
# Section marker pattern: {# SECTION:name #} ... {# END:name #}


def _process_template(
    template: str,
    values: dict,
    enabled_sections: Set[str],
) -> str:
    """
    Process a template by:
    1. Removing disabled sections (between {# SECTION:x #} and {# END:x #})
    2. Removing section markers from enabled sections
    3. Formatting with provided values
    4. Cleaning up extra blank lines

    Args:
        template: The template string with section markers
        values: Dict of values to format into the template
        enabled_sections: Set of section names to keep (others are removed)

    Returns:
        Processed template string
    """
    result = template

    # Find all section names in the template
    all_sections = set(re.findall(r"\{# SECTION:(\w+) #\}", result))

    # Remove disabled sections entirely (including their content)
    for section in all_sections - enabled_sections:
        # Pattern matches from SECTION marker to END marker, including newlines
        pattern = rf"\{{# SECTION:{section} #\}}.*?\{{# END:{section} #\}}\n?"
        result = re.sub(pattern, "", result, flags=re.DOTALL)

    # Strip section markers from enabled sections (keep content)
    result = re.sub(r"\{# SECTION:\w+ #\}\n?", "", result)
    result = re.sub(r"\{# END:\w+ #\}\n?", "", result)

    # Format with values
    result = result.format(**values)

    # Clean up multiple consecutive blank lines (3+ newlines -> 2)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def _has_content(value: Optional[str]) -> bool:
    """Check if a value has actual content (not None, empty, or placeholder)."""
    if value is None:
        return False
    stripped = value.strip()
    return bool(stripped) and stripped.lower() not in ("none", "none yet")


def _format_memory_section(
    scope: str,
    blob: Optional[str],
    recent: Optional[str],
) -> str:
    """
    Format a memory section with blob and recent content.

    Returns empty string if both blob and recent are empty/None.
    Only includes lines that have actual content.

    Args:
        scope: The scope type ("agent", "user", "session")
        blob: Consolidated blob content (or None)
        recent: Recent reflections content (or None)

    Returns:
        Formatted memory section string, or empty string if no content
    """
    has_blob = _has_content(blob)
    has_recent = _has_content(recent)

    # Return empty if no content at all
    if not has_blob and not has_recent:
        return ""

    tag_name = f"current_{scope}_memory"
    lines = [f"<{tag_name}>"]

    if has_blob:
        lines.append(blob.strip())

    if has_recent:
        lines.append(f"Recent context: {recent.strip()}")

    lines.append(f"</{tag_name}>")

    return "\n".join(lines)


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

    # Build JSON examples based on enabled scopes (user first, then agent - matches original)
    # Note: Double braces {{ }} are literal characters that appear in output (not format escapes)
    json_examples = []
    if "user" in enabled_scopes:
        json_examples.append(
            '    {{ "content": "User\'s email is john@example.com", "scope": "user" }}'
        )
    if "agent" in enabled_scopes:
        json_examples.append(
            '    {{ "content": "Project Apollo launch date is June 1st", "scope": "agent" }}'
        )

    # Determine enabled sections
    enabled_sections = set()
    if "user" in enabled_scopes:
        enabled_sections.add("user")
    if "agent" in enabled_scopes:
        enabled_sections.add("agent")

    return _process_template(
        template=FACT_EXTRACTION_TEMPLATE,
        values={
            "agent_persona": agent_persona,
            "max_words": max_words,
            "conversation_text": conversation_text,
            "json_examples": ",\n".join(json_examples) if json_examples else "",
        },
        enabled_sections=enabled_sections,
    )


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

    Empty sections are completely omitted to reduce token usage.
    Only includes memory context sections that have actual content.

    Args:
        conversation_text: The conversation to extract reflections from
        agent_persona: Agent persona/description
        memory_context: Dict with keys like "agent_blob", "agent_recent", etc.
        newly_formed_facts: Formatted string of newly formed facts (empty string if none)
        enabled_scopes: List of enabled scopes ["session", "user", "agent"]
        max_words: Maximum words per reflection

    Returns:
        Complete prompt string
    """
    # Build memory sections for enabled scopes (only if they have content)
    memory_parts = []

    if "agent" in enabled_scopes:
        agent_section = _format_memory_section(
            scope="agent",
            blob=memory_context.get("agent_blob"),
            recent=memory_context.get("agent_recent"),
        )
        if agent_section:
            memory_parts.append(agent_section)

    if "user" in enabled_scopes:
        user_section = _format_memory_section(
            scope="user",
            blob=memory_context.get("user_blob"),
            recent=memory_context.get("user_recent"),
        )
        if user_section:
            memory_parts.append(user_section)

    if "session" in enabled_scopes:
        session_section = _format_memory_section(
            scope="session",
            blob=memory_context.get("session_blob"),
            recent=memory_context.get("session_recent"),
        )
        if session_section:
            memory_parts.append(session_section)

    # If no memory sections, add placeholder note
    # Each section needs leading \n (for blank line before it) to match original
    # Also add trailing \n to separate from next section
    if memory_parts:
        memory_sections = "".join("\n" + part for part in memory_parts) + "\n"
    else:
        memory_sections = "\nNo existing memory context yet.\n\n"

    # Calculate hierarchy numbers based on enabled scopes (order: agent -> user -> session)
    hierarchy_num = 1
    agent_hierarchy_num = ""
    user_hierarchy_num = ""
    session_hierarchy_num = ""

    if "agent" in enabled_scopes:
        agent_hierarchy_num = str(hierarchy_num)
        hierarchy_num += 1
    if "user" in enabled_scopes:
        user_hierarchy_num = str(hierarchy_num)
        hierarchy_num += 1
    if "session" in enabled_scopes:
        session_hierarchy_num = str(hierarchy_num)

    # Build JSON fields based on enabled scopes
    # Note: Double braces {{ }} are literal characters that appear in output (not format escapes)
    json_fields = []
    if "agent" in enabled_scopes:
        json_fields.append('  "agent_reflections": [{{"content": "..."}}]')
    if "user" in enabled_scopes:
        json_fields.append('  "user_reflections": [{{"content": "..."}}]')
    if "session" in enabled_scopes:
        json_fields.append('  "session_reflections": [{{"content": "..."}}]')

    # Determine enabled sections
    enabled_sections = set()
    if "agent" in enabled_scopes:
        enabled_sections.add("agent")
    if "user" in enabled_scopes:
        enabled_sections.add("user")
    if "session" in enabled_scopes:
        enabled_sections.add("session")
    if newly_formed_facts and newly_formed_facts.strip():
        enabled_sections.add("has_facts")

    return _process_template(
        template=REFLECTION_EXTRACTION_TEMPLATE,
        values={
            "agent_persona": agent_persona,
            "memory_sections": memory_sections,
            "newly_formed_facts": newly_formed_facts,
            "conversation_text": conversation_text,
            "agent_hierarchy_num": agent_hierarchy_num,
            "user_hierarchy_num": user_hierarchy_num,
            "session_hierarchy_num": session_hierarchy_num,
            "max_words": max_words,
            "json_fields": ",\n".join(json_fields),
        },
        enabled_sections=enabled_sections,
    )
