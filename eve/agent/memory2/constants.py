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
LOCAL_DEV = True  # Set to False for production


# -----------------------------------------------------------------------------
# LLM Model Configuration
# -----------------------------------------------------------------------------
if LOCAL_DEV:
    MEMORY_LLM_MODEL_FAST = "gpt-5-mini"
    MEMORY_LLM_MODEL_SLOW = "gpt-5-mini"
else:
    MEMORY_LLM_MODEL_FAST = "gpt-5-mini"
    MEMORY_LLM_MODEL_SLOW = "gpt-5.2"


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
    "session": 300,  # Rolling summary of session events and status
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
# Prompt Templates
# -----------------------------------------------------------------------------

# Fact extraction prompt (LLM Call 1 - no memory context needed)
FACT_EXTRACTION_PROMPT = """You are extracting factual information from a conversation for a RAG (retrieval) system.

## HOW FACTS ARE USED
Facts are stored in a searchable database and retrieved ONLY when the agent explicitly searches for them.
The agent will query facts using natural language searches like:
- "What is Alice's birthday?"
- "What's the project budget?"
- "Who is the team lead?"

Facts are NOT always visible to the agent - they must be specifically queried to be retrieved.
This means facts should be SEARCHABLE ANSWERS to specific questions.

<conversation>
{conversation_text}
</conversation>

## WHAT MAKES A GOOD FACT
Facts should be specific, queryable pieces of information that answer concrete questions:

EXTRACT as facts:
- Specific dates, numbers, names ("Alice's birthday is March 15th", "Budget is $50,000")
- Relationships and roles ("Bob is the project lead", "Carol reports to Dave")
- Concrete identifiers ("Project codename is Phoenix", "API key prefix is sk-prod")
- Permanent attributes ("Alice is a software engineer", "Company HQ is in Austin")

DO NOT extract as facts:
- Behavioral preferences ("Alice likes concise responses") → use reflections instead
- Ongoing states or context ("We're debugging the login issue") → use reflections
- Opinions or evolving views ("Bob thinks we should use React") → use reflections
- Vague information without specific searchable content
- Anything that wouldn't make sense as an answer to a specific question

## SCOPE
- "user": Information about this specific user (their birthday, job, personal details)
- "agent": Information relevant to all users (project details, team members, deadlines)

Return JSON:
{{
  "facts": [
    {{"content": "...", "scope": ["user"]}},
    {{"content": "...", "scope": ["agent"]}}
  ]
}}

CRITICAL: Ask yourself - "Would the agent search for this?" and "Is this a specific answer to a concrete question?"
If the answer is no, don't extract it. Most conversations have FEW or NO facts.

Guidelines:
- Facts must be self-contained answers that make sense without additional context
- Maximum {max_words} words per fact
- ALWAYS use specific names (NEVER "User", "the user", or "they")
- Return empty list if no facts found (this is common and expected)
"""


# Reflection extraction prompt (LLM Call 2 - needs memory context + facts)
REFLECTION_EXTRACTION_PROMPT = """You are updating an agent's always-in-context memory based on a conversation.

## HOW REFLECTIONS ARE USED
Reflections are ALWAYS visible to the agent in every conversation - they shape how the agent behaves.
Because reflections consume tokens on EVERY message, storage is LIMITED and reflections get consolidated.
Use reflections for information that should CONTINUOUSLY influence the agent's behavior.

Unlike facts (which are searched for specific answers), reflections answer:
- "How should I interact with this user?"
- "What's the current context/state I need to be aware of?"
- "What behavioral rules should I follow?"

## CURRENT MEMORY STATE

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
{newly_formed_facts}
</newly_formed_facts>

<conversation>
{conversation_text}
</conversation>

## EXTRACTION RULES

Extract reflections in HIERARCHICAL ORDER - each level should only contain information NOT already captured at a higher level, in existing memory, or in newly_formed_facts.

### 1. AGENT REFLECTIONS (broadest scope - affects ALL conversations)
What should continuously influence the agent across all users?

GOOD for agent reflections:
- Ongoing initiatives ("Currently prioritizing the memory system redesign")
- Operational learnings ("When generating images, always confirm style first")
- Universal context ("Eden is pivoting toward enterprise customers")

NOT for agent reflections (use facts instead):
- Specific searchable data ("Budget is $50k", "Launch date is March 1st")
- Static information that doesn't affect behavior

### 2. USER REFLECTIONS (user scope - affects all conversations with THIS user)
What should continuously influence how the agent interacts with this specific user?

GOOD for user reflections:
- Interaction preferences ("Alice prefers bullet points over paragraphs")
- Behavioral rules ("Always ask Bob for confirmation before running tools")
- Working context ("Carol is exploring AI for her VJ performances")

NOT for user reflections:
- One-time requests ("make this image larger")
- Searchable personal data ("Alice's birthday is March 15") → use facts
- Session-specific tasks → use session reflections

### 3. SESSION REFLECTIONS (narrowest scope - THIS session only)
What context is needed for continuity in this specific conversation?

GOOD for session reflections:
- Current task state ("Debugging auth - tried solutions A and B, testing C")
- Active creative context ("Story features a robot named Max in Tokyo")
- Session instructions ("User wants all responses in bullet points today")

NOT for session reflections:
- Permanent preferences → use user reflections
- Searchable details → use facts

Return JSON:
{{
  "agent_reflections": [{{"content": "..."}}],
  "user_reflections": [{{"content": "..."}}],
  "session_reflections": [{{"content": "..."}}]
}}

CRITICAL: Reflections should affect BEHAVIOR, not store SEARCHABLE DATA.
Ask: "Does the agent need to know this on every message, or only when specifically asked?"
- Every message → reflection
- Only when asked → fact (already extracted in newly_formed_facts)

Maximum {max_words} words per reflection. Use specific names (never "User" or "the user").
Return empty arrays when nothing meaningful to extract (this is common).
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


# Consolidation prompt template (used for all scope levels)
CONSOLIDATION_PROMPT = """You are consolidating {scope_type} memory reflections.

<current_consolidated_memory>
{existing_blob}
</current_consolidated_memory>

<new_reflections>
{new_reflections}
</new_reflections>

Merge the new reflections into the existing memory, creating an updated consolidated memory.

{scope_specific_instructions}

Integration Guidelines:
- Do NOT simply append new items - INTEGRATE them into the existing structure
- Preserve existing sections UNCHANGED if new reflections don't affect them (copy verbatim)
- Resolve contradictions: newer information takes precedence
- When consensus is unclear, preserve both perspectives ("Some prefer X while others prefer Y")
- Be careful not to lose ANY existing information - once lost, it's gone forever
- After consolidation, all new_reflections will be discarded, so ensure all valuable info is integrated
- Do NOT invent or hallucinate any new information not explicitly in new_reflections
- Maximum {word_limit} words

CRITICAL:
- Sections of existing memory that don't need updates should be copied UNCHANGED
- The goal is a coherent synthesis, not a list of appended items
- Focus on actionable information that will help guide future interactions

Return ONLY the updated consolidated memory text (no additional formatting or explanation).
"""


# Scope-specific consolidation instructions
CONSOLIDATION_INSTRUCTIONS = {
    "agent": """Focus on agent-wide memory:
- Ongoing projects, plans, roadmaps shared across all users
- Domain knowledge and insights that apply universally
- Learnings about how the agent should operate
- Agent character/persona evolution

Structure suggestions (adapt as needed):
- Current priorities and active initiatives
- Decisions and consensus reached
- Open proposals requiring input
- Domain insights and learnings

Discard ONLY: spam, completely off-topic content, or factually impossible claims.
For conflicting viewpoints: note as "disputed" rather than discarding.""",

    "user": """Focus on user-specific memory:
- Persistent behavioral rules and preferences for this user
- Communication style and expertise level
- Personal context that affects all future interactions
- How the agent should interact with this specific user

Keep concise - focus on what affects agent behavior.
Remove redundancies but preserve all actionable rules.
Store actual username when available (avoid "User" or "the user").""",

    "session": """Focus on session-specific memory:
- Rolling summary of what has happened
- Current task status and progress
- Active creative context (characters, storylines, references)
- Temporary instructions for this session only

This is a working memory - prioritize recent context over older details.
Keep as a coherent narrative summary, not a list of events.
Focus on information needed to maintain conversation continuity.""",
}
