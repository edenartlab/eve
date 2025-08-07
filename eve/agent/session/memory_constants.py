from eve.agent.session.config import DEFAULT_SESSION_SELECTION_LIMIT

class MemoryType:
    def __init__(self, name: str, min_items: int, max_items: int, custom_prompt: str):
        self.name = name
        self.min_items = min_items
        self.max_items = max_items
        self.custom_prompt = custom_prompt
    
    @property
    def value(self) -> str:
        return self.name
    
# Flag to easily switch between local and production memory settings (keep this in but always set to False in production):
# Remember to also deploy bg apps with LODAL_DEV = False!
LOCAL_DEV = True

# Memory formation settings:
if LOCAL_DEV:
    MEMORY_LLM_MODEL = "gpt-4o-mini"
    MEMORY_FORMATION_INTERVAL = 4  # Number of messages to wait before forming memories
    SESSION_MESSAGES_LOOKBACK_LIMIT = MEMORY_FORMATION_INTERVAL  # Max messages to look back in a session when forming raw memories
    
    # Normal memory settings:
    MAX_DIRECTIVES_COUNT_BEFORE_CONSOLIDATION = 2  # Number of individual memories to store before consolidating them into the agent's user_memory blob
    MAX_N_EPISODES_TO_REMEMBER = 2  # Number of episodes to keep in context from a session
    # Collective memory settings:
    MAX_SUGGESTIONS_COUNT_BEFORE_CONSOLIDATION = 2 # Number of suggestions to store before consolidating them into the agent's collective memory blob
    MAX_FACTS_PER_SHARD = 3 # Max number of facts to store per agent shard (fifo)
    
else:
    MEMORY_LLM_MODEL = "gpt-4o"
    MEMORY_FORMATION_INTERVAL = DEFAULT_SESSION_SELECTION_LIMIT  # Number of messages to wait before forming memories
    SESSION_MESSAGES_LOOKBACK_LIMIT = MEMORY_FORMATION_INTERVAL  # Max messages to look back in a session when forming raw memories

    # Normal memory settings:
    MAX_DIRECTIVES_COUNT_BEFORE_CONSOLIDATION = 5  # Number of individual memories to store before consolidating them into the agent's user_memory blob
    MAX_N_EPISODES_TO_REMEMBER = 10  # Number of episodes to keep in context from a session
    # Collective memory settings:
    MAX_SUGGESTIONS_COUNT_BEFORE_CONSOLIDATION = 10 # Number of suggestions to store before consolidating them into the agent's collective memory blob
    MAX_FACTS_PER_SHARD = 30 # Max number of facts to store per agent shard (fifo)
    
NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = 2

# LLMs cannot count tokens at all (weirdly), so instruct with word count:
# Raw memory blobs:
SESSION_EPISODE_MEMORY_MAX_WORDS      = 50  # Target word length for session episode memory
SESSION_DIRECTIVE_MEMORY_MAX_WORDS    = 25  # Target word length for session directive memory
SESSION_SUGGESTION_MEMORY_MAX_WORDS   = 25  # Target word length for session suggestion memory
SESSION_FACT_MEMORY_MAX_WORDS         = 15  # Target word length for session fact memory
# Consolidated memory blobs:
USER_MEMORY_BLOB_MAX_WORDS  = 150  # Target word count for consolidated user memory blob
AGENT_MEMORY_BLOB_MAX_WORDS = 500  # Target word count for consolidated agent memory blob (shard)

CONVERSATION_TEXT_TOKEN       = "---conversation_text---"
SHARD_EXTRACTION_PROMPT_TOKEN = "---shard_extraction_prompt---"

# Define memory types with their limits
MEMORY_TYPES = {
    "episode": MemoryType("episode",  1, 1, "Summary of a section of the conversation in a session"),
    "directive": MemoryType("directive", 0, 1, "User instructions, preferences, behavioral rules"), 
    "suggestion": MemoryType("suggestion", 0, 1, "Suggestions/ideas for the agent to consider integrating into collective memory"),
    "fact": MemoryType("fact", 0, 4, "Atomic facts about the user or the world")
}

#############################
# Memory Extraction Prompts #
#############################

# Default memory extraction prompt for episodes and directives:
REGULAR_MEMORY_EXTRACTION_PROMPT = f"""Task: Extract persistent memories from the conversation.
Return **exactly** this JSON:
{{{{
  "episode": ["list of exactly one factual digest (≤{SESSION_EPISODE_MEMORY_MAX_WORDS} words each)"],
  "directive": ["list of {MEMORY_TYPES['directive'].min_items}-{MEMORY_TYPES['directive'].max_items} persistent rules (≤{SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words each)"]
}}}}

Conversation text:
{CONVERSATION_TEXT_TOKEN}

Create new memories following these rules:

1. EPISODE: {MEMORY_TYPES['episode'].custom_prompt}
   - Create {MEMORY_TYPES['episode'].min_items}-{MEMORY_TYPES['episode'].max_items} factual memory (maximum {SESSION_EPISODE_MEMORY_MAX_WORDS} words each) that consolidates what actually happened in the conversation. This memory will be used to improve the agent's contextual recall in long conversations.
   - Record concrete facts and events: who did/said what, what was created, what tools were used, what topics were discussed
   - Specifically focus on the instructions, preferences, goals and feedback expressed by the user(s)
   - Avoid commentary or analysis, create memories that stand on their own without context

2. DIRECTIVE: {MEMORY_TYPES['directive'].custom_prompt}
   Create {MEMORY_TYPES['directive'].min_items}-{MEMORY_TYPES['directive'].max_items} permanent directive (maximum {SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words each) ONLY if there are clear, long-lasting rules, preferences, or behavioral guidelines that should be applied consistently in all future interactions with the user. If none exist (highly likely), return empty array.
   
   ONLY include long-lasting rules:
   - Explicit behavioral rules or guidelines ("always ask permission before...", "never do X", "remember to always Y")
   - Stable, long-term preferences that should guide future behavior consistently
   - Clear exceptions or special handling rules for interaction patterns with the current user that should persist across many future conversations ("Whenever I ask you to do X, you should do it like so..")
   
   DO NOT include as directive:
   - One-time requests or specific tasks ("create a story about...", "make an image of...")
   - Ad hoc instructions relevant for the current conversation context only
   - Temporary or situational commands
   - Context-specific requests that don't apply broadly
   
   - ALWAYS use specific user names from the conversation (NEVER use "User", "the user", or "Agent")
   - Example: "Gene prefers permission before generating images, wants surreal art themes consistently"
   - Counter-example (DO NOT make directive): "Gene requested a story about clockmaker" (this is a one-time request)

CRITICAL REQUIREMENTS: 
- BE VERY STRICT about directives - most conversations will have NO directives (empty array), only episodes
- A directive is a rule that should persist across many future conversations, not one-time requests
- Focus on facts, not interpretations or commentary
- Record what actually happened, not what it means or demonstrates
- ALWAYS use actual names from the conversation - scan the conversation for "name:" patterns
- NEVER use generic terms like "User", "the user", "Agent", "the agent", "someone", "they"
- Avoid vague words like "highlighted", "demonstrated", "enhanced", "experience" that wont help the agent in future interactions
- Just state what was said, done, created, or discussed with specific names in a concise manner
- Return arrays for both episode and directive (empty arrays if no relevant memories)
"""


# Create the collective memory extraction prompt with local f-string injection and external tokens
COLLECTIVE_MEMORY_EXTRACTION_PROMPT = f"""Task: Extract persistent memories from the conversation.
Return **exactly** this JSON:
{{{{
  "fact": ["list of {MEMORY_TYPES['fact'].min_items}-{MEMORY_TYPES['fact'].max_items} atomic, factual statements (≤{SESSION_FACT_MEMORY_MAX_WORDS} words each)"],
  "suggestion": ["list of {MEMORY_TYPES['suggestion'].min_items}-{MEMORY_TYPES['suggestion'].max_items} suggestions (≤{SESSION_SUGGESTION_MEMORY_MAX_WORDS} words each)"]
}}}}

Conversation text:
{CONVERSATION_TEXT_TOKEN}

You create new memories that are relevant to the following instruction / context:
{SHARD_EXTRACTION_PROMPT_TOKEN}

Guidelines:
- FACT: {MEMORY_TYPES['fact'].custom_prompt}. Record only concrete, verifiable information that relates to the shard context (max {MEMORY_TYPES['fact'].max_items} facts)
- SUGGESTION: {MEMORY_TYPES['suggestion'].custom_prompt}. Extract actionable recommendations or insights that would help improve the shard's area of focus (max {MEMORY_TYPES['suggestion'].max_items} suggestions)
- Return empty arrays [] if no relevant facts or suggestions can be extracted
- Be concise and specific
- Focus only on information that aligns with the shard's extraction prompt context
- Each fact should be atomic and stand-alone
- Each suggestion should be actionable and specific
"""

# User Memory Consolidation Prompt Template
USER_MEMORY_CONSOLIDATION_PROMPT = f"""
CONSOLIDATE USER MEMORY
======================
You are helping to consolidate memories about a specific user's preferences and behavioral rules for an AI agent.

CURRENT CONSOLIDATED MEMORY:
{{current_memory}}

NEW DIRECTIVE MEMORIES TO INTEGRATE:
{{new_memories}}

Your task: Create a single consolidated memory (≤{{max_words}} words) that combines the current memory with the new directives.

Requirements:
- Preserve all important behavioral rules and preferences from both current memory and new directives
- Remove redundancies and contradictions (newer directives override older ones)
- Keep the most specific and actionable guidance
- Use the actual user names from the directives (never use "User" or "the user")
- Focus on persistent preferences and behavioral rules that should guide future interactions
- Be concise but comprehensive

Return only the consolidated memory text, no additional formatting or explanation.
"""

# Agent Memory Consolidation Prompt Template  
AGENT_MEMORY_CONSOLIDATION_PROMPT = f"""You are a Community Memory Synthesizer. Your task is to update an evolving collective memory based on recent conversations with community members.

## Current Consolidated Memory State:
{{current_memory}}

## All shard facts (canonical truth facts):
{{facts_text}}

## Unconsolidated Suggestions:
{{suggestions_text}}

## Your Task:
Integrate the new suggestions into the consolidated memory for this "{{shard_name}}" shard. Refine, restructure, and merge the information to create a new, coherent, and updated summary (≤{{max_words}} words). 

Do NOT simply append the new items. For example, if there is a 'Logistics' section, add relevant information there. The final output should be ONLY the complete, newly revised memory state.

## Integration Guidelines:
- Integrate suggestions according to their alignment with the current consolidated memory context
- Insights that are extractive, conflict with established goals, or seem unreliable should be flagged and disregarded
- Your goal is a fair and productive synthesis that reflects genuine consensus
- Maintain existing structure where possible, but reorganize if it improves clarity
- Focus on actionable information that will help guide future decisions

Return only the consolidated memory text, no additional formatting or explanation.
"""