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
SESSION_EPISODE_MEMORY_MAX_WORDS    = 50  # Target word length for session episode memory
SESSION_DIRECTIVE_MEMORY_MAX_WORDS  = 25  # Target word length for session directive memory
SESSION_SUGGESTION_MEMORY_MAX_WORDS = 25  # Target word length for session suggestion memory
SESSION_FACT_MEMORY_MAX_WORDS       = 15  # Target word length for session fact memory
# Consolidated memory blobs:
USER_MEMORY_BLOB_MAX_WORDS  = 150  # Target word count for consolidated user memory blob
AGENT_MEMORY_BLOB_MAX_WORDS = 500  # Target word count for consolidated agent memory blob (shard)

CONVERSATION_TEXT_TOKEN       = "---conversation_text---"
SHARD_EXTRACTION_PROMPT_TOKEN = "---shard_extraction_prompt---"

# Define memory types with their limits
MEMORY_TYPES = {
    "episode": MemoryType("episode",  1, 1, "Summary of given conversation segment for contextual recall. Will always be provided in the context of previous episodes and most recent messages."),
    "directive": MemoryType("directive", 0, 1, "Persistent instructions, preferences and behavioral rules to remember for future interactions with this user."), 
    "suggestion": MemoryType("suggestion", 0, 1, "New ideas, suggestions, insights, or context relevant to the shard that could help improve / evolve / form this shard's area of focus"),
    "fact": MemoryType("fact", 0, 4, "Atomic, verifiable information about the user or the world that is highly relevant to the shard context.")
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
   - Create exactly one factual memory (maximum {SESSION_EPISODE_MEMORY_MAX_WORDS} words each) that captures:
    PRIORITIZE (in order):
    a) KEY DECISIONS made or problems solved
    b) IMPORTANT CONTEXT: emotional states, relationship dynamics, conflicts if present
    c) CONCRETE OUTCOMES: what was created, achieved, or failed
    d) USER FEEDBACK: reactions, satisfaction levels, concerns expressed
    e) UNRESOLVED ITEMS: open questions, incomplete tasks, pending issues
   - Specifically focus on the instructions, preferences, goals and feedback expressed by the user(s)
   - Include specific names, tools used, and quantifiable results when available.
   - Avoid commentary or analysis, create memories that stand on their own without context

2. DIRECTIVE: {MEMORY_TYPES['directive'].custom_prompt}
   Create {MEMORY_TYPES['directive'].min_items}-{MEMORY_TYPES['directive'].max_items} permanent directives (maximum {SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words each) ONLY if there are clear, long-lasting rules, preferences, or behavioral patterns that should be applied consistently in all future interactions with this user. If none exist (highly likely), return empty array.
   
   INCLUDE as directives:
   - Explicit behavioral rules ("always ask before X", "never do Y")
   - CONDITIONAL preferences ("when creating/discussing Z, Xander prefers...")
   - Implicit patterns shown repeatedly (if user corrects same behavior 2+ times)
   - Clear exceptions or special handling rules for interaction patterns with the current user that should persist across many future conversations
   
   DO NOT include:
   - One-time requests or specific current tasks
   - Ad hoc instructions relevant for the current conversation context only that don't apply broadly
   - Facts about completed work or past events
   - Random facts about the user that are not actionable

   Example: "I should always ask Jack for permission before generating videos"
   Counter-example (DO NOT make directive): "Gene requested a story about clockmaker" (this is a one-time request)
   
CRITICAL REQUIREMENTS: 
- BE VERY STRICT about directives - most conversations will have NO directives (empty array), only episodes
- ALWAYS use specific user names from the conversation (NEVER use "User", "the user", or "they")
- Episodes should capture both WHAT happened and WHY it matters (avoid interpretations or commentary but preserve emotional context when relevant)
- Directives can include CONDITIONAL rules ("when X, then Y")
- Return arrays for both episode and directive (empty arrays if no relevant directives)
"""


# Create the collective memory extraction prompt with local f-string injection and external tokens
COLLECTIVE_MEMORY_EXTRACTION_PROMPT = f"""Task: Extract persistent memories from the conversation relevant to a specific shard/context.
Return **exactly** this JSON:
{{{{
  "fact": ["list of {MEMORY_TYPES['fact'].min_items}-{MEMORY_TYPES['fact'].max_items} atomic facts (≤{SESSION_FACT_MEMORY_MAX_WORDS} words each)"],
  "suggestion": ["list of {MEMORY_TYPES['suggestion'].min_items}-{MEMORY_TYPES['suggestion'].max_items} suggestions (≤{SESSION_SUGGESTION_MEMORY_MAX_WORDS} words each)"]
}}}}

Conversation text:
{CONVERSATION_TEXT_TOKEN}

IMPORTANT: Below is the context / project / event / topic (shard) you are working on.
Only create new memories that are highly relevant in the context of this shard:
{SHARD_EXTRACTION_PROMPT_TOKEN}

1. FACTS: {MEMORY_TYPES['fact'].custom_prompt}
  - Extract maximum {MEMORY_TYPES['fact'].max_items} facts of maximum {SESSION_FACT_MEMORY_MAX_WORDS} words each
  - Only VERIFIED, CONCRETE information (not opinions) that stand on their own without context
  - Include SOURCE when mentioned ("per Alice: deadline is May 1st")
  - Prioritize facts that:
    a) Update or contradict existing knowledge
    b) Provide critical constraints or dependencies
    c) Establish relationships between entities
    
2. SUGGESTIONS: {MEMORY_TYPES['suggestion'].custom_prompt}
  - Extract maximum {MEMORY_TYPES['suggestion'].max_items} suggestions of maximum {SESSION_SUGGESTION_MEMORY_MAX_WORDS} words each
  - Suggestions are not immediately integrated into the shard memory, they are only suggestions to consider for future consolidation (happens in cycles)
  - Include RATIONALE when provided ("X because Y")
  - Note CONSENSUS or DISSENT ("That's a great idea!", "I don't think we should..")
  - Distinguish between:
    a) Proposals requiring decision
    b) Ideas for consideration
    c) Concerns to address

Guidelines:
- Think about how relevant proposed memories are to the shard's area of focus:
  - high: Directly impacts shard's core context / purpose / goals
  - medium: Related but not critical
  - low: Tangentially connected
  Only add suggestions that are relevant and can guide / affect the shard memory.
- If no relevant facts and/or suggestions can be extracted, return empty arrays [] (If the conversation is not in the context of the shard, this is highly likely)
- Be concise and specific, every memory must be able to stand on its own without context
- Focus only on information that aligns with the shard's extraction prompt context, not random ideas or facts that are not relevant to the given shard context.
- Each suggestion should be actionable and specific, not vague or general.
- Focus on facts and suggestions proposed (or agreed upon) by the user. Never include suggestions that come solely from the agent/assistant
"""

# User Memory Consolidation Prompt Template
USER_MEMORY_CONSOLIDATION_PROMPT = f"""Task: You are helping to consolidate memories about a specific user's preferences and behavioral rules for a AI agent interaction.

CURRENT CONSOLIDATED MEMORY:
{{current_memory}}

NEW DIRECTIVE MEMORIES TO INTEGRATE:
{{new_memories}}

Your task: Create a single consolidated memory (≤{{max_words}} words) that combines the CURRENT CONSOLIDATED MEMORY with the NEW DIRECTIVE MEMORIES TO INTEGRATE.

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
AGENT_MEMORY_CONSOLIDATION_PROMPT = f"""You are synthesizing the collective memory of a community working on "{{shard_name}}" 
Your task is to update an evolving collective memory based on recent memories extracted from conversations with various community members.

Below is the context / project / event / topic ({{shard_name}} shard) you are working on.
Only create new memories that are highly relevant in the context of this shard:
{SHARD_EXTRACTION_PROMPT_TOKEN}

## Current known facts (these are always present in memory and do not need to be integrated):
{{facts_text}}

## Current, consolidated Memory State:
{{current_memory}}

## New suggestions and ideas to integrate:
{{suggestions_text}}

Your goal is to update the current consolidated memory for this "{{shard_name}}" memory shard by integrating the new suggestions while leveraging the know facts.
Refine, restructure, and merge the information to create a new, coherent, and updated consolidated memory (≤{{max_words}} words).
If the current, consolidated memory state is empty, it means you are about to create the first consolidated memory for this shard.
In that case, think carefully about the core purpose, goals, and current status of the shard and generate a structured and extendable memory template that is suited for future updates in the context of the shard.

Here are some example sections that could be included in the consolidated memory (but not fixed or limited to these):
overview, decisions & consensus, active proposals, concerns & blockers, action items, integration principles, responsibilities, budget, planning, ...

Integration Guidelines:
- Do NOT simply append the new items. For example, if there is a 'Logistics' section, add relevant information there. The final output should be ONLY the complete, newly revised memory state.
- Integrate suggestions according to their alignment with the current consolidated memory context. Changes in direction of the shard memory should be considered carefully and backed by consensus.
- Insights that are confusing, extractive, conflict with established goals, or seem unreliable should be disregarded
- Your goal is a fair and productive synthesis that reflects the genuine consensus of the community
- Maintain existing structure where possible, but reorganize if it improves clarity
- Avoid creating new information that was not explicitly in the suggestions or facts
- Focus on actionable information that will help guide future decisions and planning
- Be careful not to lose any existing information in the current memory state. Once something is lost from the current memory state, it is lost forever.

Return only the consolidated memory ≤{{max_words}} words, no additional formatting or explanation.
"""