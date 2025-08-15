from eve.agent.session.config import DEFAULT_SESSION_SELECTION_LIMIT

class MemoryType:
    def __init__(self, name: str, min_items: int, max_items: int, custom_prompt: str):
        self.name = name
        self.min_items = min_items
        self.max_items = max_items
        self.custom_prompt = custom_prompt
    
# Flag to easily switch between local and production memory settings (keep this in but always set to False in production):
# Remember to also deploy bg apps with LODAL_DEV = False!
LOCAL_DEV = False

# Memory formation settings:
if LOCAL_DEV:
    MEMORY_LLM_MODEL_FAST = "gpt-5-mini-2025-08-07"
    #MEMORY_LLM_MODEL_FAST = "gpt-5-2025-08-07"
    MEMORY_LLM_MODEL_SLOW = "gpt-5-2025-08-07"

    MEMORY_FORMATION_MSG_INTERVAL   = 4   # Number of messages to wait before forming memories
    MEMORY_FORMATION_TOKEN_INTERVAL = 200 # Number of tokens to wait before forming memories

    # Normal memory settings:
    MAX_USER_MEMORIES_BEFORE_CONSOLIDATION = 3  # Number of individual memories to store before consolidating them into the agent's user_memory blob
    MAX_N_EPISODES_TO_REMEMBER = 2  # Number of episodes to keep in context from a session
    # Collective memory settings:
    MAX_AGENT_MEMORIES_BEFORE_CONSOLIDATION = 3 # Number of suggestions to store before consolidating them into the agent's collective memory blob
    MAX_FACTS_PER_SHARD = 3 # Max number of facts to store per agent shard (fifo)
    
else:
    MEMORY_LLM_MODEL_FAST = "gpt-5-mini-2025-08-07"
    #MEMORY_LLM_MODEL_SLOW = "gpt-5-2025-08-07"
    MEMORY_LLM_MODEL_SLOW = "gpt-5-mini-2025-08-07"
    MEMORY_FORMATION_MSG_INTERVAL   = 10    # Number of messages to wait before forming memories
    MEMORY_FORMATION_TOKEN_INTERVAL = 1000  # Number of tokens to wait before forming memories

    # Normal memory settings:
    MAX_USER_MEMORIES_BEFORE_CONSOLIDATION = 4  # Number of individual memories to store before consolidating them into the agent's user_memory blob
    MAX_N_EPISODES_TO_REMEMBER = 8  # Number of episodes to keep in context from a session
    # Collective memory settings:
    MAX_AGENT_MEMORIES_BEFORE_CONSOLIDATION = 10 # Number of suggestions to store before consolidating them into the agent's collective memory blob
    MAX_FACTS_PER_SHARD = 50 # Max number of facts to store per agent shard (fifo)
    
# Configuration for cold session processing
CONSIDER_COLD_AFTER_MINUTES = 8  # Consider a session cold if no activity for this many minutes
CLEANUP_COLD_SESSIONS_EVERY_MINUTES = 8  # Run the background task every N minutes

SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES = 4
NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES = 4
AGENT_TOKEN_MULTIPLIER = 0.25  # Multiplier to downscale agent/assistant message importance for token interval trigger

# LLMs cannot count tokens at all (weirdly), so instruct with word count:
# Raw memory blobs:
SESSION_EPISODE_MEMORY_MAX_WORDS    = 50  # Target word length for session episode memory
SESSION_DIRECTIVE_MEMORY_MAX_WORDS  = 20  # Target word length for session directive memory
SESSION_SUGGESTION_MEMORY_MAX_WORDS = 30  # Target word length for session suggestion memory
SESSION_FACT_MEMORY_MAX_WORDS       = 20  # Target word length for session fact memory
# Consolidated memory blobs:
USER_MEMORY_BLOB_MAX_WORDS  = 200  # Target word count for consolidated user memory blob
AGENT_MEMORY_BLOB_MAX_WORDS = 500  # Target word count for consolidated agent memory blob (shard)

# Define different memory types and their extraction limits:
MEMORY_TYPES = {
    "episode":    MemoryType("episode",    1, 1, "Summary of given conversation segment for contextual recall. Will always be provided in the context of previous episodes and most recent messages."),
    "directive":  MemoryType("directive",  0, 3, "Persistent instructions, preferences and behavioral rules to remember for future interactions with this user."), 
    "suggestion": MemoryType("suggestion", 0, 5, "New ideas, suggestions, insights, or context relevant to the shard that could help improve / evolve / form this shard's area of focus"),
    "fact":       MemoryType("fact",       0, 2, "Atomic, verifiable information about the user or the world that is relevant to the shard context and should be kept in memory FOREVER.")
}

#############################
# Memory Extraction Prompts #
#############################

CONVERSATION_TEXT_TOKEN         = "-&&-conversation_text-&&-"
SHARD_EXTRACTION_PROMPT_TOKEN   = "-&&-shard_extraction_prompt-&&-"
FULLY_FORMED_AGENT_MEMORY_TOKEN = "-&&-fully_formed_agent_memory-&&-"

# Default memory extraction prompt for episodes and directives:
REGULAR_MEMORY_EXTRACTION_PROMPT = f"""Task: Extract persistent memories from the provided conversation following these rules:

1. EPISODE: {MEMORY_TYPES['episode'].custom_prompt}
  - Create exactly one episodic memory (maximum {SESSION_EPISODE_MEMORY_MAX_WORDS} words each) that captures (in order of importance):
    a) WHAT HAPPENED: Key plans, decisions, problems solved, or actions taken
    b) WHY IT MATTERS: User goals, feedback, and emotional context if significant
    c) WHAT'S NEXT: Unresolved items or explicit next steps mentioned
  - Specifically focus on the instructions, preferences, goals and feedback expressed by the user(s)
  - Include specific names, tools used, and quantifiable results when available.
  - Avoid commentary or analysis, create memories that stand on their own without context

2. DIRECTIVE: {MEMORY_TYPES['directive'].custom_prompt}
  Create {MEMORY_TYPES['directive'].min_items}-{MEMORY_TYPES['directive'].max_items} permanent directives (maximum {SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words each) ONLY if there are clear, long-lasting rules, preferences, or behavioral patterns that should be applied consistently in all future interactions with this user. If none exist (highly likely), return empty array.
   
  INCLUDE as directives:
  - Explicit behavioral rules ("always ask before X", "never do Y")
  - CONDITIONAL preferences ("when creating/discussing Z, Xander prefers...")
  - Implicit patterns shown repeatedly (eg if user corrects same behavior 2+ times)
  - Clear exceptions or special handling rules for interaction patterns with the current user that should persist across many future conversations
   
  DO NOT include:
  - One-time requests or specific current tasks
  - Ad hoc instructions relevant for the current conversation context only that don't apply broadly
  - Facts about completed work or past events
  - Random facts about the user that are not actionable

  Good examples: 
  - "Always ask Jack for permission before generating videos"
  - "Before generating images always check what aspect ratio the user prefers"
  Bad examples (DO NOT make these directives):
  - "Gene requested a story about clockmaker" (one-time request)
  - "The deadline is next Friday" (temporal fact, not behavioral rule)
  - "User works at Google" (fact about user, not actionable rule)
   
CRITICAL REQUIREMENTS: 
- BE VERY STRICT about directives - most conversations will have NO directives (empty array), only episodes
- ALWAYS use specific user names from the conversation (NEVER use "User", "the user", or "they")
- Episodes should capture both WHAT happened and WHY it matters (avoid interpretations or commentary but preserve emotional context when relevant)
- Directives can include CONDITIONAL rules ("when X, then Y")
- Return arrays for both episode and directive (empty arrays if no relevant directives)

Now carefully read the conversation text and extract the episodes and directives:
<conversation_text>
{CONVERSATION_TEXT_TOKEN}
</conversation_text> 

Return **exactly** this JSON:
{{{{
  "episode": ["list of exactly one factual digest (≤{SESSION_EPISODE_MEMORY_MAX_WORDS} words each)"],
  "directive": ["list of {MEMORY_TYPES['directive'].min_items}-{MEMORY_TYPES['directive'].max_items} persistent rules (≤{SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words each)"]
}}}}
"""


########################################################


# Create the collective memory extraction prompt with local f-string injection and external tokens
AGENT_MEMORY_EXTRACTION_PROMPT = f"""Task: Extract persistent memories from the conversation relevant to a specific shard/context.

IMPORTANT: Below is the context / project / event / topic (shard) you are working on.
<shard_context>
{FULLY_FORMED_AGENT_MEMORY_TOKEN}
</shard_context>

Your goal is to extract facts and suggestions that are relevant to the shard's context according to the following guidelines:

1. FACTS: {MEMORY_TYPES['fact'].custom_prompt}
  - Extract 0 to {MEMORY_TYPES['fact'].max_items} facts (maximum {SESSION_FACT_MEMORY_MAX_WORDS} words each). Typically, you will extract much less than {MEMORY_TYPES['fact'].max_items} #facts.
  - Each fact must be UNIQUE, ATOMIC and VERIFIED - one specific piece of information coming from the user(s) that will never change.
  - Include SOURCE when provided ("Alice: deadline is May 1st" or "Bob: the max budget is $1000")
  - Facts must be self-contained and understandable without conversation context
  - Only include facts that are relevant to the shard's context and were actually spoken by the user(s) themselves.
  - Facts are permanently stored in the shard memory and so they must be true in future phases of the project / context, not just true right this moment. A current rule / preference that could evolve over time should be stored as a suggestion.
  - Prioritize facts that:
    a) Create or complement existing knowledge
    b) Provide critical constraints or dependencies
    c) Establish relationships between entities
    
2. SUGGESTIONS: {MEMORY_TYPES['suggestion'].custom_prompt}
  - Extract maximum {MEMORY_TYPES['suggestion'].max_items} suggestions of maximum {SESSION_SUGGESTION_MEMORY_MAX_WORDS} words each
  - Suggestions are not immediately integrated into the shard memory, they are only suggestions to consider for future consolidation (happens in cycles)
  - Include rationale when provided ("X because Y") and note down when further consensus is needed, also remember when people disagree with existing ideas / suggestions and try to find a compromise.
  - Distinguish for example between:
    a) New proposals or ideas requiring further discussion
    b) Ideas for collective consideration
    c) Concerns to address

Guidelines:
- Think about how relevant the proposed memories are to the shard's area of focus:
  - high: Directly impacts shard's core context / purpose / goals
  - medium: Related but not critical
  - low: Tangentially connected
  Only add suggestions that are relevant and can guide / affect the shard memory.
- If no relevant facts and/or suggestions can be extracted, return empty arrays [] (If the conversation is not in the context of the shard, this is highly likely)
- Be concise and specific, every memory must be able to stand on its own without context
- ALWAYS use specific user names from the conversation (NEVER use "User", "the user", or "they")
- Focus only on information that aligns with the shard's extraction prompt context, not random ideas or facts that are not relevant to the given shard context.
- Each suggestion should be actionable, specific or generally contribute to the shard's context. Avoid vague or general commentary.
- Focus on facts and suggestions proposed (or agreed upon) by the user itself. NEVER include facts or suggestions that come solely from the agent/assistant's messages / interpretation unless the user explicitly confirms them as good.
- IMPORTANT: do not extract any new facts or suggestions that are already part of the shard_context.

Now carefully read the conversation text and extract the facts and suggestions:
<conversation_text>
{CONVERSATION_TEXT_TOKEN}
</conversation_text>

Return **exactly** this JSON:
{{{{
  "fact": ["list of {MEMORY_TYPES['fact'].min_items}-{MEMORY_TYPES['fact'].max_items} atomic facts (≤{SESSION_FACT_MEMORY_MAX_WORDS} words each)"],
  "suggestion": ["list of {MEMORY_TYPES['suggestion'].min_items}-{MEMORY_TYPES['suggestion'].max_items} suggestions (≤{SESSION_SUGGESTION_MEMORY_MAX_WORDS} words each)"]
}}}}
"""


########################################################

# User Memory Consolidation Prompt Template
USER_MEMORY_CONSOLIDATION_PROMPT = f"""Task: You are helping to consolidate all memories regarding one specific user's preferences and behavioral rules for all your future interactions with this user.

CURRENT CONSOLIDATED MEMORY:
{{current_memory}}

NEW DIRECTIVE MEMORIES TO INTEGRATE:
{{new_memories}}

Your task: Create a single consolidated memory (≤{{max_words}} words) that combines the CURRENT CONSOLIDATED MEMORY with the NEW DIRECTIVE MEMORIES TO INTEGRATE.

Requirements:
- Preserve all important behavioral rules and preferences from both current memory and new directives
- Remove redundancies and contradictions (newer directives override older ones, although directive age should not be integrated)
- Keep the most specific and actionable guidance
- Use the actual user name from the directives (never use "User" or "the user"). There is only one single user in this context.
- Focus on persistent preferences and behavioral rules that should guide future interactions
- Be concise but comprehensive

Return only the consolidated memory text, no additional formatting or explanation.
"""

########################################################

# Agent Memory Consolidation Prompt Template  
AGENT_MEMORY_CONSOLIDATION_PROMPT = f"""You are synthesizing the collective memory of a community working on "{{shard_name}}" 
Your task is to update an evolving collective memory based on new memories extracted from recent conversations with various community members.

Below is the context / project / event / topic ({{shard_name}} shard) you are working on.
Only create new memories that are highly relevant in the context of this shard:
<shard_context>
{SHARD_EXTRACTION_PROMPT_TOKEN}
</shard_context>

## Current known facts (Provided for context: these facts will always be automaticallypresent in memory and do NOT need to be integrated!):
{{facts_text}}

## Current, consolidated MEMORY STATE (The thing to update):
{{current_memory}}

## NEW SUGGESTIONS and ideas to integrate (The information to integrate):
{{suggestions_text}}

Your goal is to update the current consolidated MEMORY STATE for this "{{shard_name}}" memory shard by integrating the new suggestions while leveraging the know facts.
Refine, restructure, and merge the information to create a new, coherent, and updated consolidated memory (≤{{max_words}} words).

If the current, consolidated MEMORY STATE is EMPTY:
 - this means you are about to create the first consolidated memory for this shard, add "VERSION: 1" (integer) at the top of the MEMORY STATE
 - In that case, think carefully about the core purpose, goals, and context of the shard and generate a structured and extendable memory template that is suited for future updates.
 - Typically, there is little initial information available, so don't start filling up the MEMORY STATE with invented information. EVERYTHING you store must be based on collective user input, not the your free-form interpretation / generation! Don't rush to fill this up, more NEW SUGGESTIONS will come in the future.

Here are just a few example sections that could be included in the MEMORY STATE. These are just examples, you can include any other sections that are relevant to the shard's context or leave the MEMORY STATE super basic if there is not a lot of information available yet.
overview, tasks, decisions & consensus, active proposals, concerns & blockers, action items, integration principles, responsibilities, budget, planning, ...

Integration Guidelines:
- always increment the VERSION by 1 (integer) when you update the MEMORY STATE
- Do NOT simply append the new items but integrate. The final output should be ONLY the complete, newly revised MEMORY STATE.
- Integrate suggestions according to their alignment with the current consolidated memory context. Changes in direction of the shard memory should be considered carefully and backed by consensus.
- Discard suggestions only if they are: spam, completely off-topic, or factually impossible
- Integrate conflicting viewpoints by noting them as "disputed" or "minority view" rather than discarding
- When consensus is unclear, preserve both perspectives (e.g., "Some members propose X while others prefer Y")
- Your goal is a fair and productive synthesis that reflects the genuine consensus of the collective input
- Maintain existing structure where possible, but reorganize if it improves clarity / conciseness
- Do not generate / hallucinate any new information that was not explicitly in the suggestions or facts. All of the updates must be driven by the collective input, not the agent's interpretation.
- Focus on actionable information that will help guide future decisions and planning
- Be careful not to lose any existing information in the current MEMORY STATE. Once something is lost from the current MEMORY STATE, it is lost forever. 
- After this integration step, all NEW SUGGESTIONS will be deleted forever so make sure to integrate all their information.
- Format contested items clearly: "Proposed by Alice, supported by Bob, opposed by Carol: [suggestion]"
- Separate "agreed actions" from "open proposals" in the MEMORY STATE

Return only the consolidated memory (strictly ≤{{max_words}} words!), no additional formatting or explanation.
"""