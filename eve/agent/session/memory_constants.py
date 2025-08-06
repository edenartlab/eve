from eve.agent.session.config import DEFAULT_SESSION_SELECTION_LIMIT

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
    
# LLMs cannot count tokens at all (weirdly), so instruct with word count:
SESSION_CONSOLIDATED_MEMORY_MAX_WORDS = 50  # Target word length for session consolidated memory
SESSION_DIRECTIVE_MEMORY_MAX_WORDS    = 25  # Target word length for session directive memory
USER_MEMORY_MAX_WORDS  = 150   # Target word count for consolidated user memory blob
MEMORY_SHARD_MAX_WORDS = 1000  # Target word count for consolidated agent collective memory blob
CONVERSATION_TEXT_TOKEN = "---conversation_text---"

# Default memory extraction prompt for episodes and directives:
MEMORY_EXTRACTION_PROMPT = f"""Task: Extract persistent memories from the conversation.
Return **exactly** this JSON:
{{
  "episode": "<ONE factual digest of ≤{SESSION_CONSOLIDATED_MEMORY_MAX_WORDS} words>",
  "directive": "<(at most) one persistent rule of ≤{SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words OR an empty string>"
}}

Conversation text:
{CONVERSATION_TEXT_TOKEN}

Create new memories following these rules:

1. EPISODE: Create EXACTLY ONE factual memory (maximum {SESSION_CONSOLIDATED_MEMORY_MAX_WORDS} words) that consolidates what actually happened in the conversation. This memory will be used to improve the agent's contextual recall in long conversations.
   - Record concrete facts and events: who did/said what, what was created, what tools were used, what topics were discussed
   - Specifically focus on the instructions, preferences, goals and feedback expressed by the user(s)
   - Avoid commentary or analysis, create memories that stand on their own without context

2. DIRECTIVE: Create AT MOST ONE permanent directive (maximum {SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words) ONLY if there are clear, long-lasting rules, preferences, or behavioral guidelines that should be applied consistently in all future interactions with the user. If none exist (highly likely), just leave empty.
   
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
- BE VERY STRICT about the directive - most conversations will have NO directive, only an episode
- A directive is a rule that should persist across many future conversations, not one-time requests
- Focus on facts, not interpretations or commentary
- Record what actually happened, not what it means or demonstrates
- ALWAYS use actual names from the conversation - scan the conversation for "name:" patterns
- NEVER use generic terms like "User", "the user", "Agent", "the agent", "someone", "they"
- Avoid vague words like "highlighted", "demonstrated", "enhanced", "experience" that wont help the agent in future interactions
- Just state what was said, done, created, or discussed with specific names in a concise manner
"""