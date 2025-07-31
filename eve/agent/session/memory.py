"""
Agent Memory System for Eve Platform

Provides automatic memory formation and context assembly for multi-agent conversations.
Memories are categorized as directives (behavioral rules) and episodes (conversation summaries
including contextual information) with full source message traceability.
"""

import json
import logging
import time
import traceback
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from bson import ObjectId
from eve.agent.session.session_llm import async_prompt, LLMContext, LLMConfig
from eve.agent.session.models import ChatMessage, Session
from eve.agent.session.memory_state import update_session_state, get_session_state
from eve.agent.session.memory_primitives import MemoryType, SessionMemory, UserMemory, AgentMemory
from eve.agent.session.config import DEFAULT_SESSION_SELECTION_LIMIT


"""
Main flow of memory formation:
Raw messages → LLM extraction → Raw Memories → Consolidation → Consolidated text blob

Database collections:
- memory_sessions: Raw memories formed inside of sessions.
    Contains a single document per raw memory (episode, directive, suggestion, fact)
- memory_user: Agent/User memory containing raw_memory_ids and a consolidated blob
    Contains a single document per agent/user pair
- memory_agent: Agent collective memory blobs containing raw_memory_ids and a consolidated blob
    Contains a single document per agent shard, where shards are specific topics / events / projects / etc that require a shared memory.

Main algorithm:
- Every N messages, trigger raw memory formation inside a session
- Run one LLM call to extract episodes and directives
- (if active) Run one extra LLM call for each collective agent shard to extract suggestions and facts (this includes the agent's persona and a custom shard prompt)
- Store all raw memories (episodes, directives, suggestions, facts) in memory_sessions and keep track of their ids
- Update memory_user and memory_agent with the corresponding new raw_memory_ids, trigger consolidations where needed
- Set SESSION_STATE.should_refresh_memory = True in modal dict to trigger a refresh of the memory context in the session
- (if active) Update AgentMemory.last_updated_at so that other sessions with this agent also refresh their collective memory cache


TODO:
Near term:
- actually use AgentMemory.is_active
- actually use AgentMemory.last_updated_at when checking session memory cache


Next:
- trigger memory formation based on token count instead of message count
- actually deploy background tasks for memory formation
- extract all hardcoded prompts into a single file

Finally:
- Create UI for memory shards and prompts, facts, suggestions, version history, ...

"""


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
    MAX_N_EPISODES_TO_REMEMBER = 2  # Number of episodes to remember from a session
    # Collective memory settings:
    MAX_SUGGESTIONS_COUNT_BEFORE_CONSOLIDATION = 2 # Number of suggestions to store before consolidating them into the agent's collective memory blob
    MAX_FACTS_PER_SHARD = 3 # Max number of facts to store per agent shard (fifo)
    
else:
    MEMORY_LLM_MODEL = "gpt-4o"
    MEMORY_FORMATION_INTERVAL = DEFAULT_SESSION_SELECTION_LIMIT  # Number of messages to wait before forming memories
    SESSION_MESSAGES_LOOKBACK_LIMIT = MEMORY_FORMATION_INTERVAL  # Max messages to look back in a session when forming raw memories

    # Normal memory settings:
    MAX_DIRECTIVES_COUNT_BEFORE_CONSOLIDATION = 5  # Number of individual memories to store before consolidating them into the agent's user_memory blob
    MAX_N_EPISODES_TO_REMEMBER = 10  # Number of episodes to remember from a session
    # Collective memory settings:
    MAX_SUGGESTIONS_COUNT_BEFORE_CONSOLIDATION = 10 # Number of suggestions to store before consolidating them into the agent's collective memory blob
    MAX_FACTS_PER_SHARD = 30 # Max number of facts to store per agent shard (fifo)
    
# LLMs cannot count tokens at all (weirdly), so instruct with word count:
SESSION_CONSOLIDATED_MEMORY_MAX_WORDS = 50  # Target word length for session consolidated memory
SESSION_DIRECTIVE_MEMORY_MAX_WORDS = 25  # Target word length for session directive memory
USER_MEMORY_MAX_WORDS = 150  # Target word count for consolidated user memory blob
MEMORY_SHARD_MAX_WORDS = 1000  # Target word count for consolidated agent collective memory blob

def get_agent_owner(agent_id: ObjectId) -> Optional[ObjectId]:
    """Get the owner of the agent"""
    try:
        from eve.agent.agent import Agent

        agent = Agent.from_mongo(agent_id)
        return agent.owner
    except Exception as e:
        print(f"Warning: Could not load agent owner for {agent_id}: {e}")
        return None


def messages_to_text(messages: List[ChatMessage]) -> str:
    """Convert messages to readable text for LLM processing"""
    text_parts = []
    for msg in messages:
        speaker = msg.name or msg.role
        content = msg.content

        # Add tool calls summary if present
        if msg.tool_calls:
            tools_summary = (
                f" [Used tools: {', '.join([tc.tool for tc in msg.tool_calls])}]"
            )
            content += tools_summary

        text_parts.append(f"{speaker}: {content}")

    return "\n".join(text_parts)


def get_memory_source_context(
    memory: SessionMemory, session_messages: List[ChatMessage] = None
) -> Dict[str, Any]:
    """
    Retrieve the full context for a memory by looking up its source messages.
    Useful for users who want to see the original conversation that led to a memory.
    """

    if not memory.source_message_ids:
        return {"error": "No source message IDs found for this memory"}

    source_context = {
        "memory_content": memory.content,
        "memory_type": memory.memory_type.value,
        "source_session_id": str(memory.source_session_id),
        "source_messages": [],
        "conversation_context": "",
    }

    # If session messages provided, look up from there
    if session_messages:
        MEMORY_SOURCE_CONTENT_TRUNCATION = 500
        source_messages = []
        for msg in session_messages:
            if hasattr(msg, "id") and msg.id in memory.source_message_ids:
                source_messages.append(
                    {
                        "id": str(msg.id),
                        "role": msg.role,
                        "name": msg.name,
                        "content": msg.content[:MEMORY_SOURCE_CONTENT_TRUNCATION]
                        + "..."
                        if len(msg.content) > MEMORY_SOURCE_CONTENT_TRUNCATION
                        else msg.content,
                        "has_tool_calls": bool(msg.tool_calls),
                        "tool_count": len(msg.tool_calls) if msg.tool_calls else 0,
                    }
                )

        source_context["source_messages"] = source_messages
        source_context["conversation_context"] = messages_to_text(
            [
                msg
                for msg in session_messages
                if hasattr(msg, "id") and msg.id in memory.source_message_ids
            ]
        )
    else:
        # In production, would query ChatMessage collection by IDs
        source_context["source_messages"] = [
            {"id": str(mid), "note": "Would query ChatMessage collection"}
            for mid in memory.source_message_ids
        ]

    return source_context


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4.5 characters per token)"""
    return int(len(text) / 4.5)


def store_session_memory(
    agent_id: ObjectId,
    extracted_data: Dict[str, List[str]],
    messages: List[ChatMessage],
    session_id: ObjectId,
) -> List[SessionMemory]:
    """Store extracted session to MongoDB with source traceability"""
    logging.debug(f"Extracted data: {extracted_data}")
    message_ids = [msg.id for msg in messages]
    # Extract all non-agent user_ids:
    related_users = list(
        set([msg.sender for msg in messages if msg.sender and msg.sender != agent_id])
    )

    # Get agent owner
    agent_owner = get_agent_owner(agent_id)

    memories_created = []
    new_directive_memories = []

    # Store directives
    for directive in extracted_data.get("directives", []):
        memory = SessionMemory(
            agent_id=agent_id,
            source_session_id=session_id,
            memory_type=MemoryType.DIRECTIVE,
            content=directive,
            source_message_ids=message_ids,
            related_users=related_users,
            agent_owner=agent_owner,
        )
        memory.save()
        memories_created.append(memory)
        new_directive_memories.append(memory)

    # Store episodes
    for episode in extracted_data.get("episodes", []):
        memory = SessionMemory(
            agent_id=agent_id,
            source_session_id=session_id,
            memory_type=MemoryType.EPISODE,
            content=episode,
            source_message_ids=message_ids,
            related_users=related_users,
            agent_owner=agent_owner,
        )
        memory.save()
        memories_created.append(memory)

    # Check if we need to consolidate any user memories for each related user
    for user_id in related_users:
        if user_id:  # Only process non-None user IDs
            _update_user_memory(agent_id, user_id, new_directive_memories)

    return memories_created


def _update_user_memory(
    agent_id: ObjectId, user_id: ObjectId, new_directive_memories: List[SessionMemory]
):
    """
    Add new directives to user_memory and maybe consolidate.
    Called after new directive memories are created.
    """

    try:
        # Get or create user memory record
        if agent_id is None or user_id is None:
            return

        user_memory = UserMemory.find_one_or_create(
            {"agent_id": agent_id, "user_id": user_id}
        )
        
        logging.debug(
            f"Found existing user memory with {len(user_memory.unabsorbed_directive_ids)} unabsorbed directives"
        )

        # Add new directives to unabsorbed list
        for directive_id in [m.id for m in new_directive_memories]:
            user_memory.unabsorbed_directive_ids.append(directive_id)
        user_memory.save()
        logging.debug(f"Added {len(new_directive_memories)} new directives to user memory")
        logging.debug(
            f"Total Unabsorbed directives in user memory: {len(user_memory.unabsorbed_directive_ids)}"
        )

        # Check if we need to consolidate
        if len(user_memory.unabsorbed_directive_ids) >= MAX_DIRECTIVES_COUNT_BEFORE_CONSOLIDATION:
            logging.debug(
                f"Triggering user memory consolidation for agent {agent_id}, user {user_id}: integrating {len(user_memory.unabsorbed_directive_ids)} unabsorbed directives"
            )
            # Store the user_memory for async consolidation
            import asyncio

            # Create a task to run the consolidation asynchronously
            loop = asyncio.get_event_loop()
            loop.create_task(_consolidate_user_directives(user_memory))

    except Exception as e:
        print(f"Error checking user directive consolidation: {e}")
        traceback.print_exc()


async def _consolidate_user_directives(user_memory: UserMemory):
    """
    Consolidate unabsorbed directive memories into the user memory blob using LLM.
    """
    try:
        # Get all unabsorbed directive memories
        unabsorbed_directives = []
        for directive_id in user_memory.unabsorbed_directive_ids:
            try:
                unabsorbed_directives.append(SessionMemory.from_mongo(directive_id))
            except Exception as e:
                print(f"Warning: Could not load directive {directive_id}: {e}")

        if not unabsorbed_directives:
            return

        # Prepare content for LLM consolidation
        current_memory = user_memory.content
        new_directives_text = "\n".join(
            [f"- {d.content}" for d in unabsorbed_directives]
        )

        n_mems = len(unabsorbed_directives)
        logging.debug(f"Consolidating {n_mems} directives to user_memory")
        logging.debug(f"Current memory length: {len(current_memory)} chars")
        logging.debug(f"New directives: {new_directives_text}")

        # Use LLM to consolidate memories
        consolidated_content = await _consolidate_memories_with_llm(
            current_memory, new_directives_text
        )

        # Update user memory
        user_memory.content = consolidated_content
        user_memory.unabsorbed_directive_ids = []  # Reset unabsorbed list
        user_memory.save()
        logging.debug(f"✓ Consolidated user memory updated (length: {len(consolidated_content)} chars)")
        

    except Exception as e:
        print(f"Error consolidating user directives: {e}")
        traceback.print_exc()


async def _consolidate_memories_with_llm(
    current_memory: str, new_directives: str
) -> str:
    """Use LLM to consolidate current user memory with new directive memories"""

    consolidation_prompt = f"""
CONSOLIDATE USER MEMORY
======================
You are helping to consolidate memories about a specific user's preferences and behavioral rules for an AI agent.

CURRENT CONSOLIDATED MEMORY:
{current_memory if current_memory else "(empty - this is the first consolidation)"}

NEW DIRECTIVE MEMORIES TO INTEGRATE:
{new_directives}

Your task: Create a single consolidated memory (≤{USER_MEMORY_MAX_WORDS} words) that combines the current memory with the new directives.

Requirements:
- Preserve all important behavioral rules and preferences from both current memory and new directives
- Remove redundancies and contradictions (newer directives override older ones)
- Keep the most specific and actionable guidance
- Use the actual user names from the directives (never use "User" or "the user")
- Focus on persistent preferences and behavioral rules that should guide future interactions
- Be concise but comprehensive

Return only the consolidated memory text, no additional formatting or explanation.
"""

    context = LLMContext(
        messages=[ChatMessage(role="user", content=consolidation_prompt)],
        config=LLMConfig(model=MEMORY_LLM_MODEL),
    )

    response = await async_prompt(context)
    consolidated_content = response.content.strip()

    logging.debug(f"LLM consolidation result: {consolidated_content}")
    return consolidated_content


async def process_memory_formation(
    agent_id: ObjectId, session_messages: List[ChatMessage], session: Session
) -> bool:
    """
    Extract memories from recent conversation using LLM.
    Called every N messages to form new memories.
    Only processes messages since session.last_memory_message_id to avoid duplicates.

    Returns True if memories were formed, False otherwise.
    """
    logging.debug(
        f"Processing memory formation for Session {session.id} with {len(session_messages)} messages"
    )

    # Find messages since last memory formation
    if session.last_memory_message_id:
        # Create message ID to index mapping for O(1) lookup
        message_id_to_index = {msg.id: i for i, msg in enumerate(session_messages)}
        
        # Find the position of the last memory formation message
        last_memory_position = message_id_to_index.get(session.last_memory_message_id, -1)
        
        # Get messages since last memory formation (excluding the last memory message itself)
        start_idx = last_memory_position + 1
        recent_messages = session_messages[start_idx:]
    else:
        # No previous memory formation, use all messages
        start_idx = 0
        recent_messages = session_messages

    logging.debug(
        f"Extracting memories from messages {start_idx+1}-{len(session_messages)} (total: {len(recent_messages)} messages)"
    )

    if not recent_messages:
        print("No recent messages to process")
        return False

    try:
        # Convert messages to text for LLM processing
        conversation_text = messages_to_text(recent_messages)
        logging.debug(f"Conversation text length: {len(conversation_text)} characters")

        # Extract memories using LLM
        extracted_data = await extract_memories_with_llm(conversation_text)

        # Store extracted memories in database
        memories_created = store_session_memory(
            agent_id, extracted_data, recent_messages, session.id
        )

        if memories_created:
            logging.debug(
                f"✓ Formed {len(memories_created)} memories from {len(recent_messages)} messages"
            )
            logging.debug(
                f"  Memory types: {[m.memory_type.value for m in memories_created]}"
            )
            return True

    except Exception as e:
        print(f"Error forming memories: {e}")
        traceback.print_exc()

    return False


async def extract_memories_with_llm(conversation_text: str) -> Dict[str, List[str]]:
    """Use LLM to extract categorized memories from conversation text"""

    memory_extraction_prompt = f"""
EXTRACT DURABLE MEMORIES
========================
CONVERSATION:
{conversation_text}

Return **exactly** this JSON:
{{
  "consolidated_memory": "<ONE factual digest, ≤{SESSION_CONSOLIDATED_MEMORY_MAX_WORDS} words>",
  "directive": "<persistent rule(s), ≤{SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words or empty string>"
}}

Create new memories following these rules:

1. CONSOLIDATED_MEMORY: Create EXACTLY ONE factual memory (maximum {SESSION_CONSOLIDATED_MEMORY_MAX_WORDS} words) that consolidates what actually happened in the conversation. This memory will be used to improve the agent's contextual recall in long conversations.
   - Record concrete facts and events: who did/said what, what was created, what tools were used, what topics were discussed
   - Specifically focus on the instructions, preferences, goals and feedback expressed by the user(s)
   - Avoid commentary or analysis, create memories that stand on their own without context

2. DIRECTIVE: Create AT MOST ONE permanent directive (maximum {SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words) ONLY if there are clear, long-lasting rules, preferences, or behavioral guidelines that should be applied consistently in all future interactions with the user. If none exist (highly likely), just leave empty.
   
   ONLY include as long-lasting directives:
   - Explicit behavioral rules or guidelines ("always ask permission before...", "never do X", "remember to always Y")
   - Stable, long-term preferences that should guide future behavior consistently
   - Clear exceptions or special handling rules for interaction patterns with the current user that should persist across many future conversations ("Whenever I ask you to do X, you should do it like so..")
   
   DO NOT include as directives:
   - One-time requests or specific tasks ("create a story about...", "make an image of...")
   - Ad hoc instructions relevant for the current conversation context only
   - Temporary or situational commands
   - Context-specific requests that don't apply broadly
   
   - ALWAYS use specific user names from the conversation (NEVER use "User", "the user", or "Agent")
   - Example: "Gene prefers permission before generating images, wants surreal art themes consistently"
   - Counter-example (DO NOT make directive): "Gene requested a story about clockmaker" (this is a one-time request)

CRITICAL REQUIREMENTS: 
- BE VERY STRICT about directives - most conversations will have NO directives, only consolidated memories
- Directives are for rules that should persist across many future conversations, not one-time requests
- Focus on facts, not interpretations or commentary
- Record what actually happened, not what it means or demonstrates
- ALWAYS use actual names from the conversation - scan the conversation for "name:" patterns
- NEVER use generic terms like "User", "the user", "Agent", "the agent", "someone", "they"
- Avoid vague words like "highlighted", "demonstrated", "enhanced", "experience" that wont help the agent in future interactions
- Just state what was said, done, created, or discussed with specific names in a concise manner
"""

    # Use LLM to extract memories
    context = LLMContext(
        messages=[ChatMessage(role="user", content=memory_extraction_prompt)],
        config=LLMConfig(model=MEMORY_LLM_MODEL),
    )

    response = await async_prompt(context)

    # Clean up response to extract JSON
    response_text = response.content.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    # Parse JSON response
    try:
        extracted_data = json.loads(response_text)
        # Convert to the expected format for storage
        formatted_data = {
            "directives": [extracted_data.get("directive", "")]
            if extracted_data.get("directive", "").strip()
            else [],
            "episodes": [extracted_data.get("consolidated_memory", "")]
            if extracted_data.get("consolidated_memory", "").strip()
            else [],
        }
        return formatted_data
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from LLM response: {response_text[:200]}...")
        extracted_data = {"directives": [], "episodes": []}

    # Print the messages and the prompt:
    logging.debug("########################")
    logging.debug("Forming new memories...")
    logging.debug(f"--- Messages: ---\n{context.messages}")
    logging.debug(f"--- Prompt: ---\n{memory_extraction_prompt}")
    logging.debug(f"--- Memories: ---\n{extracted_data}")
    logging.debug("########################")

    return extracted_data

async def assemble_memory_context(agent_id: ObjectId, session_id: Optional[ObjectId] = None, last_speaker_id: Optional[ObjectId] = None, session: Optional['Session'] = None) -> str:
    """
    Assemble relevant memories for context injection into prompts.
    Uses session-level caching to minimize database queries.
    
    Args:
        agent_id: ID of the agent to get memories for
        session_id: Current session ID to prioritize session-specific memories.
        last_speaker_id: ID of the user who spoke the last message for prioritization.
        session: Optional session object to update in place (avoids extra MongoDB query).
    
    Returns:
        Formatted memory context string for prompt injection.
    """

    start_time = time.time()
    
    # print(f"🧠 MEMORY ASSEMBLY PROFILING - Agent: {agent_id}")
    # print(f"   Session: {session_id}, Last Speaker: {last_speaker_id}")
    
    # Check if we can use cached memory context from modal dict
    if session_id and agent_id:
        try:
            get_session_state_start = time.time()
            session_state = await get_session_state(agent_id, session_id)
            get_session_state_time = time.time() - get_session_state_start
            # print(f"   ⏱️  get_session_state took: {get_session_state_time:.3f}s")

            cached_context = session_state.get("cached_memory_context")
            should_refresh = session_state.get("should_refresh_memory", True)
            
            if cached_context and not should_refresh:
                total_time = time.time() - start_time
                # print(f"   ⚡ USING CACHED MEMORY: {total_time:.3f}s")
                logging.debug("Not refreshing memory context:")
                logging.debug(f"Cached context: {cached_context}")
                return cached_context
            else:
                # print(f"   🔄 Cache missing or refresh needed")
                logging.debug(f"Memory context, Should refresh: {should_refresh}")
                
        except Exception as e:
            print(f"   ❌ Error checking cached memory: {e}")
    
    # TODO: instead of last speaker, iterate over all human users in the session
    user_id = last_speaker_id

    # Initialize all variables at the start to avoid scoping issues
    user_memory_content = ""
    unabsorbed_directives = []
    episode_memories = []
    user_memory = None

    # Step 1: User Memory
    try:
        query_start = time.time()
        # Get user memory blob for this user:
        if (
            user_id is not None and agent_id is not None
        ):  # Only query if both IDs are not None
            user_memory = UserMemory.find_one_or_create(
                {"agent_id": agent_id, "user_id": user_id}
            )
            if user_memory:
                user_memory_content = user_memory.content or ""  # Handle None content
                unabsorbed_directive_ids = (
                    user_memory.unabsorbed_directive_ids or []
                )  # Handle None list
                # Get unabsorbed directives:
                if (
                    unabsorbed_directive_ids and user_id is not None
                ):  # Only query if there are IDs to look up and user_id is valid
                    unabsorbed_directives = SessionMemory.find(
                        {
                            "agent_id": agent_id,
                            "memory_type": "directive",
                            "related_users": user_id,
                            "_id": {"$in": unabsorbed_directive_ids},
                        }
                    )
        query_time = time.time() - query_start
        # print(
        #     f"   ⏱️  User Memory Assembly: {query_time:.3f}s (user_memory: {'yes' if user_memory else 'no'}, {len(unabsorbed_directives)} unabsorbed directives)"
        # )

    except Exception as e:
        print(f"   ❌ Error retrieving user memory: {e}")

    # Step 2: Episodes from SessionMemory:
    try:
        if session_id and agent_id is not None:
            query_start = time.time()
            episode_query = {"source_session_id": session_id, "memory_type": MemoryType.EPISODE.value}
            episode_memories = SessionMemory.find(episode_query, sort="createdAt", desc=True)

            # Get list of MAX_N_EPISODES_TO_REMEMBER most recent, raw episode memories:
            episode_memories = episode_memories[:MAX_N_EPISODES_TO_REMEMBER]
            # Reverse the list to put the most recent episodes at the bottom:
            episode_memories.reverse()

            query_time = time.time() - query_start
            # print(
            #     f"   ⏱️  Session memory assembly: {query_time:.3f}s (user_memory: {'yes' if user_memory else 'no'}, {len(episode_memories)} episodes)"
            # )
    except Exception as e:
        print(f"   ❌ Error assembling session memories: {e}")

    # Step 3: Full memory context assembly:
    memory_context = ""

    if len(user_memory_content) > 0:  # user_memory blob:
        memory_context += f"### Consolidated User Memory:\n\n{user_memory_content}\n\n"

    if len(unabsorbed_directives) > 0:
        memory_context += f"### Recent Directives (most recent at bottom):\n\n"
        for directive in unabsorbed_directives:
            memory_context += f"- {directive.content}\n"
        memory_context += "\n"

    if len(episode_memories) > 0:
        memory_context += (
            f"### Current conversation context (most recent episodes at bottom):\n\n"
        )
        for episode in episode_memories:
            memory_context += f"- {episode.content}\n"
        memory_context += "\n"

    if len(memory_context) > 0:
        memory_context = "## Your Memory:\n\n" + memory_context
    else:
        memory_context = ""
    
    # Step 4: Cache the memory context in modal dict
    if session_id and agent_id:
        try:
            cache_start = time.time()
            await update_session_state(agent_id, session_id, {
                "cached_memory_context": memory_context,
                "should_refresh_memory": False
            })
            # print(f"   💾 Memory context cached for session {session_id} in {time.time() - cache_start:.3f}s")
        except Exception as e:
            print(f"   ❌ Error caching memory context: {e}")
    
    # Step 5: Final stats
    total_time = time.time() - start_time
    final_tokens = estimate_tokens(memory_context)
    # print(f"   ⏱️  TOTAL TIME: {total_time:.3f}s")
    # print(f"   📏 Context Length: {len(memory_context)} chars (~{final_tokens} tokens)")

    logging.debug(f"Fully Assembled Memory context:\n{memory_context}")
    if LOCAL_DEV:
        print(f"Fully Assembled Memory context:\n{memory_context}")

    return memory_context

async def maybe_form_memories(agent_id: ObjectId, session: Session, force_memory_formation: bool = False) -> bool:
    """
    Check if memory formation should run based on messages elapsed since last formation.
    If force_memory_formation is True, skip the interval check and always form memories.
    Returns True if memories were formed.
    """
    start_time = time.time()
    
    from eve.agent.session.session import select_messages
    session_messages = select_messages(
        session, selection_limit=SESSION_MESSAGES_LOOKBACK_LIMIT
    )

    if not agent_id or not session_messages or len(session_messages) == 0:
        print(f"No agent or messages found for session {session.id}")
        return False

    # Create message ID to index mapping for O(1) lookup
    message_id_to_index = {msg.id: i for i, msg in enumerate(session_messages)}
    
    # Find the position of the last memory formation message
    last_memory_position = -1
    if session.last_memory_message_id:
        last_memory_position = message_id_to_index.get(session.last_memory_message_id, -1)

    # Calculate messages since last memory formation
    messages_since_last = len(session_messages) - last_memory_position - 1

    # Update memory state on new message
    await update_session_state(agent_id, session.id, {
        "last_activity": datetime.now(timezone.utc).isoformat(),
        "message_count_since_memory": messages_since_last
    })

    logging.debug(f"Session {session.id}: {len(session_messages)} total messages, {messages_since_last} since last memory formation")
    
    # Check if we should form memories (early return to avoid slow queries)
    if not force_memory_formation and messages_since_last < MEMORY_FORMATION_INTERVAL:
        logging.debug(f"No memory formation needed: {messages_since_last} < {MEMORY_FORMATION_INTERVAL} interval")
        logging.debug(f"Maybe form memories took {time.time() - start_time:.2f} seconds to complete")
        return False
    
    if force_memory_formation:
        print(f"Triggering FORCED memory formation: {messages_since_last} messages to process")
    else:
        print(f"Triggering memory formation: {messages_since_last} >= {MEMORY_FORMATION_INTERVAL} interval")

    try:
        if session_messages:
            # Process memory formation
            result = await process_memory_formation(
                agent_id, 
                session_messages, 
                session
            )
            
            # Update the session's last memory formation position and invalidate memory cache
            latest_message = session_messages[-1]
            session.last_memory_message_id = latest_message.id
            session.save()
            logging.debug(f"Updated last_memory_message_id to {latest_message.id}")
            
            # Also update the modal.Dict state to reset message count and update last memory message
            try:
                await update_session_state(agent_id, session.id, {
                    "last_memory_message_id": str(latest_message.id),
                    "message_count_since_memory": 0, 
                    "should_refresh_memory": True
                })
                logging.debug(f"Updated modal.Dict state for agent {agent_id}, session {session.id}")
            except Exception as e:
                logging.error(f"Error updating modal.Dict state after memory formation: {e}")
                traceback.print_exc()
        
        print(f"Maybe form memories took {time.time() - start_time:.2f} seconds to complete")
        return result

    except Exception as e:
        print(f"Error processing memory formation: {e}")
        traceback.print_exc()
        return False
