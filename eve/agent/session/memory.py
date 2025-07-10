"""
Agent Memory System for Eve Platform

Provides automatic memory formation and context assembly for multi-agent conversations.
Memories are categorized as directives (behavioral rules) and episodes (conversation summaries
including contextual information) with full source message traceability.
"""

import json
import time
import traceback
from enum import Enum
from typing import List, Optional, Dict, Any
from bson import ObjectId
from pydantic import BaseModel, field_serializer

from eve.mongo import Collection, Document, get_collection
from eve.agent.session.session_llm import async_prompt, LLMContext, LLMConfig
from eve.agent.session.models import ChatMessage, Session

LOCAL_DEV = True

# Memory formation settings:
if LOCAL_DEV:
    MEMORY_FORMATION_INTERVAL = 4 # Number of messages to wait before forming memories
    SESSION_MESSAGES_LOOKBACK_LIMIT = MEMORY_FORMATION_INTERVAL  # Max messages to look back in a session when forming raw memories
    MAX_RAW_MEMORY_COUNT      = 2 # Number of individual memories to store before consolidating them into the agent's user_memory blob
    MAX_N_EPISODES_TO_REMEMBER = 10 # Number of episodes to remember from a session
    MEMORY_LLM_MODEL = "gpt-4o-mini"
else:
    MEMORY_FORMATION_INTERVAL = 20 # Number of messages to wait before forming memories
    SESSION_MESSAGES_LOOKBACK_LIMIT = MEMORY_FORMATION_INTERVAL  # Max messages to look back in a session when forming raw memories
    MAX_RAW_MEMORY_COUNT      = 3 # Number of individual memories to store before consolidating them into the agent's user_memory blob
    MAX_N_EPISODES_TO_REMEMBER = 10 # Number of episodes to remember from a session
    MEMORY_LLM_MODEL = "gpt-4o"

# LLMs cannot count tokens at all (weirdly), so instruct with word count:
SESSION_CONSOLIDATED_MEMORY_MAX_WORDS = 50 # Target word length for session consolidated memory
SESSION_DIRECTIVE_MEMORY_MAX_WORDS    = 25 # Target word length for session directive memory
USER_MEMORY_MAX_WORDS = 150 # Target word count for consolidated user memory blob

class MemoryType(Enum):
    EPISODE = "episode"      # Summary of a section of the conversation in a session
    DIRECTIVE = "directive"  # User instructions, preferences, behavioral rules

@Collection("memory_sessions")
class SessionMemory(Document):
    """Individual memory record stored in MongoDB"""
    
    agent_id: ObjectId
    memory_type: MemoryType
    content: str
    
    # Context tracking for traceability
    source_session_id: Optional[ObjectId] = None
    source_message_ids: List[ObjectId] = []
    related_users: List[ObjectId] = []
    
    agent_owner: Optional[ObjectId] = None
    
    @field_serializer("memory_type")
    def serialize_memory_type(self, value: MemoryType) -> str:
        return value.value
    
    @classmethod
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert enum to string for MongoDB storage"""
        # Ensure kwargs is a dict to prevent "argument after ** must be a mapping" error
        if kwargs is None:
            kwargs = {}
        if "memory_type" in schema and hasattr(schema["memory_type"], "value"):
            schema["memory_type"] = schema["memory_type"].value
        return schema
    
    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert string back to enum from MongoDB"""
        # Ensure kwargs is a dict to prevent "argument after ** must be a mapping" error
        if kwargs is None:
            kwargs = {}
        if "memory_type" in schema and isinstance(schema["memory_type"], str):
            schema["memory_type"] = MemoryType(schema["memory_type"])
        return schema

@Collection("memory_user")
class UserMemory(Document):
    """Consolidated user memory blob for agent/user pairs"""
    
    agent_id: ObjectId
    user_id: ObjectId
    content: str  # Consolidated memory blob
    agent_owner: Optional[ObjectId] = None
    unabsorbed_directive_ids: List[ObjectId] = []  # Track which directive memories haven't been consolidated yet

    @classmethod
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data for MongoDB storage"""
        # Ensure kwargs is a dict to prevent "argument after ** must be a mapping" error
        if kwargs is None:
            kwargs = {}
        return schema
    
    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data from MongoDB"""
        # Ensure kwargs is a dict to prevent "argument after ** must be a mapping" error
        if kwargs is None:
            kwargs = {}
        return schema
    @classmethod
    def find_one_or_create(cls, query, defaults=None):
        """Find a document or create and save a new one if it doesn't exist."""
        if defaults is None:
            defaults = {}
        
        doc = cls.find_one(query)
        if doc:
            return doc
        else:
            # Create new instance and save it
            new_doc = {**query, **defaults}
            instance = cls(**new_doc)
            instance.save()
            return instance


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
            tools_summary = f" [Used tools: {', '.join([tc.tool for tc in msg.tool_calls])}]"
            content += tools_summary
            
        text_parts.append(f"{speaker}: {content}")
    
    return "\n".join(text_parts)

def get_memory_source_context(
    memory: SessionMemory, 
    session_messages: List[ChatMessage] = None
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
        "conversation_context": ""
    }
    
    # If session messages provided, look up from there
    if session_messages:
        MEMORY_SOURCE_CONTENT_TRUNCATION = 500
        source_messages = []
        for msg in session_messages:
            if hasattr(msg, 'id') and msg.id in memory.source_message_ids:
                source_messages.append({
                    "id": str(msg.id),
                    "role": msg.role,
                    "name": msg.name,
                    "content": msg.content[:MEMORY_SOURCE_CONTENT_TRUNCATION] + "..." if len(msg.content) > MEMORY_SOURCE_CONTENT_TRUNCATION else msg.content,
                    "has_tool_calls": bool(msg.tool_calls),
                    "tool_count": len(msg.tool_calls) if msg.tool_calls else 0
                })
        
        source_context["source_messages"] = source_messages
        source_context["conversation_context"] = messages_to_text(
            [msg for msg in session_messages if hasattr(msg, 'id') and msg.id in memory.source_message_ids]
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
    session_id: ObjectId
) -> List[SessionMemory]:
    """Store extracted session to MongoDB with source traceability"""
    
    print(f"Extracted data: {extracted_data}")
    message_ids = [msg.id for msg in messages]
    # Extract all non-agent user_ids:
    related_users = list(set([msg.sender for msg in messages if msg.sender and msg.sender != agent_id]))

    # Get agent owner
    agent_owner = get_agent_owner(agent_id)
    
    memories_created = []
    new_directive_memories = []
    
    # Store directives (highest priority)
    for directive in extracted_data.get("directives", []):
        memory = SessionMemory(
            agent_id=agent_id,
            memory_type=MemoryType.DIRECTIVE,
            content=directive,
            source_session_id=session_id,
            source_message_ids=message_ids,
            related_users=related_users,
            agent_owner=agent_owner
        )
        memory.save()
        memories_created.append(memory)
        new_directive_memories.append(memory)
    
    # Store episodes
    for episode in extracted_data.get("episodes", []):
        memory = SessionMemory(
            agent_id=agent_id,
            memory_type=MemoryType.EPISODE,
            content=episode,
            source_session_id=session_id,
            source_message_ids=message_ids,
            related_users=related_users,
            agent_owner=agent_owner
        )
        memory.save()
        memories_created.append(memory)

    # Check if we need to consolidate any user memories for each related user
    for user_id in related_users:
        if user_id:  # Only process non-None user IDs
            _update_user_memory(agent_id, user_id, new_directive_memories)
    
    return memories_created

def _update_user_memory(agent_id: ObjectId, user_id: ObjectId, new_directive_memories: List[SessionMemory]):
    """
    Add new directives to user_memory and maybe consolidate.
    Called after new directive memories are created.
    """

    try:
        # Get or create user memory record
        # Only proceed if both agent_id and user_id are not None
        if agent_id is None or user_id is None:
            print(f"Skipping user memory update - agent_id: {agent_id}, user_id: {user_id}")
            return
        
        user_memory = UserMemory.find_one_or_create({
            "agent_id": agent_id, 
            "user_id": user_id
        })
        print(f"Found existing user memory with {len(user_memory.unabsorbed_directive_ids)} unabsorbed directives")

        # Add new directive to unabsorbed list
        for directive_id in [m.id for m in new_directive_memories]:
            user_memory.unabsorbed_directive_ids.append(directive_id)
        user_memory.save()
        print(f"Added {len(new_directive_memories)} new directives to user memory")
        print(f"Total Unabsorbed directives in user memory: {len(user_memory.unabsorbed_directive_ids)}")
        
        # Check if we need to consolidate
        if len(user_memory.unabsorbed_directive_ids) >= MAX_RAW_MEMORY_COUNT:
            print(f"Triggering user memory consolidation for agent {agent_id}, user {user_id}: integrating {len(user_memory.unabsorbed_directive_ids)} unabsorbed directives")
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
            print("No valid unabsorbed directives found for consolidation")
            return
        
        # Prepare content for LLM consolidation
        current_memory = user_memory.content
        new_directives_text = "\n".join([f"- {d.content}" for d in unabsorbed_directives])
        
        print(f"Consolidating {len(unabsorbed_directives)} directives into user memory")
        print(f"Current memory length: {len(current_memory)} chars")
        print(f"New directives: {new_directives_text}")
        
        # Use LLM to consolidate memories
        consolidated_content = await _consolidate_memories_with_llm(current_memory, new_directives_text)
        
        # Update user memory
        user_memory.content = consolidated_content
        user_memory.unabsorbed_directive_ids = []  # Reset unabsorbed list
        user_memory.save()
        
        print(f"✓ Consolidated user memory updated (length: {len(consolidated_content)} chars)")
        
    except Exception as e:
        print(f"Error consolidating user directives: {e}")
        traceback.print_exc()

async def _consolidate_memories_with_llm(current_memory: str, new_directives: str) -> str:
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
        config=LLMConfig(model=MEMORY_LLM_MODEL)
    )

    response = await async_prompt(context)
    consolidated_content = response.content.strip()
    
    print(f"LLM consolidation result: {consolidated_content}")
    return consolidated_content

async def process_memory_formation(
    agent_id: ObjectId,
    session_messages: List[ChatMessage], 
    session_id: ObjectId
) -> bool:
    """
    Extract memories from recent conversation using LLM.
    Called every N messages to form new memories.
    
    Returns True if memories were formed, False otherwise.
    """

    print(f"Processing memory formation for Session {session_id} with {len(session_messages)} messages")
    
    # Use interval for recent messages to process
    interval = MEMORY_FORMATION_INTERVAL
    start_idx = max(0, len(session_messages) - interval)
    recent_messages = session_messages[start_idx:]
    
    print(f"Extracting memories from messages {start_idx+1}-{len(session_messages)} (total: {len(recent_messages)} messages)")
    
    if not recent_messages:
        print("No recent messages to process")
        return False
    
    try:
        # Convert messages to text for LLM processing
        conversation_text = messages_to_text(recent_messages)
        print(f"Conversation text length: {len(conversation_text)} characters")
        
        # Extract memories using LLM
        extracted_data = await extract_memories_with_llm(conversation_text)
        
        # Store extracted memories in database
        memories_created = store_session_memory(
            agent_id, extracted_data, recent_messages, session_id
        )
        
        if memories_created:
            print(f"✓ Formed {len(memories_created)} memories from {len(recent_messages)} messages")
            print(f"  Memory types: {[m.memory_type.value for m in memories_created]}")
            return True
        else:
            print("No memories were created from extracted data")
        
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

2. DIRECTIVE: Create AT MOST ONE consolidated directive (maximum {SESSION_DIRECTIVE_MEMORY_MAX_WORDS} words) ONLY if there are clear, long-lasting rules, preferences, or behavioral guidelines that should be applied consistently in all future interactions. If none exist (highly likely), leave empty.
   
   ONLY include as long-lasting directives:
   - Explicit behavioral rules or guidelines ("always ask permission before...", "never do X", "remember to always Y")
   - Stable, long-term preferences that should guide future behavior consistently
   - Clear exceptions or special handling rules
   - Persistent working styles or interaction preferences
   
   DO NOT include as directives:
   - One-time requests or specific tasks ("create a story about...", "make an image of...")
   - Ad hoc instructions relevant for the current conversation only
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
        config=LLMConfig(model=MEMORY_LLM_MODEL)
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
            "directives": [extracted_data.get("directive", "")] if extracted_data.get("directive", "").strip() else [],
            "episodes": [extracted_data.get("consolidated_memory", "")] if extracted_data.get("consolidated_memory", "").strip() else []
        }
        return formatted_data
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from LLM response: {response_text[:200]}...")
        extracted_data = {"directives": [], "episodes": []}

    # Print the messages and the prompt:
    print("########################")
    print("Forming new memories...")
    print(f"--- Messages: ---\n{context.messages}")
    print(f"--- Prompt: ---\n{memory_extraction_prompt}")
    print(f"--- Memories: ---\n{extracted_data}")
    print("########################")
    
    return extracted_data

def assemble_memory_context(agent_id: ObjectId, session_id: Optional[ObjectId] = None, last_speaker_id: Optional[ObjectId] = None) -> str:
    """
    Assemble relevant memories for context injection into prompts.
    
    Args:
        agent_id: ID of the agent to get memories for
        session_id: Current session ID to prioritize session-specific memories.
        last_speaker_id: ID of the user who spoke the last message for prioritization.
    
    Returns:
        Formatted memory context string for prompt injection.
    """
    
    start_time = time.time()
    print(f"🧠 MEMORY ASSEMBLY PROFILING - Agent: {agent_id}")
    print(f"   Session: {session_id}, Last Speaker: {last_speaker_id}")
    
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
        if user_id is not None and agent_id is not None:  # Only query if both IDs are not None
            user_memory = UserMemory.find_one_or_create({
                "agent_id": agent_id,
                "user_id": user_id
            })
            if user_memory:
                user_memory_content = user_memory.content or ""  # Handle None content
                unabsorbed_directive_ids = user_memory.unabsorbed_directive_ids or []  # Handle None list
                # Get unabsorbed directives:
                if unabsorbed_directive_ids and user_id is not None:  # Only query if there are IDs to look up and user_id is valid
                    unabsorbed_directives = SessionMemory.find({
                        "agent_id": agent_id,
                        "memory_type": "directive",
                        "related_users": user_id,
                        "_id": {"$in": unabsorbed_directive_ids}
                    })
        query_time = time.time() - query_start
        print(f"   ⏱️  User Memory Assembly: {query_time:.3f}s (user_memory: {'yes' if user_memory else 'no'}, {len(unabsorbed_directives)} unabsorbed directives)")
    
    except Exception as e:
        print(f"   ❌ Error retrieving user memory: {e}")
        # Variables are already initialized, so we can continue
    
    # Step 2: Session Memory:
    try:
        if session_id and agent_id is not None:  # Only query if session_id and agent_id are not None
            query_start = time.time()
            episode_query = {"source_session_id": session_id}
            episode_memories = SessionMemory.find(episode_query, sort="createdAt", desc=True)
            # first = new, last = old

            # Get list of MAX_N_EPISODES_TO_REMEMBER most recent, raw episode memories:
            episode_memories = [m for m in episode_memories if m.memory_type == MemoryType.EPISODE][:MAX_N_EPISODES_TO_REMEMBER]
            # Reverse the list to put the most recent episodes at the bottom:
            episode_memories.reverse()

            if not unabsorbed_directives: # Get list of all session directives:
                unabsorbed_directives = [m for m in episode_memories if m.memory_type == MemoryType.DIRECTIVE]
                # Reverse the list to put the most recent directives at the bottom:
                unabsorbed_directives.reverse()

            query_time = time.time() - query_start
            print(f"   ⏱️  Session memory assembly: {query_time:.3f}s (user_memory: {'yes' if user_memory else 'no'}, {len(unabsorbed_directives)} unabsorbed directives, {len(episode_memories)} episodes)")

    except Exception as e:
        print(f"   ❌ Error retrieving session memories: {e}")
        # Variables are already initialized, so we can continue

    # Step 3: Full memory context assembly:
    memory_context = ""

    if len(user_memory_content) > 0: # user_memory blob:
        memory_context += f"### Consolidated User Memory:\n\n{user_memory_content}\n\n"
    
    if len(unabsorbed_directives) > 0:
        memory_context += f"### Recent Directives (most recent at bottom):\n\n"
        for directive in unabsorbed_directives:
            memory_context += f"- {directive.content}\n"
        memory_context += "\n"
    
    if len(episode_memories) > 0:
        memory_context += f"### Current conversation context (most recent episodes at bottom):\n\n"
        for episode in episode_memories:
            memory_context += f"- {episode.content}\n"
        memory_context += "\n"

    if len(memory_context) > 0:
        memory_context = "## Your Memory:\n\n" + memory_context
    else:
        print("No memory context to assemble")
        memory_context = ""
    
    # Step 4: Final stats
    total_time = time.time() - start_time
    final_tokens = estimate_tokens(memory_context)
    print(f"   ⏱️  TOTAL TIME: {total_time:.3f}s")
    print(f"   📏 Context Length: {len(memory_context)} chars (~{final_tokens} tokens)")
    
    return memory_context

async def maybe_form_memories(
    agent_id: ObjectId, 
    session: Session,
    interval: int = MEMORY_FORMATION_INTERVAL
) -> bool:
    """
    Check if memory formation should run based on messages elapsed since last formation.
    Returns True if memories were formed.
    """
    from eve.agent.session.session import select_messages

    session_messages = select_messages(session, selection_limit=SESSION_MESSAGES_LOOKBACK_LIMIT)

    if not agent_id or not session_messages:
        print(f"No agent or messages found for session {session.id}")
        return False
    
    # Find the position of the last memory formation message
    last_memory_position = -1
    if session.last_memory_message_id:
        for i, msg in enumerate(session_messages):
            if msg.id == session.last_memory_message_id:
                last_memory_position = i
                break
    
    # Calculate messages since last memory formation
    messages_since_last = len(session_messages) - last_memory_position - 1
    
    print(f"Session {session.id}: {len(session_messages)} total messages, {messages_since_last} since last memory formation")
    
    # Get the number of unabsorbed directives for the last speaker:
    unabsorbed_directive_count = -1
    try: 
        # Only query if both agent_id and session.owner are not None
        if agent_id is not None and session.owner is not None:
            user_memory = UserMemory.find_one_or_create({
                "agent_id": agent_id,
                "user_id": session.owner
            })
            
            unabsorbed_directive_count = len(user_memory.unabsorbed_directive_ids) if user_memory else 0
    except Exception as e:
        print(f"Error getting unabsorbed directive count: {e}")
        traceback.print_exc()
        
    print(f"Unabsorbed directive count: {unabsorbed_directive_count} / {MAX_RAW_MEMORY_COUNT}")
    
    # Check if we should form memories
    if messages_since_last < interval:
        print(f"No memory formation needed: {messages_since_last} < {interval} interval")
        return False
    
    print(f"Triggering memory formation: {messages_since_last} >= {interval} interval")
    if unabsorbed_directive_count >= MAX_RAW_MEMORY_COUNT:
        print(f"Will also absorb unabsorbed directives: {unabsorbed_directive_count} >= {MAX_RAW_MEMORY_COUNT} unabsorbed directives")
            
    try:
        # Process memory formation
        result = await process_memory_formation(
            agent_id, 
            session_messages, 
            session.id
        )
        
        # Update the session's last memory formation position
        if session_messages:
            latest_message = session_messages[-1]
            session.last_memory_message_id = latest_message.id
            session.save()
            print(f"Updated last_memory_message_id to {latest_message.id}")
        
        return result
        
    except Exception as e:
        print(f"Error processing memory formation: {e}")
        traceback.print_exc()
        return False
