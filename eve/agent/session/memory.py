"""
Agent Memory System for Eve Platform

Provides automatic memory formation and context assembly for multi-agent conversations.
Memories are categorized as directives (behavioral rules), facts (contextual information), 
and episodes (conversation summaries) with full source message traceability.
"""

import json
import hashlib
import traceback
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel, field_serializer

from eve.mongo import Collection, Document
from eve.agent.session.session_llm import async_prompt, LLMContext, LLMConfig
from eve.agent.session.models import ChatMessage, Session

# Memory formation settings:
MEMORY_FORMATION_INTERVAL = 4 # Number of messages to wait before forming memories
MEMORY_LLM_MODEL = "gpt-4o-mini"

# Memory context assembly settings:
DEFAULT_MEMORY_TOKEN_BUDGET = 5000      # Default max tokens for memory context
DIRECTIVE_TOKEN_BUDGET_RATIO = 0.5      # Ratio of token budget for directives
SESSION_MESSAGES_LOOKBACK_LIMIT = 1000  # Max messages to look back in a session
MEMORY_SOURCE_CONTENT_TRUNCATION = 500    # Max characters for source message content display

class MemoryType(Enum):
    EPISODE = "episode"      # Conversation summaries and notable moments
    DIRECTIVE = "directive"  # User instructions, preferences, behavioral rules
    FACT = "fact"            # Important contextual information about users/projects


@Collection("agent_memories")
class AgentMemory(Document):
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
        if "memory_type" in schema and hasattr(schema["memory_type"], "value"):
            schema["memory_type"] = schema["memory_type"].value
        return schema
    
    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert string back to enum from MongoDB"""
        if "memory_type" in schema and isinstance(schema["memory_type"], str):
            schema["memory_type"] = MemoryType(schema["memory_type"])
        return schema


def get_agent_owner(agent_id: ObjectId) -> Optional[ObjectId]:
    """Get the owner of the agent"""
    try:
        from eve.agent.agent import Agent
        agent = Agent.from_mongo(agent_id)
        return agent.owner
    except Exception as e:
        print(f"Warning: Could not load agent owner for {agent_id}: {e}")
        return None


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
    
    # Use interval for recent messages to process (4 messages)
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
        print(f"LLM extracted: {extracted_data}")
        
        # Store extracted memories in database
        memories_created = store_extracted_memories(
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
Analyze this recent conversation and extract memories for the agent:

CONVERSATION:
{conversation_text}

Create consolidated memories following these rules:

1. DIRECTIVE: Create AT MOST ONE consolidated directive ONLY if there are clear, long-lasting rules, preferences, or behavioral guidelines that should be applied consistently in future interactions. If none exist, leave empty.
   
   ONLY include as directives:
   - Explicit behavioral rules or guidelines ("always ask permission before...", "never do X", "remember to Y")
   - Stable, long-term preferences that should guide future behavior consistently
   - Clear exceptions or special handling rules
   - Persistent working styles or interaction preferences
   
   DO NOT include as directives:
   - One-time requests or specific tasks ("create a story about...", "make an image of...")
   - Ad hoc instructions for the current conversation only
   - Temporary or situational commands
   - Context-specific requests that don't apply broadly
   
   - ALWAYS use specific user names from the conversation (NEVER use "User", "the user", or "Agent")
   - Example: "Gene prefers permission before generating images, wants surreal art themes consistently"
   - Counter-example (DO NOT make directive): "Gene requested a story about clockmaker" (this is a one-time request)

2. CONSOLIDATED_MEMORY: Create EXACTLY ONE factual memory that consolidates what actually happened in the conversation.
   - Record concrete facts: who did what, what was created, what tools were used, what topics were discussed
   - Include ALL requests, actions, and events - even one-time instructions go here
   - ALWAYS use specific user names and agent names from the conversation (NEVER use "User", "the user", "Agent", or "the agent")
   - Focus on actions, creations, and concrete events - avoid commentary or analysis
   - Example: "Gene requested story about clockmaker, Eve created 'The Clockmaker's Secret' featuring Elias and magical clock, added characters Azfar (camel) and Liora (sheepherder with mechanical heart), generated 5-panel comic and video using flux_schnell, Jill requested adventure with Verdelis"

Respond in JSON format:
{{
  "directive": "long-lasting rule or persistent preference that applies to future interactions, or empty string if none",
  "consolidated_memory": "factual memory of what happened, who participated, what was created"
}}

CRITICAL REQUIREMENTS: 
- BE VERY STRICT about directives - most conversations will have NO directives, only consolidated memories
- Directives are for rules that should persist across many future conversations, not one-time requests
- Focus on facts, not interpretations or commentary
- Record what actually happened, not what it means or demonstrates
- ALWAYS use actual names from the conversation - scan the conversation for "name:" patterns
- NEVER use generic terms like "User", "the user", "Agent", "the agent", "someone", "they"
- If you see "Gene:", use "Gene". If you see "Eve:", use "Eve". Always be specific.
- Avoid phrases like "highlighted", "demonstrated", "enhanced", "experience"
- Just state what was said, done, created, or discussed with specific names
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
            "facts": [],  # Facts are now part of consolidated_memory
            "episodes": [extracted_data.get("consolidated_memory", "")] if extracted_data.get("consolidated_memory", "").strip() else []
        }
        return formatted_data
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from LLM response: {response_text[:200]}...")
        extracted_data = {"directives": [], "facts": [], "episodes": []}
    
    return extracted_data


def store_extracted_memories(
    agent_id: ObjectId,
    extracted_data: Dict[str, List[str]], 
    messages: List[ChatMessage], 
    session_id: ObjectId
) -> List[AgentMemory]:
    """Store extracted memories to MongoDB with source traceability"""
    
    print(f"Storing memories for agent {agent_id}")
    print(f"Extracted data: {extracted_data}")
    
    # Ensure all messages have IDs for traceability
    message_ids = []
    for msg in messages:
        if hasattr(msg, 'id') and msg.id:
            message_ids.append(msg.id)
        else:
            # Generate consistent ID for messages without IDs
            content_hash = hashlib.md5(msg.content.encode()).hexdigest()[:24]
            fake_id = ObjectId(content_hash)
            msg.id = fake_id
            message_ids.append(fake_id)
    
    print(f"Message IDs: {[str(mid) for mid in message_ids]}")
    
    related_users = list(set([msg.sender for msg in messages if msg.sender]))
    print(f"Related users: {[str(uid) for uid in related_users]}")
    
    # Get agent owner
    agent_owner = get_agent_owner(agent_id)
    print(f"Agent owner: {agent_owner}")
    
    memories_created = []
    
    # Store directives (highest priority)
    for i, directive in enumerate(extracted_data.get("directives", [])):
        print(f"Creating directive memory {i+1}: {directive}")
        memory = AgentMemory(
            agent_id=agent_id,
            memory_type=MemoryType.DIRECTIVE,
            content=directive,
            source_session_id=session_id,
            source_message_ids=message_ids,
            related_users=related_users,
            agent_owner=agent_owner
        )
        try:
            memory.save()
            print(f"✓ Saved directive memory with ID: {memory.id}")
            memories_created.append(memory)
        except Exception as e:
            print(f"Error saving directive memory: {e}")
            traceback.print_exc()
    
    # Store facts
    for fact in extracted_data.get("facts", []):
        memory = AgentMemory(
            agent_id=agent_id,
            memory_type=MemoryType.FACT,
            content=fact,
            source_session_id=session_id,
            source_message_ids=message_ids,
            related_users=related_users,
            agent_owner=agent_owner
        )
        try:
            memory.save()
            memories_created.append(memory)
        except Exception as e:
            print(f"Error saving fact memory: {e}")
    
    # Store episodes
    for episode in extracted_data.get("episodes", []):
        memory = AgentMemory(
            agent_id=agent_id,
            memory_type=MemoryType.EPISODE,
            content=episode,
            source_session_id=session_id,
            source_message_ids=message_ids,
            related_users=related_users,
            agent_owner=agent_owner
        )
        try:
            memory.save()
            memories_created.append(memory)
        except Exception as e:
            print(f"Error saving episode memory: {e}")
    
    return memories_created


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


def assemble_memory_context(agent_id: ObjectId, token_budget: Optional[int] = None, session_id: Optional[ObjectId] = None, last_speaker_id: Optional[ObjectId] = None) -> str:
    """
    Assemble relevant memories for context injection into prompts.
    
    Args:
        agent_id: ID of the agent to get memories for
        token_budget: Maximum tokens to use for memory context (defaults to 5000).
        session_id: Current session ID to prioritize session-specific memories.
        last_speaker_id: ID of the user who spoke the last message for prioritization.
    
    Returns:
        Formatted memory context string for prompt injection.
    """
    import time
    
    if token_budget is None:
        token_budget = DEFAULT_MEMORY_TOKEN_BUDGET  # Default max memory tokens
    
    start_time = time.time()
    print(f"🧠 MEMORY ASSEMBLY PROFILING - Agent: {agent_id}")
    print(f"   Session: {session_id}, Last Speaker: {last_speaker_id}")
    print(f"   Token Budget: {token_budget}")
    
    # Allocate token budget across memory types
    directive_budget = int(token_budget * DIRECTIVE_TOKEN_BUDGET_RATIO)  # 50% for directives
    fact_episode_budget = token_budget - directive_budget  # Remainder for facts and episodes
    
    # Step 1: Database query timing
    query_start = time.time()
    try:
        # Retrieve memories with different rules for directives vs facts/episodes:
        # - Directives: ONLY from current agent
        # - Facts/Episodes: From current agent OR from the active session
        directive_query = {"agent_id": agent_id, "memory_type": "directive"}
        fact_episode_query = {
            "memory_type": {"$in": ["fact", "episode"]},
            "$or": [
                {"agent_id": agent_id},
                {"source_session_id": session_id} if session_id else {"agent_id": agent_id}
            ]
        }
        
        directive_memories = AgentMemory.find(directive_query, sort="createdAt", desc=True)
        fact_episode_memories = AgentMemory.find(fact_episode_query, sort="createdAt", desc=True)
        
        all_memories = directive_memories + fact_episode_memories
        
        query_time = time.time() - query_start
        print(f"   ⏱️  Database Query: {query_time:.3f}s ({len(directive_memories)} directives, {len(fact_episode_memories)} facts/episodes)")
    except Exception as e:
        print(f"   ❌ Error retrieving memories: {e}")
        return ""
    
    if not all_memories:
        total_time = time.time() - start_time
        print(f"   ⏱️  Total Time: {total_time:.3f}s (no memories found)")
        return ""
    
    # Step 2: Agent owner lookup timing
    owner_start = time.time()
    agent_owner = get_agent_owner(agent_id)
    owner_time = time.time() - owner_start
    print(f"   ⏱️  Agent Owner Lookup: {owner_time:.3f}s (owner: {agent_owner})")
    
    # Step 3: Memory categorization timing
    categorize_start = time.time()
    directives = [m for m in all_memories if m.memory_type == MemoryType.DIRECTIVE]
    facts = [m for m in all_memories if m.memory_type == MemoryType.FACT]
    episodes = [m for m in all_memories if m.memory_type == MemoryType.EPISODE]
    categorize_time = time.time() - categorize_start
    print(f"   ⏱️  Memory Categorization: {categorize_time:.3f}s")
    print(f"      📊 Breakdown: {len(directives)} directives, {len(facts)} facts, {len(episodes)} episodes")
    
    # Priority function for directives: prioritize by agent owner
    def directive_priority(memory):
        has_agent_owner = agent_owner and agent_owner in (memory.related_users or [])
        return (0 if has_agent_owner else 1, -memory.createdAt.timestamp())
    
    # Priority function for facts and episodes
    def fact_episode_priority(memory):
        is_agent = memory.agent_id == agent_id
        is_session = memory.source_session_id == session_id if session_id else False
        has_last_speaker = last_speaker_id and last_speaker_id in (memory.related_users or [])
        
        if is_session and is_agent and has_last_speaker:
            return (0, -memory.createdAt.timestamp())  # Highest priority
        elif is_session and is_agent:
            return (1, -memory.createdAt.timestamp())  # High priority
        elif is_session or is_agent:
            return (2, -memory.createdAt.timestamp())  # Medium priority
        else:
            return (3, -memory.createdAt.timestamp())  # Lowest priority
    
    # Step 4: Memory sorting timing
    sort_start = time.time()
    directives.sort(key=directive_priority)
    
    # Combine facts and episodes, then sort by shared priority
    facts_and_episodes = facts + episodes
    facts_and_episodes.sort(key=fact_episode_priority)
    sort_time = time.time() - sort_start
    print(f"   ⏱️  Memory Sorting: {sort_time:.3f}s")
    
    # Step 5: Context assembly timing
    assembly_start = time.time()
    context = "## Your Memory\n\n"
    
    # Directives first - behavioral rules (newer takes precedence)
    directive_count = 0
    if directives:
        context += "### Instructions & Preferences (newer takes precedence):\n"
        directive_context = ""
        for directive in directives:
            new_line = f"- {directive.content}\n"
            if estimate_tokens(directive_context + new_line) > directive_budget:
                break
            directive_context += new_line
            directive_count += 1
        context += directive_context
        context += "\n"
    
    # Facts and Episodes combined - using shared token budget but maintaining type distinction
    fact_count = 0
    episode_count = 0
    if facts_and_episodes:
        fact_episode_context = ""
        current_facts = []
        current_episodes = []
        
        # Process facts and episodes together within the shared budget
        for memory in facts_and_episodes:
            new_line = f"- {memory.content}\n"
            if estimate_tokens(fact_episode_context + new_line) > fact_episode_budget:
                break
            fact_episode_context += new_line
            
            # Track which memories we're including
            if memory.memory_type == MemoryType.FACT:
                current_facts.append(memory)
                fact_count += 1
            else:
                current_episodes.append(memory)
                episode_count += 1
        
        # Display facts section if we have any
        if current_facts:
            context += "### Known Facts:\n"
            for fact in current_facts:
                context += f"- {fact.content}\n"
            context += "\n"
        
        # Display episodes section if we have any
        if current_episodes:
            context += "### Previous Conversations:\n"
            for episode in current_episodes:
                context += f"- {episode.content}\n"
    
    assembly_time = time.time() - assembly_start
    print(f"   ⏱️  Context Assembly: {assembly_time:.3f}s")
    print(f"      📝 Included: {directive_count} directives, {fact_count} facts, {episode_count} episodes")
    
    # Step 6: Final stats
    total_time = time.time() - start_time
    final_tokens = estimate_tokens(context)
    print(f"   ⏱️  TOTAL TIME: {total_time:.3f}s")
    print(f"   📏 Context Length: {len(context)} chars (~{final_tokens} tokens)")
    print(f"   🎯 Token Budget Usage: {final_tokens}/{token_budget} ({final_tokens/token_budget*100:.1f}%)")
    
    return context


def get_memory_source_context(
    memory: AgentMemory, 
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
    """Rough token estimation (4 characters per token)"""
    return len(text) // 4

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
    
    # Check if we should form memories
    if messages_since_last < interval:
        print(f"No memory formation needed: {messages_since_last} < {interval} interval")
        return False
    
    print(f"Triggering memory formation: {messages_since_last} >= {interval} interval")
    
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
