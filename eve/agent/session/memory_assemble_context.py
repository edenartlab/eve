from eve.agent.session.memory_models import SessionMemory, UserMemory, AgentMemory
from eve.agent.session.memory_constants import MAX_N_EPISODES_TO_REMEMBER, LOCAL_DEV, SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES

import time, logging
from bson import ObjectId
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from eve.agent.session.models import Session

async def _assemble_user_memory(agent_id: ObjectId, user_id: Optional[ObjectId] = None) -> str:
    """
    Step 1: Assemble user memory content.
    Returns user_memory_content
    """
    user_memory_content = ""
    user_memory = None

    try:
        query_start = time.time()
        if user_id is not None and agent_id is not None:
            user_memory = UserMemory.find_one_or_create(
                {"agent_id": agent_id, "user_id": user_id}
            )
            if user_memory:
                # Check if fully_formed_memory exists and is up-to-date
                if user_memory.fully_formed_memory:
                    user_memory_content = user_memory.fully_formed_memory
                else:
                    # Regenerate fully formed memory if missing or empty
                    from eve.agent.session.memory import _regenerate_fully_formed_user_memory
                    await _regenerate_fully_formed_user_memory(user_memory)
                    user_memory_content = user_memory.fully_formed_memory or ""
                
        query_time = time.time() - query_start
        print(f"   ⏱️  User Memory Assembly: {query_time:.3f}s (user_memory: {'yes' if user_memory else 'no'})")

    except Exception as e:
        print(f"   ❌ Error retrieving user memory: {e}")

    return user_memory_content


async def _get_episode_memories(session: Session, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Get episode memories with smart caching in session.
    Returns list of episode memory dicts.
    """
    # Check if we have cached episodes:
    if (not force_refresh and 
        session.memory_context.cached_episode_memories):
        print(f"   ⚡ Using cached episode memories ({len(session.memory_context.cached_episode_memories)} episodes)")
        return session.memory_context.cached_episode_memories
    
    # Query and cache episode memories
    try:
        query_start = time.time()
        episode_query = {"source_session_id": session.id, "memory_type": "episode"}
        episode_memories = SessionMemory.find(episode_query, sort="createdAt", desc=True)
        
        # Get list of MAX_N_EPISODES_TO_REMEMBER most recent episodes
        episode_memories = episode_memories[:MAX_N_EPISODES_TO_REMEMBER]
        # Reverse to put most recent at bottom
        episode_memories.reverse()
        
        # Cache in session
        session.memory_context.cached_episode_memories = [
            {
                "id": str(e.id), 
                "content": e.content, 
                "created_at": e.createdAt.isoformat() if e.createdAt else None
            }
            for e in episode_memories
        ]
        session.memory_context.episode_memories_timestamp = datetime.now(timezone.utc)
        # Save will be done by caller to batch updates
        query_time = time.time() - query_start
        print(f"   ⏱️  Episode memory query & cache: {query_time:.3f}s ({len(episode_memories)} episodes)")
        
        return session.memory_context.cached_episode_memories
        
    except Exception as e:
        print(f"   ❌ Error assembling episode memories: {e}")
        return []


async def _assemble_agent_memories(agent_id: ObjectId) -> List[Dict[str, str]]:
    """
    Step 3: Assemble agent collective memories from active shards.
    Returns list of memory shards with name and content.
    """
    agent_collective_memories = []
    
    try:
        query_start = time.time()
        active_shards = AgentMemory.find({"agent_id": agent_id, "is_active": True})
        print(f"   --> found {len(active_shards)} active agent memory shards")
        
        for shard in active_shards:
            # Check if fully_formed_memory exists and is non-empty
            if shard.fully_formed_memory and shard.fully_formed_memory.strip():
                agent_collective_memories.append({
                    'name': shard.shard_name or 'unnamed_shard',
                    'content': shard.fully_formed_memory
                })
            else:
                # Import here to avoid circular imports
                from eve.agent.session.memory import _regenerate_fully_formed_agent_memory
                await _regenerate_fully_formed_agent_memory(shard)
                
                # Use the regenerated content if it's non-empty
                if shard.fully_formed_memory and shard.fully_formed_memory.strip():
                    agent_collective_memories.append({
                        'name': shard.shard_name or 'unnamed_shard',
                        'content': shard.fully_formed_memory
                    })
        
        query_time = time.time() - query_start
        print(f"   ⏱️  Agent Memory Assembly: {query_time:.3f}s ({len(agent_collective_memories)} active shards)")
        
    except Exception as e:
        print(f"   ❌ Error assembling agent collective memories: {e}")

    return agent_collective_memories


async def check_memory_freshness(session: Session, agent_id: ObjectId, user_id: Optional[ObjectId]) -> bool:
    """
    Check if cached memory context is still fresh by comparing timestamps.
    Returns True if cache is fresh, False if needs refresh.
    """
    # Check agent memory freshness
    if session.memory_context.agent_memory_timestamp:
        try:
            # Query only the most recently updated active agent memory
            agent_memory = AgentMemory.find_one(
                {"agent_id": agent_id, "is_active": True},
                sort=[("last_updated_at", -1)]
            )
            if agent_memory and agent_memory.last_updated_at:
                if agent_memory.last_updated_at > session.memory_context.agent_memory_timestamp:
                    print(f"   🔄 Agent memory updated since cache")
                    return False
        except Exception as e:
            print(f"   ⚠️ Error checking agent memory freshness: {e}")
            return False  # Refresh on error to be safe
    
    # Check user memory freshness
    if user_id and session.memory_context.user_memory_timestamp:
        try:
            user_memory = UserMemory.find_one(
                {"agent_id": agent_id, "user_id": user_id}
            )
            if user_memory and user_memory.last_updated_at:
                if user_memory.last_updated_at > session.memory_context.user_memory_timestamp:
                    print(f"   🔄 User memory updated since cache")
                    return False
        except Exception as e:
            print(f"   ⚠️ Error checking user memory freshness: {e}")
            return False  # Refresh on error to be safe
    
    return True  # Cache is fresh


def _build_memory_xml(
    user_memory_content: str,
    agent_collective_memories: List[Dict[str, str]],
    episode_memories: List[Dict[str, Any]]
) -> str:
    """Build the formatted XML memory context."""
    memory_context = ""
    
    # Build collective memory section first
    collective_memory_section = ""
    if len(agent_collective_memories) > 0:
        collective_memory_section += "<collective_memory description=\"Shared memory across all your conversations\">\n"
        for shard in agent_collective_memories:
            shard_name = shard['name']
            shard_content = shard['content']
            collective_memory_section += f"<memory_shard name=\"{shard_name}\">\n{shard_content}\n</memory_shard>\n\n"
        collective_memory_section += "</collective_memory>\n\n"
    
    # Build user-specific memory section
    user_memory_section = ""
    if user_memory_content and user_memory_content.strip():
        user_memory_section += f"<user_memory description=\"Memory and context specific to this user\">\n{user_memory_content}\n</user_memory>\n\n"
    
    # Build episode memories section
    episode_memory_section = ""
    if len(episode_memories) > 0:
        episode_memory_section += "<current_conversation_context description=\"Recent exchanges from this conversation (most recent at bottom)\">\n"
        for episode in episode_memories:
            episode_memory_section += f"- {episode['content']}\n"
        episode_memory_section += "</current_conversation_context>\n\n"
    
    # Assemble final memory context with XML hierarchy
    if collective_memory_section or user_memory_section or episode_memory_section:
        memory_context = "<memory_contents description=\"Your complete memory context for this conversation\">\n\n"
        
        if collective_memory_section:
            memory_context += collective_memory_section
            
        if user_memory_section:
            memory_context += user_memory_section
        
        if episode_memory_section:
            memory_context += episode_memory_section
            
        memory_context += "</memory_contents>"
    
    return memory_context


async def assemble_memory_context(
    session: Session,
    agent_id: ObjectId,
    last_speaker_id: Optional[ObjectId] = None, 
    force_refresh: bool = False,
    reason: str = "unknown",
    skip_save: bool = False
) -> str:
    """
    Assemble relevant memories for context injection into prompts.
    Uses session-level caching to minimize database queries.
    
    Args:
        agent_id: ID of the agent to get memories for
        session_id: Current session ID (for compatibility, prefer passing session object)
        last_speaker_id: ID of the user who spoke the last message
        session: Session object containing memory context
    
    Returns:
        Formatted memory context string for prompt injection.
    """
    start_time = time.time()
    
    print(f"\n🧠 MEMORY ASSEMBLY - Agent: {agent_id}, Session: {session.id}")
    
    # Check if we can use cached memory context
    if (session.memory_context.cached_memory_context is not None)and not force_refresh:
        memory_timestamp = session.memory_context.memory_context_timestamp
        if memory_timestamp and memory_timestamp.tzinfo is None:
            memory_timestamp = memory_timestamp.replace(tzinfo=timezone.utc)
        
        time_since_context_refresh = datetime.now(timezone.utc) - (memory_timestamp or datetime.min.replace(tzinfo=timezone.utc))
        
        if time_since_context_refresh < timedelta(minutes=SYNC_MEMORIES_ACROSS_SESSIONS_EVERY_N_MINUTES):
            # Optional: Check if memories were updated by other sessions
            if 1: #or await check_memory_freshness(session, agent_id, last_speaker_id):
                print(f"   ⚡ Using cached memory context ({time.time() - start_time:.3f}s)")
                if LOCAL_DEV:
                    print(f"\n\n------------- Cached Memory Context --------------\n{session.memory_context.cached_memory_context}")
                    print("-----------------------------------------------------------\n\n")
                return session.memory_context.cached_memory_context
        else:
            reason = "syncing_session_memories"
    
    print(f"   🔄 Rebuilding memory context... Reason: {reason}")
    
    # Rebuild memory context
    # 1. Get user memory (1 query)
    user_memory_content = await _assemble_user_memory(agent_id, last_speaker_id)
    
    # 2. Get agent memories (1 query)
    agent_collective_memories = await _assemble_agent_memories(agent_id)
    
    # 3. Get episode memories (0-1 queries with caching)
    episode_memories = await _get_episode_memories(session, force_refresh=force_refresh)
    
    # 4. Build XML context
    memory_context = _build_memory_xml(user_memory_content, agent_collective_memories, episode_memories)
    
    # 5. Update session with cached context
    current_time = datetime.now(timezone.utc)
    session.memory_context.cached_memory_context = memory_context
    session.memory_context.memory_context_timestamp = current_time
    session.memory_context.agent_memory_timestamp = current_time
    session.memory_context.user_memory_timestamp = current_time
    
    if not skip_save:
        session.save()
    
    total_time = time.time() - start_time
    print(f"   ✓ Memory context rebuilt and cached ({total_time:.3f}s)")
    
    if LOCAL_DEV:
        print(f"\n\n------------- Rebuilt Memory Context --------------\n{memory_context}")
        print("-----------------------------------------------------------\n\n")
    
    return memory_context