from eve.agent.session.memory_models import SessionMemory, UserMemory, AgentMemory
from eve.agent.session.memory_state import update_session_state
from eve.agent.session.memory_constants import MAX_N_EPISODES_TO_REMEMBER, LOCAL_DEV

import time, logging, asyncio
from bson import ObjectId
from typing import Optional
from datetime import datetime, timezone

async def _assemble_user_memory(agent_id: ObjectId, user_id: Optional[ObjectId] = None) -> tuple[str, list]:
    """
    Step 1: Assemble user memory content and unabsorbed directives.
    Returns user_memory_content
    """
    user_memory_content = ""
    user_memory = None

    try:
        query_start = time.time()
        if (user_id is not None and agent_id is not None):
            user_memory = UserMemory.find_one_or_create(
                {"agent_id": agent_id, "user_id": user_id}
            )
            if user_memory:
                # Check if fully_formed_memory exists and is up-to-date
                if user_memory.fully_formed_memory and user_memory.fully_formed_memory.strip():
                    # Use the pre-formed memory if available
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


async def _assemble_episode_memories(agent_id: ObjectId, session_id: Optional[ObjectId] = None) -> list:
    """
    Step 2: Assemble episode memories from SessionMemory.
    Returns list of episode memories.
    """
    episode_memories = []
    
    try:
        if session_id and agent_id is not None:
            query_start = time.time()
            episode_query = {"source_session_id": session_id, "memory_type": "episode"}
            episode_memories = SessionMemory.find(episode_query, sort="createdAt", desc=True)

            # Get list of MAX_N_EPISODES_TO_REMEMBER most recent, raw episode memories:
            episode_memories = episode_memories[:MAX_N_EPISODES_TO_REMEMBER]
            # Reverse the list to put the most recent episodes at the bottom:
            episode_memories.reverse()

            query_time = time.time() - query_start
            print(f"   ⏱️  Episode memory assembly: {query_time:.3f}s ({len(episode_memories)} episodes)")
    except Exception as e:
        print(f"   ❌ Error assembling episode memories: {e}")

    return episode_memories


async def _assemble_agent_collective_memories(agent_id: ObjectId) -> list:
    """
    Step 3: Assemble agent collective memories from active shards.
    Returns list of memory shards with name and content.
    """
    agent_collective_memories = []
    
    try:
        if agent_id is not None:
            query_start = time.time()
            # Get active agent memory shards
            active_shards = AgentMemory.find({"agent_id": agent_id, "is_active": True})
            print(f"   --> found {len(active_shards)} active agent memory shards")
            
            for shard in active_shards:
                # Check if fully_formed_memory_shard exists and is non-empty
                if shard.fully_formed_memory_shard and shard.fully_formed_memory_shard.strip():
                    agent_collective_memories.append({
                        'name': shard.shard_name or 'unnamed_shard',
                        'content': shard.fully_formed_memory_shard
                    })
                else:
                    # Import here to avoid circular imports
                    from eve.agent.session.memory import _regenerate_fully_formed_memory_shard
                    await _regenerate_fully_formed_memory_shard(shard)
                    
                    # Use the regenerated content if it's non-empty
                    if shard.fully_formed_memory_shard and shard.fully_formed_memory_shard.strip():
                        agent_collective_memories.append({
                            'name': shard.shard_name or 'unnamed_shard',
                            'content': shard.fully_formed_memory_shard
                        })
            
            query_time = time.time() - query_start
            print(f"   ⏱️  Agent Memory Assembly: {query_time:.3f}s ({len(agent_collective_memories)} active shards)")
            
    except Exception as e:
        print(f"   ❌ Error assembling agent collective memories: {e}")

    return agent_collective_memories


async def regenerate_memory_context(agent_id: ObjectId, session_id: Optional[ObjectId] = None, last_speaker_id: Optional[ObjectId] = None, session: Optional['Session'] = None) -> str:
    """
    Regenerate memory context by querying the database and assembling all memory components.
    
    Args:
        agent_id: ID of the agent to get memories for
        session_id: Current session ID to prioritize session-specific memories.
        last_speaker_id: ID of the user who spoke the last message for prioritization.
        session: Optional session object to update in place (avoids extra MongoDB query).
    
    Returns:
        Formatted memory context string for prompt injection.
    """
    start_time = time.time()
    
    # TODO: instead of last speaker, iterate over all human users in the session
    user_id = last_speaker_id

    # Step 1: User Memory
    user_memory_content = await _assemble_user_memory(agent_id, user_id)

    # Step 2: Episodes from SessionMemory
    episode_memories = await _assemble_episode_memories(agent_id, session_id)

    # Step 3: Agent Collective Memories
    agent_collective_memories = await _assemble_agent_collective_memories(agent_id)

    # Step 4: Full memory context assembly:
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

    # Build episode memories section (outside user_specific_memory)
    episode_memory_section = ""
    if len(episode_memories) > 0:
        episode_memory_section += "<current_conversation_context description=\"Recent exchanges from this conversation (most recent at bottom)\">\n"
        for episode in episode_memories:
            episode_memory_section += f"- {episode.content}\n"
        episode_memory_section += "</current_conversation_context>\n\n"

    # Assemble final memory context with XML hierarchy
    if collective_memory_section or user_memory_section or episode_memory_section:
        memory_context = "<memory_contents description=\"Your complete memory context for this conversation\">\n\n"
        
        if collective_memory_section:
            memory_context += collective_memory_section
            
        if user_memory_section:
            memory_context += user_memory_section
        
        # Add episode memories outside user_memory but inside memory_contents
        if episode_memory_section:
            memory_context += episode_memory_section
            
        memory_context += "</memory_contents>"
    else:
        memory_context = ""
    
    # Step 5: Cache the full memory context in modal dict (non-blocking)
    if session_id and agent_id:
        async def cache_memory_context():
            try:
                cache_start = time.time()
                current_time = datetime.now(timezone.utc).isoformat()
                
                await update_session_state(agent_id, session_id, {
                    "cached_memory_context": memory_context,
                    "should_refresh_memory": False,
                    "agent_collective_memory_timestamp": current_time,
                    "user_memory_timestamp": current_time
                })
                print(f"   💾 Memory context cached for session {session_id} in {time.time() - cache_start:.3f}s")
            except Exception as e:
                print(f"   ❌ Error caching memory context: {e}")
        
        # Start the caching task in the background (non-blocking)
        asyncio.create_task(cache_memory_context())
    
    # Step 6: Final stats
    total_time = time.time() - start_time
    print(f"   ⏱️  Memory context full regeneration time: {total_time:.3f}s")

    return memory_context


async def try_use_cached_memory_context(agent_id: ObjectId, session_id: ObjectId, last_speaker_id: Optional[ObjectId] = None) -> Optional[str]:
    """
    Attempt to use cached memory context from modal dict.
    Returns the cached context if valid, None if refresh is needed.
    
    Args:
        agent_id: ID of the agent
        session_id: Current session ID
        last_speaker_id: ID of the user who spoke the last message
    
    Returns:
        Cached memory context string if valid, None if refresh needed
    """
    try:
        get_session_state_start = time.time()
        
        from eve.agent.session.memory_state import memory_state_manager, DEFAULT_SESSION_STATE
        
        # Use optimized memory context data fetching
        memory_data = await memory_state_manager.get_memory_context_data(
            agent_id=agent_id,
            session_id=session_id,
            user_id=last_speaker_id
        )
        
        local_session_state = memory_data["session_state"]
        local_agent_memory_status = memory_data["agent_memory_timestamp"]
        local_user_memory_status = memory_data["user_memory_timestamp"]
        
        if memory_data["session_created"]:
            print(f"No session state found for session {session_id}, created default session state")
        
        get_session_state_time = time.time() - get_session_state_start
        print(f"   ⏱️  batch modal.Dict fetch took: {get_session_state_time:.3f}s")

        # Work with local copies from here - no more network calls to modal.Dict
        cached_context = local_session_state.get("cached_memory_context")
        should_refresh = local_session_state.get("should_refresh_memory", True)
        
        # Check if agent collective memory has been updated since last fetch
        agent_memory_updated = False
        
        if local_agent_memory_status:
            agent_last_updated = local_agent_memory_status
            session_last_fetched = local_session_state.get("agent_collective_memory_timestamp")
            
            if agent_last_updated and session_last_fetched:
                # Convert to datetime for comparison
                try:
                    agent_updated_dt = datetime.fromisoformat(agent_last_updated.replace('Z', '+00:00'))
                    session_fetched_dt = datetime.fromisoformat(session_last_fetched.replace('Z', '+00:00'))
                    agent_memory_updated = agent_updated_dt > session_fetched_dt
                except Exception as e:
                    agent_memory_updated = True
            else:
                # If either timestamp is missing, refresh to be safe
                agent_memory_updated = True

        
        # Check if user memory has been updated since last fetch
        user_memory_updated = False
        
        # Only check user memory if we have a cache to potentially use
        if cached_context and not should_refresh and not agent_memory_updated:
            if last_speaker_id and local_user_memory_status:
                start_probing_user_memory = time.time()
                try:
                    user_last_updated = local_user_memory_status
                    session_user_memory_timestamp = local_session_state.get("user_memory_timestamp")
                    
                    if user_last_updated and session_user_memory_timestamp:
                        try:
                            user_updated_dt = datetime.fromisoformat(user_last_updated.replace('Z', '+00:00'))
                            session_user_fetched_dt = datetime.fromisoformat(session_user_memory_timestamp.replace('Z', '+00:00'))
                            user_memory_updated = user_updated_dt > session_user_fetched_dt
                        except Exception as e:
                            user_memory_updated = True
                    elif user_last_updated and not session_user_memory_timestamp:
                        # If user memory exists but session has no timestamp, refresh
                        user_memory_updated = True
                    else:
                        # If no user memory exists or no timestamp, don't force refresh
                        user_memory_updated = False

                except Exception as e:
                    logging.warning(f"Error checking user memory timestamp: {e}")
                    user_memory_updated = True

                probing_user_memory_time = time.time() - start_probing_user_memory
                print(f"   ⏱️  probing user memory took: {probing_user_memory_time:.3f}s")
            else:
                user_memory_updated = False
                print(f"   ⏱️  probing user memory took: 0.000s (skipped - no last_speaker_id or user memory data)")
        else:
            user_memory_updated = True  # Force refresh if no cache or other refresh reasons
            print(f"   ⏱️  probing user memory took: 0.000s (skipped - cache invalid)")
            
        if cached_context and not should_refresh and not agent_memory_updated and not user_memory_updated:
            return cached_context
        else:
            refresh_reasons = []
            if not cached_context:
                refresh_reasons.append("no_cache")
            if should_refresh:
                refresh_reasons.append("should_refresh_flag")
            if agent_memory_updated:
                refresh_reasons.append("agent_memory_updated")
            if user_memory_updated:
                refresh_reasons.append("user_memory_updated")
            print(f"   🔄 Memory context refresh needed: {', '.join(refresh_reasons)}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error checking cached memory: {e}")
        return None


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
    
    print(f"\n🧠 MEMORY ASSEMBLY PROFILING - Agent: {agent_id}")
    print(f"   Session: {session_id}, Last Speaker: {last_speaker_id}")
    
    # Check if we can use cached memory context from modal dict
    if session_id and agent_id:
        cached_context = await try_use_cached_memory_context(agent_id, session_id, last_speaker_id)
        
        if cached_context:
            total_time = time.time() - start_time
            print(f"   ⚡ USING CACHED MEMORY: {total_time:.3f}s")
            time_taken = time.time() - start_time
            print(f"-----> Time taken to assemble memory context: {time_taken:.2f} seconds")
            if LOCAL_DEV:
                print(f"\n\n------------- Fully Assembled Memory context: --------------\n{cached_context}")
                print("-----------------------------------------------------------\n\n")
            return cached_context

    memory_context = await regenerate_memory_context(agent_id, session_id, last_speaker_id, session)

    time_taken = time.time() - start_time
    print(f"-----> Time taken to assemble memory context: {time_taken:.2f} seconds")

    if LOCAL_DEV:
        print(f"\n\n--- Fully Assembled Memory context: ---\n{memory_context}")
        print("----------------------------------------\n\n")

    return memory_context