from eve.agent.session.memory_primitives import MemoryType, SessionMemory, UserMemory, estimate_tokens
from eve.agent.session.memory_state import get_session_state, update_session_state
from eve.agent.session.memory import MAX_N_EPISODES_TO_REMEMBER, LOCAL_DEV

import time, logging
from bson import ObjectId
from typing import Optional

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