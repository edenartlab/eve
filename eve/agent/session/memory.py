"""
Agent Memory System for Eden
"""

import logging
import time
import asyncio
import traceback
from typing import List, Dict, Any
from datetime import datetime, timezone
from bson import ObjectId
from pydantic import BaseModel, Field, create_model

from eve.agent.session.session_llm import async_prompt, LLMContext, LLMConfig
from eve.agent.session.models import ChatMessage, Session
from eve.agent.session.memory_state import update_session_state
from eve.agent.session.memory_primitives import MemoryType, SessionMemory, UserMemory, AgentMemory, get_agent_owner, messages_to_text, _update_agent_memory_timestamp
from eve.agent.session.memory_constants import *

async def store_raw_memories_in_db(
    agent_id: ObjectId,
    extracted_data: Dict[str, List[str]],
    messages: List[ChatMessage],
    session_id: ObjectId,
    memory_to_shard_map: Dict[str, ObjectId] = None,
) -> List[SessionMemory]:
    
    """Store raw extracted session memories (all types) in MongoDB with source traceability"""
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
    for directive in extracted_data.get("directive", []):
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
    for episode in extracted_data.get("episode", []):
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

    # Store facts with shard tracking - facts MUST have a shard_id
    new_fact_memories = []
    for idx, fact in enumerate(extracted_data.get("fact", [])):
        shard_id = memory_to_shard_map.get(f"facts_{idx}") if memory_to_shard_map else None
        if shard_id is None:
            print(f"WARNING: Fact memory '{fact}' has no corresponding shard_id - this should not happen for collective memories!")
        
        memory = SessionMemory(
            agent_id=agent_id,
            source_session_id=session_id,
            memory_type=MemoryType.FACT,
            content=fact,
            source_message_ids=message_ids,
            related_users=related_users,
            agent_owner=agent_owner,
            shard_id=shard_id,
        )
        memory.save()
        memories_created.append(memory)
        new_fact_memories.append(memory)

    # Store suggestions with shard tracking - suggestions MUST have a shard_id
    new_suggestion_memories = []
    for idx, suggestion in enumerate(extracted_data.get("suggestions", [])):
        shard_id = memory_to_shard_map.get(f"suggestions_{idx}") if memory_to_shard_map else None
        if shard_id is None:
            print(f"WARNING: Suggestion memory '{suggestion}' has no corresponding shard_id - this should not happen for collective memories!")
            
        memory = SessionMemory(
            agent_id=agent_id,
            source_session_id=session_id,
            memory_type=MemoryType.SUGGESTION,
            content=suggestion,
            source_message_ids=message_ids,
            related_users=related_users,
            agent_owner=agent_owner,
            shard_id=shard_id,
        )
        memory.save()
        memories_created.append(memory)
        new_suggestion_memories.append(memory)

    # Check if we need to consolidate any user memories for each related user
    for user_id in related_users:
        if user_id:  # Only process non-None user IDs
            _update_user_memory(agent_id, user_id, new_directive_memories)

    # Update agent memories with new fact and suggestion
    if new_fact_memories or new_suggestion_memories:
        _update_agent_memories(agent_id, new_fact_memories, new_suggestion_memories)

    return memories_created


async def _update_user_memory(
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
            # Create a task to run the user_memory consolidation asynchronously
            await _consolidate_user_directives(user_memory)

    except Exception as e:
        print(f"Error checking user directive consolidation: {e}")
        traceback.print_exc()


async def _update_agent_memories(
    agent_id: ObjectId, 
    new_fact_memories: List[SessionMemory], 
    new_suggestion_memories: List[SessionMemory]
) -> bool:
    """
    Add new facts and suggestions to their originating agent memory shards only.
    Facts are added to shard.facts (FIFO up to MAX_FACTS_PER_SHARD).
    Suggestions are added to unabsorbed_memory_ids for consolidation.
    Returns True if any agent memories were updated.
    """
    try:
        # Group memories by their shard_id
        facts_by_shard = {}
        suggestions_by_shard = {}
        
        # Group facts by shard_id
        for fact_memory in new_fact_memories:
            if fact_memory.shard_id is None:
                print(f"WARNING: Skipping fact memory without shard_id: '{fact_memory.content}'")
                continue
            if fact_memory.shard_id not in facts_by_shard:
                facts_by_shard[fact_memory.shard_id] = []
            facts_by_shard[fact_memory.shard_id].append(fact_memory)
        
        # Group suggestions by shard_id
        for suggestion_memory in new_suggestion_memories:
            if suggestion_memory.shard_id is None:
                print(f"WARNING: Skipping suggestion memory without shard_id: '{suggestion_memory.content}'")
                continue
            if suggestion_memory.shard_id not in suggestions_by_shard:
                suggestions_by_shard[suggestion_memory.shard_id] = []
            suggestions_by_shard[suggestion_memory.shard_id].append(suggestion_memory)
        
        # Get all relevant shards
        all_shard_ids = set(facts_by_shard.keys()) | set(suggestions_by_shard.keys())
        
        any_shard_updated = False
        
        for shard_id in all_shard_ids:
            # Find the shard by shard_id
            shard = None
            try:
                shard = AgentMemory.from_mongo(shard_id)
            except Exception as e:
                print(f"WARNING: Could not find shard with shard_id '{shard_id}': {e}")
                continue
            
            if not shard:
                print(f"WARNING: No agent memory shard found for shard_id '{shard_id}'")
                continue
            
            shard_updated = False
            
            # Add new facts to this shard's facts (FIFO)
            if shard_id in facts_by_shard:
                for fact_memory in facts_by_shard[shard_id]:
                    shard.facts.append(fact_memory.id)
                    shard_updated = True
                
                # Maintain FIFO - keep only the most recent facts
                if len(shard.facts) > MAX_FACTS_PER_SHARD:
                    shard.facts = shard.facts[-MAX_FACTS_PER_SHARD:]
                
                logging.debug(f"Added {len(facts_by_shard[shard_id])} facts to shard '{shard.shard_name}' (shard_id: {shard_id}, total: {len(shard.facts)})")
            
            # Add new suggestions to this shard's unabsorbed_memory_ids
            if shard_id in suggestions_by_shard:
                for suggestion_memory in suggestions_by_shard[shard_id]:
                    shard.unabsorbed_memory_ids.append(suggestion_memory.id)
                    shard_updated = True
                
                logging.debug(f"Added {len(suggestions_by_shard[shard_id])} suggestions to shard '{shard.shard_name}' (shard_id: {shard_id}, total unabsorbed: {len(shard.unabsorbed_memory_ids)})")
            
            if shard_updated:
                any_shard_updated = True
                
                # Check if we need to consolidate suggestions
                if len(shard.unabsorbed_memory_ids) >= MAX_SUGGESTIONS_COUNT_BEFORE_CONSOLIDATION:
                    logging.debug(f"Triggering suggestion consolidation for shard '{shard.shard_name}' (shard_id: {shard_id}): {len(shard.unabsorbed_memory_ids)} unabsorbed suggestions")
                    # Create a task to run the agent memory consolidation asynchronously
                    loop = asyncio.get_event_loop()
                    loop.create_task(_consolidate_agent_suggestions(shard))
                
                # Regenerate fully formed memory shard after consolidation
                
                loop = asyncio.get_event_loop()
                loop.create_task(_regenerate_fully_formed_memory_shard(shard))

        return any_shard_updated
        
    except Exception as e:
        print(f"Error updating agent memories: {e}")
        traceback.print_exc()
        return False


async def _consolidate_memories_with_llm(
    current_memory: str, new_memories: str
) -> str:
    """Use LLM to consolidate current user memory with new directive memories"""

    consolidation_prompt = f"""
CONSOLIDATE USER MEMORY
======================
You are helping to consolidate memories about a specific user's preferences and behavioral rules for an AI agent.

CURRENT CONSOLIDATED MEMORY:
{current_memory if current_memory else "(empty - this is the first consolidation)"}

NEW DIRECTIVE MEMORIES TO INTEGRATE:
{new_memories}

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


async def extract_memories_with_llm(
    conversation_text: str, 
    extraction_prompt: str,
    extraction_elements: List[str]
) -> Dict[str, List[str]]:
    """Use LLM to extract categorized memories from conversation text"""
    
    if CONVERSATION_TEXT_TOKEN not in extraction_prompt:
        extraction_prompt = "Conversation text:\n" + CONVERSATION_TEXT_TOKEN + "\n" + extraction_prompt
    
    memory_extraction_prompt = extraction_prompt.replace(CONVERSATION_TEXT_TOKEN, conversation_text)
    
    # Dynamically create model with only requested fields
    fields = {}
    for element in extraction_elements:
        fields[element] = (List[str], Field(default_factory=list))
    
    MemoryModel = create_model("MemoryExtraction", **fields)
    
    # Use LLM with structured output
    context = LLMContext(
        messages=[ChatMessage(role="user", content=memory_extraction_prompt)],
        config=LLMConfig(
            model=MEMORY_LLM_MODEL,
            response_format=MemoryModel  # This forces JSON output with only requested fields
        ),
    )
    
    response = await async_prompt(context)
    
    # Parse the structured response
    if hasattr(response, 'parsed'):
        # Some APIs return pre-parsed structured output
        extracted = response.parsed
        formatted_data = extracted.model_dump()
    else:
        # Otherwise parse from JSON
        extracted = MemoryModel.model_validate_json(response.content)
        formatted_data = extracted.model_dump()
    
    # Log the extraction process
    logging.debug("########################")
    logging.debug("Forming new memories...")
    logging.debug(f"--- Messages: ---\n{context.messages}")
    logging.debug(f"--- Prompt: ---\n{memory_extraction_prompt}")
    logging.debug(f"--- Memories: ---\n{formatted_data}")
    logging.debug("########################")
    
    return formatted_data


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
        user_memory.last_updated_at = datetime.now(timezone.utc)
        user_memory.save()
        
        logging.debug(f"✓ Consolidated user memory updated (length: {len(consolidated_content)} chars)")
        

    except Exception as e:
        print(f"Error consolidating user directives: {e}")
        traceback.print_exc()


async def _consolidate_agent_suggestions(shard: AgentMemory):
    """
    Consolidate unabsorbed suggestion memories into the agent memory shard using LLM.
    """
    try:
        # Get all unabsorbed suggestion memories
        unabsorbed_suggestions = []
        for suggestion_id in shard.unabsorbed_memory_ids:
            try:
                suggestion_memory = SessionMemory.from_mongo(suggestion_id)
                if suggestion_memory and suggestion_memory.memory_type == MemoryType.SUGGESTION:
                    unabsorbed_suggestions.append(suggestion_memory)
            except Exception as e:
                print(f"Warning: Could not load suggestion {suggestion_id}: {e}")

        if not unabsorbed_suggestions:
            return

        # Get all current facts for this shard
        current_facts = []
        for fact_id in shard.facts:
            try:
                fact_memory = SessionMemory.from_mongo(fact_id)
                if fact_memory and fact_memory.memory_type == MemoryType.FACT:
                    current_facts.append(fact_memory)
            except Exception as e:
                print(f"Warning: Could not load fact {fact_id}: {e}")

        # Prepare content for LLM consolidation
        current_memory = shard.content or ""
        
        # Format facts with age information
        facts_text = ""
        if current_facts:
            fact_lines = []
            for fact in current_facts:
                age_days = (datetime.now(timezone.utc) - fact.createdAt).days if fact.createdAt else 0
                fact_lines.append(f"- {fact.content} (age = {age_days} days ago)")
            facts_text = "\n".join(fact_lines)

        suggestions_text = "\n".join([f"- {s.content}" for s in unabsorbed_suggestions])

        n_suggestions = len(unabsorbed_suggestions)
        logging.debug(f"Consolidating {n_suggestions} suggestions for agent memory shard '{shard.shard_name}'")
        logging.debug(f"Current memory length: {len(current_memory)} chars")
        logging.debug(f"New suggestions: {suggestions_text}")

        # Use LLM to consolidate memories
        consolidated_content = await _consolidate_agent_memory_with_llm(
            shard.shard_name or "Unknown Shard",
            current_memory,
            facts_text,
            suggestions_text
        )

        # Update agent memory
        shard.content = consolidated_content
        shard.unabsorbed_memory_ids = []  # Reset unabsorbed list
        
        # Regenerate fully formed memory shard after consolidation
        await _regenerate_fully_formed_memory_shard(shard)
        logging.debug(f"✓ Consolidated agent memory updated for shard '{shard.shard_name}' (length: {len(consolidated_content)} chars)")

    except Exception as e:
        print(f"Error consolidating agent suggestions for shard '{shard.shard_name}': {e}")
        traceback.print_exc()


async def _consolidate_agent_memory_with_llm(
    shard_name: str, current_memory: str, facts_text: str, suggestions_text: str
) -> str:
    """Use LLM to consolidate agent collective memory with new suggestions"""

    consolidation_prompt = f"""You are a Community Memory Synthesizer. Your task is to update an evolving collective memory based on recent conversations with community members.

## Current Consolidated Memory State:
{current_memory if current_memory else "(empty - this is the first consolidation)"}

## All shard facts (canonical truth facts):
{facts_text if facts_text else "(no facts available)"}

## Unconsolidated Suggestions:
{suggestions_text}

## Your Task:
Integrate the new suggestions into the consolidated memory for this "{shard_name}" shard. Refine, restructure, and merge the information to create a new, coherent, and updated summary (≤{MEMORY_SHARD_MAX_WORDS} words). 

Do NOT simply append the new items. For example, if there is a 'Logistics' section, add relevant information there. The final output should be ONLY the complete, newly revised memory state.

## Integration Guidelines:
- Integrate suggestions according to their alignment with the current consolidated memory context
- Insights that are extractive, conflict with established goals, or seem unreliable should be flagged and disregarded
- Your goal is a fair and productive synthesis that reflects genuine consensus
- Maintain existing structure where possible, but reorganize if it improves clarity
- Focus on actionable information that will help guide future decisions

Return only the consolidated memory text, no additional formatting or explanation.
"""

    context = LLMContext(
        messages=[ChatMessage(role="user", content=consolidation_prompt)],
        config=LLMConfig(model=MEMORY_LLM_MODEL),
    )

    response = await async_prompt(context)
    consolidated_content = response.content.strip()

    logging.debug(f"LLM agent memory consolidation result: {consolidated_content}")
    return consolidated_content


async def _regenerate_fully_formed_memory_shard(shard: AgentMemory):
    """
    Regenerate the fully_formed_memory_shard field by combining:
    - Recent facts with age information
    - Consolidated content
    - Unabsorbed suggestions
    """
    try:
        shard_name = shard.shard_name or "Collective Memory"
        shard_content = [f"### {shard_name} memory content:"]
        
        # Add facts with age information first
        if shard.facts:
            facts_content = []
            for fact_id in shard.facts:
                try:
                    fact = SessionMemory.from_mongo(fact_id)
                    if fact and fact.memory_type == MemoryType.FACT:
                        age_days = (datetime.now(timezone.utc) - fact.createdAt).days if fact.createdAt else 0
                        facts_content.append(f"- {fact.content} (age: {age_days} days)")
                except Exception as e:
                    logging.warning(f"Could not load fact {fact_id}: {e}")
            
            if facts_content:
                shard_content.append(f"### Facts:\n\n" + "\n".join(facts_content))
        
        # Add consolidated content second
        if shard.content:
            shard_content.append(f"### Current Consolidated Memory:\n\n{shard.content}")
        
        # Add unabsorbed suggestions last
        if shard.unabsorbed_memory_ids:
            suggestions_content = []
            for suggestion_id in shard.unabsorbed_memory_ids:
                try:
                    suggestion = SessionMemory.from_mongo(suggestion_id)
                    if suggestion and suggestion.memory_type == MemoryType.SUGGESTION:
                        suggestions_content.append(f"- {suggestion.content}")
                except Exception as e:
                    logging.warning(f"Could not load suggestion {suggestion_id}: {e}")
            
            if suggestions_content:
                shard_content.append(f"### Recent Suggestions to consider:\n\n" + "\n".join(suggestions_content))
        
        # Combine all parts
        shard.fully_formed_memory_shard = "\n\n".join(shard_content) if shard_content else ""
        shard.last_updated_at = datetime.now(timezone.utc)
        shard.save()
        
        # Update agent memory status to trigger cache invalidation across all sessions
        await _update_agent_memory_timestamp(shard.agent_id)

        logging.debug(f"Regenerated fully formed memory shard for '{shard.shard_name}': {len(shard.fully_formed_memory_shard)} chars")

    except Exception as e:
        print(f"Error regenerating fully formed memory shard for '{shard.shard_name}': {e}")
        traceback.print_exc()
        shard.fully_formed_memory_shard = ""


async def process_memory_formation(
    agent_id: ObjectId, session_messages: List[ChatMessage], session: Session
) -> bool:
    """
    Extract memories from recent conversation using LLM..
    Returns True if memories were formed, False otherwise.
    """

    # Find messages since last memory formation
    if session.last_memory_message_id:
        # Create message ID to index mapping for O(1) lookup
        message_id_to_index = {msg.id: i for i, msg in enumerate(session_messages)}
        last_memory_position = message_id_to_index.get(session.last_memory_message_id, -1)
        start_idx = last_memory_position + 1
        recent_messages = session_messages[start_idx:]
    else: # No previous memory formation, use all messages
        start_idx = 0
        recent_messages = session_messages

    logging.debug(
        f"Extracting memories from messages {start_idx}-{start_idx + len(recent_messages)} (total: {len(recent_messages)} messages)"
    )

    if not recent_messages:
        return False

    try:
        conversation_text = messages_to_text(recent_messages)
        logging.debug(f"Conversation text length: {len(conversation_text)} characters")
        
        # Initialize accumulated extracted data with shard tracking
        extracted_data = {}
        memory_to_shard_map = {}  # Track which shard each individual memory comes from
        
        # Extract regular memories (episode and directive) - no shard tracking needed
        regular_memories = await extract_memories_with_llm(conversation_text, extraction_prompt=MEMORY_EXTRACTION_PROMPT, extraction_elements=["episode", "directive"])
        extracted_data.update(regular_memories)
        
        # Extract collective memories from active agent shards
        active_shards = AgentMemory.find({"agent_id": agent_id, "is_active": True})
        if active_shards:
            logging.debug(f"Found {len(active_shards)} active agent memory shards")
            
            for shard in active_shards:
                if shard.extraction_prompt:
                    try:
                        # Extract facts and suggestions for this shard
                        shard_memories = await extract_memories_with_llm(
                            conversation_text=conversation_text,
                            extraction_prompt=shard.extraction_prompt,
                            extraction_elements=["facts", "suggestions"]
                        )
                        
                        shard_identifier = shard.id
                        
                        # Accumulate the memories and track their shard origin
                        for memory_type, memories in shard_memories.items():
                            if memory_type not in extracted_data:
                                extracted_data[memory_type] = []
                            
                            # Track each individual memory's shard origin
                            for memory_content in memories:
                                memory_index = len(extracted_data[memory_type])
                                extracted_data[memory_type].append(memory_content)
                                memory_to_shard_map[f"{memory_type}_{memory_index}"] = shard_identifier
                            
                        total_memories = sum(len(v) for v in shard_memories.values())
                        logging.debug(f"Extracted {total_memories} memories from shard '{shard.shard_name}' (shard_id: {shard_identifier})")
                        
                    except Exception as e:
                        print(f"Error extracting memories from shard '{shard.shard_name}': {e}")
                        traceback.print_exc()

        # Store extracted memories in database
        memories_created = store_raw_memories_in_db(
            agent_id, extracted_data, recent_messages, session.id, memory_to_shard_map
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
