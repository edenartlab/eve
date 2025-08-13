"""
Agent Memory System for Eden - Refactored Version
"""

import logging
import time
import traceback
import uuid
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
from bson import ObjectId
from pydantic import Field, create_model

from eve.agent.session.session_llm import async_prompt, LLMContext, LLMConfig
from eve.agent.session.models import ChatMessage, Session, LLMContextMetadata, LLMTraceMetadata
from eve.agent.session.memory_models import SessionMemory, UserMemory, AgentMemory, get_agent_owner, messages_to_text, _get_recent_messages, _format_memories_with_age, estimate_tokens
from eve.agent.session.memory_constants import *

async def _store_memories_by_type(
    agent_id: ObjectId,
    session_id: ObjectId,
    extracted_data: Dict[str, List[str]],
    message_ids: List[ObjectId],
    related_users: List[ObjectId],
    agent_owner: ObjectId,
    memory_to_shard_map: Dict[str, ObjectId] = None
) -> Dict[str, List[SessionMemory]]:
    """Store memories organized by type."""
    memories_by_type = {}
    
    # Generate memory configs from MEMORY_TYPES
    # Facts and suggestions are collective memory (require shards), episodes and directives are personal
    collective_memory_types = {"fact", "suggestion"}
    
    for key, memory_type in MEMORY_TYPES.items():
        requires_shard = key in collective_memory_types
        memories = []
        for idx, content in enumerate(extracted_data.get(key, [])):
            shard_id = None
            if requires_shard:
                shard_id = memory_to_shard_map.get(f"{key}_{idx}") if memory_to_shard_map else None
                if shard_id is None:
                    logging.warning(
                        f"{memory_type.name.capitalize()} memory '{content}' has no shard_id"
                    )
            
            memory = SessionMemory(
                agent_id=agent_id,
                source_session_id=session_id,
                memory_type=memory_type.name,
                content=content,
                source_message_ids=message_ids,
                related_users=related_users,
                agent_owner=agent_owner,
                shard_id=shard_id,
            )
            memory.save()
            memories.append(memory)
        
        if memories:
            memories_by_type[memory_type.name] = memories
    
    return memories_by_type


async def _save_all_memories(
    agent_id: ObjectId,
    extracted_data: Dict[str, List[str]],
    messages: List[ChatMessage],
    session: Session,
    memory_to_shard_map: Dict[str, ObjectId] = None,
) -> List[SessionMemory]:
    """Store raw extracted session memories (all types) in MongoDB with source traceability"""
    
    # Prepare common data
    message_ids = [msg.id for msg in messages]
    related_users = list(
        set([msg.sender for msg in messages if msg.sender and msg.sender != agent_id])
    )
    agent_owner = get_agent_owner(agent_id)
    session_id = session.id

    # Store memories by type
    memories_by_type = await _store_memories_by_type(
        agent_id, session_id, extracted_data, message_ids,
        related_users, agent_owner, memory_to_shard_map
    )
    
    # Update user memories with new directives
    for user_id in related_users:
        if user_id:
            await _update_user_memory(
                agent_id, user_id, 
                memories_by_type.get("directive", [])
            )
            session.memory_context.user_memory_timestamp = datetime.now(timezone.utc)

    # Update agent memories with new facts and suggestions
    if memories_by_type.get("fact") or memories_by_type.get("suggestion"):
        await _update_agent_memory(
            agent_id,
            memories_by_type.get("fact", []),
            memories_by_type.get("suggestion", [])
        )
        session.memory_context.agent_memory_timestamp = datetime.now(timezone.utc)

    if LOCAL_DEV:
        memories_created = [individual_memory for memory_list in memories_by_type.values() for individual_memory in memory_list if individual_memory.content.strip()]
        print(f"\n✓ Formed {len(memories_created)} new memories:")
        for memory_type, memories in extracted_data.items():
            if len(memories) > 0:
                print(f"  {len(memories)} x {memory_type}:")
                for memory in memories:
                    print(f"    - {memory}")

    return


async def _update_user_memory(
    agent_id: ObjectId, user_id: ObjectId, new_directive_memories: List[SessionMemory]
):
    """
    Add new directives to user_memory and maybe consolidate.
    Called after new directive memories are created.
    """
    try:
        if not agent_id or not user_id or not new_directive_memories:
            return

        user_memory = UserMemory.find_one_or_create(
            {"agent_id": agent_id, "user_id": user_id}
        )
        
        await _add_memories_and_maybe_consolidate(
            memory_doc=user_memory,
            new_memory_ids=[m.id for m in new_directive_memories],
            unabsorbed_field='unabsorbed_memory_ids',
            max_before_consolidation=MAX_DIRECTIVES_COUNT_BEFORE_CONSOLIDATION,
            consolidation_func=_consolidate_user_directives,
            memory_type="directive"
        )
        
        # Always regenerate fully formed user memory after any update
        await _regenerate_fully_formed_user_memory(user_memory)

        # Timestamp is already updated in user_memory.last_updated_at

    except Exception as e:
        print(f"Error updating user memory: {e}")
        traceback.print_exc()


async def _update_agent_memory(
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
        memories_by_shard = {}
        
        # Group facts
        for fact_memory in new_fact_memories:
            if fact_memory.shard_id is None:
                logging.warning(f"Skipping fact memory without shard_id: '{fact_memory.content}'")
                continue
            if fact_memory.shard_id not in memories_by_shard:
                memories_by_shard[fact_memory.shard_id] = {'facts': [], 'suggestions': []}
            memories_by_shard[fact_memory.shard_id]['facts'].append(fact_memory)
        
        # Group suggestions
        for suggestion_memory in new_suggestion_memories:
            if suggestion_memory.shard_id is None:
                logging.warning(f"Skipping suggestion memory without shard_id: '{suggestion_memory.content}'")
                continue
            if suggestion_memory.shard_id not in memories_by_shard:
                memories_by_shard[suggestion_memory.shard_id] = {'facts': [], 'suggestions': []}
            memories_by_shard[suggestion_memory.shard_id]['suggestions'].append(suggestion_memory)
        
        # Update each shard
        for shard_id, shard_memories in memories_by_shard.items():
            try:
                shard = AgentMemory.from_mongo(shard_id)
                if not shard:
                    logging.warning(f"No agent memory shard found for shard_id '{shard_id}'")
                    continue
                
                # Update shard inline instead of calling _update_single_shard
                shard_updated = False
                
                # Add new facts (FIFO)
                new_facts = shard_memories.get('facts', [])
                if new_facts:
                    for fact in new_facts:
                        shard.facts.append(fact.id)
                    
                    # Maintain FIFO - keep only the most recent facts
                    if len(shard.facts) > MAX_FACTS_PER_SHARD:
                        shard.facts = shard.facts[-MAX_FACTS_PER_SHARD:]
                    
                    shard_updated = True
                
                # Add new suggestions
                new_suggestions = shard_memories.get('suggestions', [])
                if new_suggestions:
                    await _add_memories_and_maybe_consolidate(
                        memory_doc=shard,
                        new_memory_ids=[s.id for s in new_suggestions],
                        unabsorbed_field='unabsorbed_memory_ids',
                        max_before_consolidation=MAX_SUGGESTIONS_COUNT_BEFORE_CONSOLIDATION,
                        consolidation_func=_consolidate_agent_suggestions,
                        memory_type="suggestion"
                    )
                    shard_updated = True
                
                if shard_updated:
                    await _regenerate_fully_formed_agent_memory(shard)
                    
            except Exception as e:
                logging.error(f"Could not update shard with shard_id '{shard_id}': {e}")
                continue

        return
        
    except Exception as e:
        print(f"Error updating agent memories: {e}")
        traceback.print_exc()
        return


async def _add_memories_and_maybe_consolidate(
    memory_doc,
    new_memory_ids: List[ObjectId],
    unabsorbed_field: str,
    max_before_consolidation: int,
    consolidation_func,
    memory_type: str
):
    """Generic function to add memories to a document and trigger consolidation if needed."""
    if not new_memory_ids:
        return
    
    # Add new memories to the unabsorbed list - handle legacy records that might not have the field
    unabsorbed_list = getattr(memory_doc, unabsorbed_field, [])
    unabsorbed_list.extend(new_memory_ids)
    setattr(memory_doc, unabsorbed_field, unabsorbed_list)
    memory_doc.save()
    
    # Check if consolidation is needed
    if len(unabsorbed_list) >= max_before_consolidation:
        await consolidation_func(memory_doc)


async def _consolidate_with_llm(
    consolidation_prompt_template: str,
    generation_name: str,
    agent_id: ObjectId,
    session_id: ObjectId = None,
    user_id: ObjectId = None,
    **format_args
) -> str:
    """Generic LLM consolidation function that works with any memory type.
    
    Args:
        consolidation_prompt_template: The prompt template to use (e.g., USER_MEMORY_CONSOLIDATION_PROMPT)
        generation_name: Name for this specific generation in Langfuse
        agent_id: Agent ID for tracking
        session_id: Session ID for tracking (optional)
        user_id: User ID for tracking (optional)
        **format_args: Arguments to format the prompt template with
    """
    # Handle empty current memory
    if 'current_memory' in format_args and not format_args['current_memory']:
        format_args['current_memory'] = "EMPTY (This is the first consolidation --> Be concise and avoid inventing any new information, more memories will be added soon!)"
    
    consolidation_prompt = consolidation_prompt_template.format(**format_args)

    if LOCAL_DEV and 0:
        print(f"--- Final LLM Consolidation Prompt: ---\n{consolidation_prompt}")
        print("----------------------------------------\n\n")

    context = LLMContext(
        messages=[ChatMessage(role="user", content=consolidation_prompt)],
        config=LLMConfig(model=MEMORY_LLM_MODEL),
        metadata=LLMContextMetadata(
            session_id=f"{os.getenv('DB')}-{str(session_id)}" if session_id else f"{os.getenv('DB')}-memory-consolidation",
            trace_name="FN_form_memories",
            trace_id=str(uuid.uuid4()),
            generation_name=generation_name,
            trace_metadata=LLMTraceMetadata(
                session_id=str(session_id) if session_id else None,
                user_id=str(user_id) if user_id else None,
                agent_id=str(agent_id),
            ),
        ),
    )

    llm_response = await async_prompt(context)

    return llm_response.content


async def extract_memories_with_llm(
    conversation_text: str, 
    extraction_prompt: str,
    extraction_elements: List[str],
    generation_name: str,
    agent_id: ObjectId,
    session_id: ObjectId = None,
    user_id: ObjectId = None,
    shard_name: str = None
) -> Dict[str, List[str]]:
    """Use LLM to extract categorized memories from conversation text"""
    
    if CONVERSATION_TEXT_TOKEN not in extraction_prompt:
        extraction_prompt = "Conversation text:\n" + CONVERSATION_TEXT_TOKEN + "\n" + extraction_prompt
    
    extraction_prompt = extraction_prompt.replace(CONVERSATION_TEXT_TOKEN, conversation_text)

    if "-&&-" in extraction_prompt:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@ WARNING in extract_memories_with_llm @@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        print(extraction_prompt)
    
    # Dynamically create model with only requested fields and max_length constraints
    fields = {}
    for element in extraction_elements:
        if element in MEMORY_TYPES:
            memory_type = MEMORY_TYPES[element]
            fields[element] = (List[str], Field(default_factory=list, max_length=memory_type.max_items))
        else:
            # Fallback for unknown memory types
            fields[element] = (List[str], Field(default_factory=list, max_length=1))
    
    MemoryModel = create_model("MemoryExtraction", **fields)
    
    # Use LLM with structured output
    context = LLMContext(
        messages=[ChatMessage(role="user", content=extraction_prompt)],
        config=LLMConfig(
            model=MEMORY_LLM_MODEL,
            response_format=MemoryModel  # This forces JSON output with only requested fields
        ),
        metadata=LLMContextMetadata(
            session_id=f"{os.getenv('DB')}-{str(session_id)}" if session_id else f"{os.getenv('DB')}-memory-extraction",
            trace_name="FN_form_memories",
            trace_id=str(uuid.uuid4()),
            generation_name=generation_name,
            trace_metadata=LLMTraceMetadata(
                session_id=str(session_id) if session_id else None,
                user_id=str(user_id) if user_id else None,
                agent_id=str(agent_id),
                additional_metadata={
                    "shard_name": shard_name,
                    "extraction_elements": extraction_elements
                } if shard_name else {"extraction_elements": extraction_elements}
            ),
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
    if LOCAL_DEV and 0:
        print("########################")
        print("Forming new memories...")
        #print(f"--- Messages: ---\n{context.messages}")
        #print(f"--- Prompt: ---\n{extraction_prompt}")
        print(f"--- New Memories: ---\n{formatted_data}")
        print("########################")
    
    return formatted_data


async def _consolidate_user_directives(user_memory: UserMemory):
    """
    Consolidate unabsorbed directive memories into the user memory blob using LLM.
    """
    try:
        # Load unabsorbed memories - handle legacy records that might not have unabsorbed_memory_ids field
        unabsorbed_memory_ids = getattr(user_memory, 'unabsorbed_memory_ids', [])
        unabsorbed_memories = await _load_memories_by_ids(
            unabsorbed_memory_ids, 
            memory_type_filter="directive"
        )
        
        if not unabsorbed_memories:
            return

        # Prepare content for LLM consolidation
        new_memories_text = _format_memories_with_age(unabsorbed_memories)
        
        print(f"Consolidating {len(unabsorbed_memories)} directives to user_memory")

        if "-&&-" in USER_MEMORY_CONSOLIDATION_PROMPT:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@ WARNING in AGENT_MEMORY_CONSOLIDATION_PROMPT @@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        # Use generic LLM consolidation
        consolidated_content = await _consolidate_with_llm(
            USER_MEMORY_CONSOLIDATION_PROMPT,
            generation_name="FN_form_memories_consolidate_user_memory",
            agent_id=user_memory.agent_id,
            user_id=user_memory.user_id,
            current_memory=user_memory.content,
            new_memories=new_memories_text,
            max_words=USER_MEMORY_BLOB_MAX_WORDS
        )

        # Update memory document inline - ensure unabsorbed_memory_ids field exists
        user_memory.content = consolidated_content
        user_memory.unabsorbed_memory_ids = []
        user_memory.last_updated_at = datetime.now(timezone.utc)
        user_memory.save()
        
        print(f"✓ Consolidated user memory updated (length: {len(consolidated_content)} chars)")

    except Exception as e:
        print(f"Error consolidating user directives: {e}")
        traceback.print_exc()


async def _consolidate_agent_suggestions(shard: AgentMemory):
    """
    Consolidate unabsorbed suggestion memories into the agent memory shard using LLM.
    """
    try:
        # Load unabsorbed suggestions - handle legacy shards that might not have unabsorbed_memory_ids field
        unabsorbed_memory_ids = getattr(shard, 'unabsorbed_memory_ids', [])
        unabsorbed_suggestions = await _load_memories_by_ids(
            unabsorbed_memory_ids,
            memory_type_filter="suggestion"
        )
        
        if not unabsorbed_suggestions:
            return

        # Load current facts
        current_facts = await _load_memories_by_ids(
            shard.facts,
            memory_type_filter="fact"
        )

        facts_text = _format_memories_with_age(current_facts)
        suggestions_text = "\n".join([f"- {s.content}" for s in unabsorbed_suggestions])

        print(f"Consolidating {len(unabsorbed_suggestions)} suggestions for agent memory shard '{shard.shard_name}'")
        print(f"Current memory length: {len(shard.content or '')} chars")
        print(f"New suggestions:\n{suggestions_text}")

        populated_prompt = AGENT_MEMORY_CONSOLIDATION_PROMPT.replace(
                SHARD_EXTRACTION_PROMPT_TOKEN, shard.extraction_prompt
            )
        
        if "-&&-" in populated_prompt:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@ WARNING in AGENT_MEMORY_CONSOLIDATION_PROMPT @@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        # Use generic LLM consolidation
        consolidated_content = await _consolidate_with_llm(
            populated_prompt,
            generation_name="FN_form_memories_consolidate_agent_memory",
            agent_id=shard.agent_id,
            current_memory=shard.content or "",
            facts_text=facts_text if facts_text else "(no facts available)",
            suggestions_text=suggestions_text,
            shard_name=shard.shard_name or "Unknown Shard",
            max_words=AGENT_MEMORY_BLOB_MAX_WORDS
        )

        # Update agent memory - ensure unabsorbed_memory_ids field exists
        shard.content = consolidated_content
        shard.unabsorbed_memory_ids = []  # Reset unabsorbed list
        shard.last_updated_at = datetime.now(timezone.utc)
        shard.save()
        
    except Exception as e:
        print(f"Error consolidating agent suggestions for shard '{shard.shard_name}': {e}")
        traceback.print_exc()


async def _load_memories_by_ids(
    memory_ids: List[ObjectId], 
    memory_type_filter: str = None
) -> List[SessionMemory]:
    """Helper function to load memories by IDs with optional type filtering."""
    memories = []
    for memory_id in memory_ids:
        try:
            memory = SessionMemory.from_mongo(memory_id)
            if memory and (not memory_type_filter or memory.memory_type == memory_type_filter):
                memories.append(memory)
        except Exception as e:
            logging.warning(f"Could not load memory {memory_id}: {e}")
    return memories


async def _regenerate_fully_formed_agent_memory(shard: AgentMemory):
    """
    Regenerate the fully formed memory shard by combining:
    - Recent facts with age information
    - Consolidated memory blob
    - Unabsorbed memories (suggestions)
    """
    print(f" ---> regenerating fully formed agent memory for shard {shard.shard_name}")
    try:
        shard_content = []

        shard_context = f"You have an active collective memory shard called {shard.shard_name}. Context for this memory collection: {shard.extraction_prompt}"
        shard_content.append(f"## Memory shard context:\n\n{shard_context}")
        
        facts = await _load_memories_by_ids(shard.facts, "fact")
        if facts:
            facts_formatted = _format_memories_with_age(facts)
            if facts_formatted:
                shard_content.append(f"## Shard facts:\n\n{facts_formatted}")
        
        # Add consolidated content
        if shard.content:
            shard_content.append(f"## Current consolidated shard memory:\n\n{shard.content}")
        
        # Add unabsorbed suggestions - handle legacy shards that might not have unabsorbed_memory_ids field
        unabsorbed_memory_ids = getattr(shard, 'unabsorbed_memory_ids', [])
        suggestions = await _load_memories_by_ids(
            unabsorbed_memory_ids, 
            "suggestion"
        )
        if suggestions:
            suggestions_text = "\n".join([f"- {s.content}" for s in suggestions])
            shard_content.append(f"## Recent shard suggestions:\n\n{suggestions_text}")
        
        # Combine all parts
        shard.fully_formed_memory = "\n\n".join(shard_content) if shard_content else ""
        shard.last_updated_at = datetime.now(timezone.utc)
        shard.save()

    except Exception as e:
        print(f"Error regenerating fully formed memory shard for '{shard.shard_name}': {e}")
        traceback.print_exc()
        shard.fully_formed_memory = ""


async def _regenerate_fully_formed_user_memory(user_memory: UserMemory):
    """
    Regenerate the fully formed user memory by combining:
    - Consolidated memory blob
    - Unabsorbed directive memories with age information
    """

    print(f" ---> regenerating fully formed user memory for user {user_memory.user_id}")

    try:
        user_content = []
        
        # Add consolidated content
        if user_memory.content:
            user_content.append(f"## Consolidated user memory:\n\n{user_memory.content}")
        
        # Add unabsorbed directives - handle legacy records that might not have unabsorbed_memory_ids field
        unabsorbed_memory_ids = getattr(user_memory, 'unabsorbed_memory_ids', [])
        directives = await _load_memories_by_ids(
            unabsorbed_memory_ids,
            "directive"
        )
        if directives:
            directives_text = _format_memories_with_age(directives)
            user_content.append(f"## Recent user directives:\n\n{directives_text}")
        
        # Combine all parts
        user_memory.fully_formed_memory = "\n\n".join(user_content) if user_content else ""
        user_memory.last_updated_at = datetime.now(timezone.utc)
        user_memory.save()

    except Exception as e:
        print(f"Error regenerating fully formed user memory for user {user_memory.user_id}: {e}")
        traceback.print_exc()
        user_memory.fully_formed_memory = ""




async def _extract_all_memories(
    agent_id: ObjectId, 
    conversation_text: str,
    session: Session,
    user_id: ObjectId = None
) -> Tuple[Dict[str, List[str]], Dict[str, ObjectId]]:
    """Extract both regular and collective memories from conversation."""
    extracted_data = {}
    memory_to_shard_map = {}

    session_id = session.id
    
    # Extract regular memories (episode and directive)
    regular_memories = await extract_memories_with_llm(
        conversation_text, 
        extraction_prompt=REGULAR_MEMORY_EXTRACTION_PROMPT,
        extraction_elements=["episode", "directive"],
        generation_name="FN_form_memories_extract_regular_memories",
        agent_id=agent_id,
        session_id=session_id,
        user_id=user_id
    )
    extracted_data.update(regular_memories)
    
    # Extract collective memories from active shards
    active_shards = AgentMemory.find({"agent_id": agent_id, "is_active": True})

    if not active_shards:
        print(f"No active shards found for agent {agent_id}")
        return extracted_data, memory_to_shard_map
    
    for shard in active_shards:
        if not shard.extraction_prompt:
            continue
            
        try:
            # Populate the collective memory extraction prompt with shard's fully formed memory shard
            populated_prompt = AGENT_MEMORY_EXTRACTION_PROMPT.replace(
                FULLY_FORMED_AGENT_MEMORY_TOKEN, shard.fully_formed_memory or shard.extraction_prompt
            )
            
            # Extract facts and suggestions for this shard
            shard_memories = await extract_memories_with_llm(
                conversation_text=conversation_text,
                extraction_prompt=populated_prompt,
                extraction_elements=["fact", "suggestion"],
                generation_name=f"FN_form_memories_extract_shard_memories",
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                shard_name=shard.shard_name
            )
            
            for memory_type, memories in shard_memories.items():
                if memory_type not in extracted_data.keys():
                    extracted_data[memory_type] = []
                
                # Track each individual memory's shard origin
                for memory_content in memories:
                    memory_index = len(extracted_data[memory_type])
                    extracted_data[memory_type].append(memory_content)
                    memory_to_shard_map[f"{memory_type}_{memory_index}"] = shard.id
            
        except Exception as e:
            print(f"Error extracting memories from shard '{shard.shard_name}': {e}")
            traceback.print_exc()
    
    return extracted_data, memory_to_shard_map

def should_form_memories(agent_id: ObjectId, session: Session, force_memory_formation: bool = False) -> bool:
    """
    Check if memory formation should run based on either messages elapsed or tokens accumulated since last formation.
    Returns True if memory formation should occur.
    """
    try:
        from eve.agent.session.session import select_messages
        session_messages = select_messages(
            session, selection_limit=SESSION_MESSAGES_LOOKBACK_LIMIT
        )

        if not agent_id or not session_messages or len(session_messages) == 0:
            return False

        # Create message ID to index mapping for O(1) lookup
        message_id_to_index = {msg.id: i for i, msg in enumerate(session_messages)}
        
        # Find the position of the last memory formation message
        last_memory_position = -1
        if session.memory_context.last_memory_message_id:
            last_memory_position = message_id_to_index.get(session.memory_context.last_memory_message_id, -1)

        # Get recent messages since last memory formation
        recent_messages = session_messages[last_memory_position + 1:]
        messages_since_last = len(recent_messages)
        
        # Check minimum message threshold first
        if messages_since_last < NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES:
            return False
        
        if force_memory_formation:
            return True
        
        # If using message-based trigger
        if MEMORY_FORMATION_MSG_INTERVAL is not None:
            print(f"Session {session.id}: {len(session_messages)} total messages, {messages_since_last} since last memory formation")
            return messages_since_last >= MEMORY_FORMATION_MSG_INTERVAL
        
        # Otherwise use token-based trigger
        else:
            # Convert recent messages to text and count tokens
            recent_text = messages_to_text(recent_messages, fast_dry_run=True)
            tokens_since_last = estimate_tokens(recent_text)
            print(f"Session {session.id}: {len(session_messages)} total messages, ~{tokens_since_last} tokens since last memory formation")
            return tokens_since_last >= MEMORY_FORMATION_TOKEN_INTERVAL
            
    except Exception as e:
        print(f"Error checking if memory formation should run for agent {agent_id} in session {session.id}: {e}")
        traceback.print_exc()
        return False


async def maybe_form_memories(agent_id: ObjectId, session: Session) -> bool:
    """
    Check if memories should be formed, and if so, form them.
    Also ensures the session is tracked in session_state for cold session processing.
    """
    start_time = time.time()

    if not should_form_memories(agent_id, session):
        stop_time = time.time()
        print(f" #### maybe_form_memories() returned without forming after {stop_time - start_time:.2f} seconds")
        return
    
    await form_memories(agent_id, session)
    stop_time = time.time()
    print(f" #### maybe_form_memories() took {stop_time - start_time:.2f} seconds to complete")
    return


async def form_memories(agent_id: ObjectId, session: Session) -> bool:
    """
    Form memories from recent conversation messages.
    Returns True if memories were formed.
    """
    start_time = time.time()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(f"--- STARTED FORMING MEMORIES ---")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    
    from eve.agent.session.session import select_messages
    session_messages = select_messages(
        session, selection_limit=SESSION_MESSAGES_LOOKBACK_LIMIT
    )

    if not agent_id or not session_messages or len(session_messages) == 0:
        print(f"No agent or messages found for session {session.id}")
        return False
    
    try:
        # Get messages since last memory formation
        recent_messages = _get_recent_messages(session_messages, session.memory_context.last_memory_message_id)
        
        if recent_messages:
            conversation_text = messages_to_text(recent_messages)

            extracted_data, memory_to_shard_map = await _extract_all_memories(
                agent_id, conversation_text, session
            )
            
            await _save_all_memories(
                agent_id, extracted_data, recent_messages, session, memory_to_shard_map
            )
        
        related_users = list(
            set([msg.sender for msg in session_messages if msg.sender and msg.sender != agent_id])
        )
        last_speaker_id = related_users[-1]

        from eve.agent.session.memory_assemble_context import assemble_memory_context
        await assemble_memory_context(session, agent_id, last_speaker_id, force_refresh=True, reason="form_memories", skip_save=True)
        
        # Update the session's memory formation tracking and save once
        session.memory_context.last_activity = datetime.now(timezone.utc)
        session.memory_context.last_memory_message_id = session_messages[-1].id
        session.memory_context.messages_since_memory_formation = 0
        session.save()

        print(f"Form memories took {time.time() - start_time:.2f} seconds to complete")
        return True

    except Exception as e:
        print(f"Error processing memory formation: {e}")
        traceback.print_exc()
        return False