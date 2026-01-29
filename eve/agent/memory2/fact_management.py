"""
Memory System v2 - Facts Management (Deduplication & Conflict Resolution)

This module implements the mem0-inspired fact management pipeline that:
1. Embeds new facts (batch operation)
2. Searches for similar existing facts in parallel (vector similarity)
3. Uses LLM to decide: ADD, UPDATE, DELETE, or NONE (single batched call)
4. Executes the decided operations

This ensures semantic deduplication and handles:
- Exact duplicates (hash-based)
- Semantic duplicates (same meaning, different wording)
- Contradictions (preference reversals)
- Updates (new info that supersedes old)
"""

import asyncio
import json
import os
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

from bson import ObjectId
from loguru import logger

from eve.agent.llm.llm import async_prompt
from eve.agent.memory2.constants import (
    FACTS_DEDUP_SIMILARITY_LIMIT,
    LOCAL_DEV,
    MEMORY_LLM_MODEL_FAST,
    MEMORY_UPDATE_DECISION_PROMPT,
    SIMILARITY_THRESHOLD,
)
from eve.utils.system_utils import async_exponential_backoff
from eve.agent.memory2.fact_storage import (
    delete_fact,
    get_embedding,
    get_embeddings_batch,
    store_fact,
    update_fact,
)
from eve.agent.memory2.models import (
    Fact,
    FactDecision,
    FactDecisionEvent,
    FactDecisionResponse,
)
from eve.agent.session.models import (
    ChatMessage,
    LLMConfig,
    LLMContext,
    LLMContextMetadata,
    LLMTraceMetadata,
)


async def process_extracted_facts(
    extracted_facts: List[Dict],
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
) -> Tuple[List[Fact], List[str]]:
    """
    Process extracted facts through the deduplication pipeline.

    This is the main entry point for fact processing with deduplication. It:
    1. Embeds all new facts (batch operation)
    2. Searches for similar existing facts IN PARALLEL
    3. Calls LLM to decide ADD/UPDATE/DELETE/NONE (single batched call)
    4. Executes the operations

    Args:
        extracted_facts: List of fact dicts with content, scope, etc.
        agent_id: Agent ID
        user_id: User ID (for user-scoped facts)

    Returns:
        Tuple of (saved_facts, fact_content_strings)
    """
    if not extracted_facts:
        return [], []

    try:
        # Step 1: Get embeddings for all new facts (batch operation)
        contents = [f["content"] for f in extracted_facts]
        embeddings = await get_embeddings_batch(contents)

        if LOCAL_DEV:
            logger.debug(f"Got embeddings for {len(contents)} facts")

        # Step 2: Search for similar facts IN PARALLEL
        fact_candidates = await _search_similar_facts_parallel(
            extracted_facts=extracted_facts,
            embeddings=embeddings,
            agent_id=agent_id,
            user_id=user_id,
        )

        # Step 3: Decide operations (skip LLM if no similar facts found for any)
        facts_needing_decision = [
            fc for fc in fact_candidates if fc["similar_existing"]
        ]
        facts_direct_add = [
            fc for fc in fact_candidates if not fc["similar_existing"]
        ]

        if LOCAL_DEV:
            logger.debug(
                f"Dedup: {len(facts_direct_add)} direct adds, "
                f"{len(facts_needing_decision)} need LLM decision"
            )

        # Get LLM decisions for facts with similar existing ones
        decisions = []
        if facts_needing_decision:
            decisions = await _llm_memory_update_decision(
                fact_candidates=facts_needing_decision,
                agent_id=agent_id,
            )

        # Add direct ADDs for facts with no similar existing
        for fc in facts_direct_add:
            decisions.append({
                "new_fact": fc["new_fact"]["content"],
                "event": FactDecisionEvent.ADD,
                "final_text": fc["new_fact"]["content"],
                "embedding": fc["embedding"],
                "fact_data": fc["new_fact"],
            })

        # Step 4: Execute operations
        saved_facts = []
        fact_contents = []

        for decision in decisions:
            result = await _execute_decision(
                decision=decision,
                agent_id=agent_id,
                user_id=user_id,
            )

            if result:
                saved_facts.append(result)
                fact_contents.append(result.content)

        if LOCAL_DEV:
            logger.debug(
                f"Dedup complete: {len(extracted_facts)} extracted -> "
                f"{len(saved_facts)} stored"
            )

        return saved_facts, fact_contents

    except Exception as e:
        logger.error(f"Error in process_extracted_facts: {e}")
        traceback.print_exc()
        return [], []


async def _search_similar_facts_parallel(
    extracted_facts: List[Dict],
    embeddings: List[List[float]],
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
) -> List[Dict]:
    """
    Search for similar existing facts for all new facts IN PARALLEL.

    Args:
        extracted_facts: List of fact dicts
        embeddings: List of embedding vectors (same order as extracted_facts)
        agent_id: Agent ID
        user_id: User ID

    Returns:
        List of fact candidate dicts with similar_existing populated
    """
    # Create search tasks for all facts
    search_tasks = []
    for i, fact_data in enumerate(extracted_facts):
        embedding = embeddings[i] if i < len(embeddings) else []
        fact_scope = fact_data.get("scope", "user")

        # Create coroutine for this search
        task = _search_similar_for_single_fact(
            query_embedding=embedding,
            agent_id=agent_id,
            user_id=user_id,
            scope=fact_scope,
        )
        search_tasks.append(task)

    # Run all searches in parallel
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Build fact candidates with results
    fact_candidates = []
    for i, fact_data in enumerate(extracted_facts):
        embedding = embeddings[i] if i < len(embeddings) else []
        similar = search_results[i] if i < len(search_results) else []

        # Handle exceptions from gather
        if isinstance(similar, Exception):
            logger.error(f"Error searching for similar facts: {similar}")
            similar = []

        fact_candidates.append({
            "new_fact": fact_data,
            "embedding": embedding,
            "similar_existing": similar,
        })

    return fact_candidates


async def _search_similar_for_single_fact(
    query_embedding: List[float],
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope: str,
) -> List[Dict]:
    """
    Search for similar existing facts for a single new fact.

    Facts are ONLY compared within their own scope:
    - "agent" scoped facts are compared with other agent-scoped facts for the same agent
    - "user" scoped facts are compared with other user-scoped facts for the same user

    Args:
        query_embedding: Embedding vector for the query
        agent_id: Agent ID
        user_id: User ID (required for user-scoped facts)
        scope: Scope of the new fact ("user" or "agent")

    Returns:
        List of similar facts with scores
    """
    try:
        if not query_embedding:
            return []

        # For user-scoped facts, we MUST have a user_id to search correctly
        # Without it, we can't filter to the correct user's facts
        if scope == "user" and not user_id:
            logger.warning(
                "Cannot search for similar user-scoped facts without user_id. "
                "Fact will be added without deduplication."
            )
            return []

        # Build scope filter - searches ONLY within the same scope
        pre_filter = _build_scope_filter(agent_id, user_id, [scope])

        # Run vector search
        collection = Fact.get_collection()

        # MongoDB Atlas vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "fact_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": FACTS_DEDUP_SIMILARITY_LIMIT * 10,
                    "limit": FACTS_DEDUP_SIMILARITY_LIMIT,
                    "filter": pre_filter,
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
            {
                "$match": {
                    "score": {"$gte": SIMILARITY_THRESHOLD}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "content": 1,
                    "score": 1,
                    "scope": 1,
                }
            }
        ]

        try:
            results = list(collection.aggregate(pipeline))
            return results
        except Exception as e:
            # Fallback for environments without vector search
            if "no such index" in str(e).lower() or "vectorSearch" in str(e):
                logger.warning(
                    "Vector search not available, using hash-based dedup only"
                )
                return []
            raise

    except Exception as e:
        logger.error(f"Error in _search_similar_for_single_fact: {e}")
        traceback.print_exc()
        return []


def _build_scope_filter(
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str],
) -> Dict:
    """
    Build MongoDB filter for pre-filtering in vector search.

    This ensures facts are ONLY compared within their own scope:
    - "agent" facts match: scope="agent" AND agent_id matches
    - "user" facts match: scope="user" AND user_id matches

    Args:
        agent_id: Agent ID (required)
        user_id: User ID (required for user scope)
        scope_filter: List of scopes to search (typically single scope)

    Returns:
        MongoDB filter dict for vector search pre-filtering
    """
    conditions = []

    if "agent" in scope_filter:
        conditions.append({
            "scope": "agent",
            "agent_id": agent_id,
        })

    if "user" in scope_filter:
        if user_id:
            conditions.append({
                "scope": "user",
                "user_id": user_id,
            })
        else:
            # This should not happen - caller should check user_id first
            # Return impossible filter to match nothing
            logger.warning("_build_scope_filter called for user scope without user_id")
            return {"_id": None}  # Matches nothing

    if not conditions:
        # No valid conditions - return impossible filter to match nothing
        logger.warning(f"_build_scope_filter: no valid conditions for scopes {scope_filter}")
        return {"_id": None}  # Matches nothing

    if len(conditions) == 1:
        # Single scope - no need for $or wrapper
        return conditions[0]

    return {"$or": conditions}


async def _llm_memory_update_decision(
    fact_candidates: List[Dict],
    agent_id: ObjectId,
) -> List[Dict]:
    """
    Call LLM to decide what to do with new facts vs existing memories.

    This processes ALL facts in a SINGLE batched LLM call for efficiency.

    Args:
        fact_candidates: List of {new_fact, embedding, similar_existing}
        agent_id: Agent ID

    Returns:
        List of decision dicts with event, existing_id, final_text
    """
    try:
        if not fact_candidates:
            return []

        # Format for prompt
        new_facts_text = "\n".join([
            f"- {fc['new_fact']['content']}"
            for fc in fact_candidates
        ])

        existing_memories_text = ""
        for fc in fact_candidates:
            if fc["similar_existing"]:
                for mem in fc["similar_existing"]:
                    existing_memories_text += (
                        f"- ID: {mem['_id']}, "
                        f"Content: {mem['content']}, "
                        f"Score: {mem['score']:.2f}\n"
                    )

        if not existing_memories_text:
            existing_memories_text = "None found"

        # Build prompt
        prompt = MEMORY_UPDATE_DECISION_PROMPT.format(
            new_facts=new_facts_text,
            existing_memories=existing_memories_text,
        )

        # LLM call
        context = LLMContext(
            messages=[ChatMessage(role="user", content=prompt)],
            config=LLMConfig(
                model=MEMORY_LLM_MODEL_FAST,
                response_format=FactDecisionResponse,
            ),
            metadata=LLMContextMetadata(
                session_id=f"{os.getenv('DB')}-memory2-fact-decision",
                trace_name="FN_memory2_fact_decision",
                trace_id=str(uuid.uuid4()),
                generation_name="memory2_fact_update_decision",
                trace_metadata=LLMTraceMetadata(
                    agent_id=str(agent_id),
                ),
            ),
            enable_tracing=True,
        )

        if LOCAL_DEV:
            logger.debug(
                f"Running fact decision LLM call for {len(fact_candidates)} facts..."
            )

        # LLM call with automatic retry (3 attempts with exponential backoff)
        response = await async_exponential_backoff(
            lambda: async_prompt(context),
            max_attempts=3,
            initial_delay=2,
            max_jitter=0.5,
        )

        # Parse structured JSON response
        if not response.content or not response.content.strip():
            logger.warning("LLM returned empty response for fact deduplication")
            return []

        result = FactDecisionResponse(**json.loads(response.content))

        # Match decisions back to fact candidates
        decisions = []
        for decision in result.decisions:
            # Find matching candidate
            for fc in fact_candidates:
                if fc["new_fact"]["content"] == decision.new_fact:
                    decisions.append({
                        "new_fact": decision.new_fact,
                        "event": decision.event,
                        "existing_id": decision.existing_id,
                        "existing_text": decision.existing_text,
                        "final_text": decision.final_text,
                        "reasoning": decision.reasoning,
                        "embedding": fc["embedding"],
                        "fact_data": fc["new_fact"],
                    })
                    break

        if LOCAL_DEV:
            for d in decisions:
                logger.debug(f"  {d['event'].value}: {d['new_fact'][:50]}...")

        return decisions

    except Exception as e:
        logger.error(f"Error in _llm_memory_update_decision: {e}")
        traceback.print_exc()
        # Fallback: ADD all facts
        return [
            {
                "new_fact": fc["new_fact"]["content"],
                "event": FactDecisionEvent.ADD,
                "final_text": fc["new_fact"]["content"],
                "embedding": fc["embedding"],
                "fact_data": fc["new_fact"],
            }
            for fc in fact_candidates
        ]


async def _execute_decision(
    decision: Dict,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
) -> Optional[Fact]:
    """
    Execute a single fact management decision.

    Args:
        decision: Decision dict with event, final_text, existing_id, etc.
        agent_id: Agent ID
        user_id: User ID

    Returns:
        Fact document if created/updated, None otherwise
    """
    try:
        event = decision.get("event")

        if event == FactDecisionEvent.ADD:
            fact_data = decision.get("fact_data", {})
            embedding = decision.get("embedding", [])

            fact = Fact(
                content=decision["final_text"],
                scope=fact_data.get("scope", "user"),
                agent_id=agent_id,
                user_id=user_id if fact_data.get("scope", "user") == "user" else None,
                session_id=fact_data.get("session_id"),
                source_message_ids=fact_data.get("source_message_ids", []),
                embedding=embedding,
            )
            fact.save()

            if LOCAL_DEV:
                logger.debug(f"ADD fact: {decision['final_text'][:50]}...")

            return fact

        elif event == FactDecisionEvent.UPDATE:
            existing_id = decision.get("existing_id")
            if existing_id:
                updated_fact = await update_fact(
                    fact_id=ObjectId(existing_id),
                    new_content=decision["final_text"],
                )
                return updated_fact

        elif event == FactDecisionEvent.DELETE:
            existing_id = decision.get("existing_id")
            if existing_id:
                delete_fact(ObjectId(existing_id))

            # After DELETE, add the new contradicting fact if provided
            if decision.get("final_text"):
                fact_data = decision.get("fact_data", {})
                embedding = decision.get("embedding", [])

                # Need to re-embed since content might have changed
                if decision["final_text"] != decision.get("new_fact"):
                    embedding = await get_embedding(decision["final_text"])

                fact = Fact(
                    content=decision["final_text"],
                    scope=fact_data.get("scope", "user"),
                    agent_id=agent_id,
                    user_id=user_id if fact_data.get("scope", "user") == "user" else None,
                    session_id=fact_data.get("session_id"),
                    source_message_ids=fact_data.get("source_message_ids", []),
                    embedding=embedding,
                )
                fact.save()

                if LOCAL_DEV:
                    logger.debug(f"DELETE+ADD fact: {decision['final_text'][:50]}...")

                return fact

        elif event == FactDecisionEvent.NONE:
            # No operation needed
            if LOCAL_DEV:
                logger.debug(f"NONE (skip): {decision.get('new_fact', '')[:50]}...")
            return None

        return None

    except Exception as e:
        logger.error(f"Error executing decision: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# Maintenance Functions
# =============================================================================


async def deduplicate_facts(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    dry_run: bool = True,
) -> Dict:
    """
    Run deduplication on existing facts.

    This is a maintenance function that can be run periodically to clean up
    semantic duplicates that may have accumulated.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional)
        dry_run: If True, only report duplicates without removing

    Returns:
        Dict with deduplication stats
    """
    try:
        stats = {
            "total_facts": 0,
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "errors": 0,
        }

        # Get all facts for this agent
        query = {"agent_id": agent_id}
        if user_id:
            query["$or"] = [
                {"scope": "agent"},
                {"scope": "user", "user_id": user_id},
            ]

        facts = Fact.find(query)
        stats["total_facts"] = len(facts)

        # Track seen facts by hash
        seen_hashes = {}
        duplicates = []

        for fact in facts:
            if fact.hash in seen_hashes:
                duplicates.append(fact)
            else:
                seen_hashes[fact.hash] = fact.id

        stats["duplicates_found"] = len(duplicates)

        if not dry_run:
            for dup in duplicates:
                try:
                    delete_fact(dup.id)
                    stats["duplicates_removed"] += 1
                except Exception as e:
                    logger.error(f"Error removing duplicate: {e}")
                    stats["errors"] += 1

        if LOCAL_DEV:
            logger.debug(f"Deduplication: {stats}")

        return stats

    except Exception as e:
        logger.error(f"Error in deduplicate_facts: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# =============================================================================
# Legacy function aliases (for backwards compatibility)
# =============================================================================

# These are kept for any external code that might reference them
search_similar_facts = _search_similar_for_single_fact
llm_memory_update_decision = _llm_memory_update_decision
execute_decision = _execute_decision
