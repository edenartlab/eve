"""
Memory System v2 - Facts Management (Deduplication & Conflict Resolution)

This module implements the mem0-inspired fact management pipeline that:
1. Embeds new facts
2. Searches for similar existing facts (vector similarity)
3. Uses LLM to decide: ADD, UPDATE, DELETE, or NONE
4. Executes the decided operations

This ensures semantic deduplication and handles:
- Exact duplicates (hash-based)
- Semantic duplicates (same meaning, different wording)
- Contradictions (preference reversals)
- Updates (new info that supersedes old)
"""

import json
import os
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

from bson import ObjectId
from loguru import logger

from eve.agent.llm.llm import async_prompt
from eve.agent.memory2.constants import (
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

    This is the main entry point for fact processing. It:
    1. Embeds all new facts
    2. Searches for similar existing facts
    3. Calls LLM to decide ADD/UPDATE/DELETE/NONE
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
        # Step 1: Get embeddings for all new facts
        contents = [f["content"] for f in extracted_facts]
        embeddings = await get_embeddings_batch(contents)

        # Step 2: For each fact, find similar existing facts
        fact_candidates = []
        for i, fact_data in enumerate(extracted_facts):
            embedding = embeddings[i] if i < len(embeddings) else []

            similar = await search_similar_facts(
                query_embedding=embedding,
                agent_id=agent_id,
                user_id=user_id,
                scope_filter=fact_data.get("scope", ["user", "agent"]),
                threshold=SIMILARITY_THRESHOLD,
                limit=5,
            )

            fact_candidates.append({
                "new_fact": fact_data,
                "embedding": embedding,
                "similar_existing": similar,
            })

        # Step 3: Decide operations (skip LLM if no similar facts found)
        facts_needing_decision = [
            fc for fc in fact_candidates if fc["similar_existing"]
        ]
        facts_direct_add = [
            fc for fc in fact_candidates if not fc["similar_existing"]
        ]

        # Get LLM decisions for facts with similar existing ones
        decisions = []
        if facts_needing_decision:
            decisions = await llm_memory_update_decision(
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
            result = await execute_decision(
                decision=decision,
                agent_id=agent_id,
                user_id=user_id,
            )

            if result:
                saved_facts.append(result)
                fact_contents.append(result.content)

        if LOCAL_DEV:
            logger.debug(f"Processed {len(extracted_facts)} facts: {len(saved_facts)} stored")

        return saved_facts, fact_contents

    except Exception as e:
        logger.error(f"Error in process_extracted_facts: {e}")
        traceback.print_exc()
        return [], []


async def search_similar_facts(
    query_embedding: List[float],
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str],
    threshold: float = 0.7,
    limit: int = 5,
) -> List[Dict]:
    """
    Search for semantically similar existing facts using vector search.

    Args:
        query_embedding: Embedding vector for the query
        agent_id: Agent ID
        user_id: User ID
        scope_filter: List of scopes to search
        threshold: Minimum similarity threshold
        limit: Maximum results

    Returns:
        List of similar facts with scores
    """
    try:
        if not query_embedding:
            return []

        # Build scope filter
        pre_filter = _build_scope_filter(agent_id, user_id, scope_filter)

        # Run vector search
        collection = Fact.get_collection()

        # MongoDB Atlas vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "fact_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
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
                    "score": {"$gte": threshold}
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
                logger.warning("Vector search not available, using hash-based dedup only")
                return []
            raise

    except Exception as e:
        logger.error(f"Error in search_similar_facts: {e}")
        traceback.print_exc()
        return []


def _build_scope_filter(
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str],
) -> Dict:
    """Build MongoDB filter for pre-filtering in vector search."""
    conditions = []

    if "agent" in scope_filter:
        conditions.append({
            "scope": "agent",
            "agent_id": agent_id,
        })

    if "user" in scope_filter and user_id:
        conditions.append({
            "scope": "user",
            "user_id": user_id,
        })

    if conditions:
        return {"$or": conditions}
    return {"agent_id": agent_id}


async def llm_memory_update_decision(
    fact_candidates: List[Dict],
    agent_id: ObjectId,
) -> List[Dict]:
    """
    Call LLM to decide what to do with new facts vs existing memories.

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
                    existing_memories_text += f"- ID: {mem['_id']}, Content: {mem['content']}, Score: {mem['score']:.2f}\n"

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
            logger.debug("Running fact decision LLM call...")

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
        logger.error(f"Error in llm_memory_update_decision: {e}")
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


async def execute_decision(
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
                scope=fact_data.get("scope", ["user"]),
                agent_id=agent_id,
                user_id=user_id if "user" in fact_data.get("scope", []) else None,
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
                    scope=fact_data.get("scope", ["user"]),
                    agent_id=agent_id,
                    user_id=user_id if "user" in fact_data.get("scope", []) else None,
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
