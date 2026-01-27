"""
Memory System v2 - RAG Retrieval

This module implements the RAG (Retrieval-Augmented Generation) pipeline for
searching facts stored in the vector database.

The RAG system is completely independent from the always-in-context memory
system and can be enabled/disabled separately.

Key features:
- Semantic search using MongoDB Atlas vector search
- Text search using MongoDB Atlas Search
- Hybrid search with Reciprocal Rank Fusion (RRF)
- Pre-filtering by scope for efficient queries
- Access tracking for importance scoring
"""

import asyncio
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.constants import (
    LOCAL_DEV,
    RAG_SEMANTIC_SCORE_THRESHOLD,
    RAG_TEXT_SCORE_THRESHOLD,
    RAG_RRF_SCORE_THRESHOLD,
)
from eve.agent.memory2.fact_storage import get_embedding
from eve.agent.memory2.models import Fact


# =============================================================================
# Atlas Search Index Names (environment-specific)
# =============================================================================
def _get_index_suffix() -> str:
    """Get index suffix based on DB environment."""
    db_env = os.getenv("DB", "STAGE").upper()
    return "prod" if db_env == "PROD" else "stage"


def get_vector_index_name() -> str:
    """Get the vector search index name for current environment."""
    return f"fact_vector_index_{_get_index_suffix()}"


def get_text_index_name() -> str:
    """Get the text search index name for current environment."""
    return f"fact_text_index_{_get_index_suffix()}"


async def search_facts(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    scope_filter: List[str] = None,
    match_count: int = 10,
    search_type: str = "hybrid",
) -> List[Dict[str, Any]]:
    """
    Search facts in vector store with PRE-FILTERING.

    Scope filter is applied INSIDE the vector/text search (not post-filter).
    This is more efficient: narrows search space before computing similarities.

    Args:
        query: Search query text
        agent_id: Agent ID
        user_id: User ID (for user-scoped facts)
        scope_filter: List of scopes to search (default: ["user", "agent"])
        match_count: Number of results to return
        search_type: "hybrid", "semantic", or "text"

    Returns:
        List of fact documents with relevance scores
    """
    if scope_filter is None:
        scope_filter = ["user", "agent"]

    try:
        # Build pre-filter for MongoDB Atlas Vector Search (MQL format)
        pre_filter = _build_scope_filter(agent_id, user_id, scope_filter)

        if search_type == "hybrid":
            # Run both searches in parallel
            semantic_task = _semantic_search(query, pre_filter, match_count * 2)
            # Text search needs raw params to build Atlas Search filter format
            text_task = _text_search(query, agent_id, user_id, scope_filter, match_count * 2)

            semantic_results, text_results = await asyncio.gather(
                semantic_task, text_task
            )

            # Merge using RRF
            results = _reciprocal_rank_fusion(
                [semantic_results, text_results],
                k=60,
                limit=match_count,
            )

        elif search_type == "semantic":
            results = await _semantic_search(query, pre_filter, match_count)

        else:  # text
            results = await _text_search(query, agent_id, user_id, scope_filter, match_count)

        # Update access tracking
        if results:
            await _update_access_tracking([r["_id"] for r in results])

        if LOCAL_DEV:
            logger.debug(f"RAG search returned {len(results)} results for: {query[:50]}...")

        return results

    except Exception as e:
        logger.error(f"Error in search_facts: {e}")
        traceback.print_exc()
        return []


def _build_scope_filter(
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str],
) -> Dict:
    """
    Build MongoDB MQL filter for pre-filtering in vector search.

    Returns a standard MQL filter dict that can be used with $vectorSearch.
    For Atlas Vector Search, filters must only reference indexed filter paths.
    """
    conditions = []

    if "agent" in scope_filter:
        conditions.append({
            "$and": [
                {"scope": {"$eq": "agent"}},
                {"agent_id": {"$eq": agent_id}},
            ]
        })

    if "user" in scope_filter and user_id:
        conditions.append({
            "$and": [
                {"scope": {"$eq": "user"}},
                {"agent_id": {"$eq": agent_id}},
                {"user_id": {"$eq": user_id}},
            ]
        })

    if len(conditions) == 1:
        # Single condition - unwrap from list
        return conditions[0]
    elif len(conditions) > 1:
        # Multiple conditions - use $or
        return {"$or": conditions}

    # Fallback - just filter by agent
    return {"agent_id": {"$eq": agent_id}}


def _build_atlas_search_filter(
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str],
) -> List[Dict]:
    """
    Build Atlas Search filter clauses for $search compound queries.

    Atlas Search uses a different filter syntax than MQL:
    - equals: {"path": "field", "value": value}
    - compound.should: for OR logic

    Returns a list of filter clauses for the compound.filter array.
    """
    # For Atlas Search, we build filter conditions differently
    # We'll use compound.should inside filter for OR logic
    conditions = []

    if "agent" in scope_filter:
        conditions.append({
            "compound": {
                "must": [
                    {"equals": {"path": "scope", "value": "agent"}},
                    {"equals": {"path": "agent_id", "value": agent_id}},
                ]
            }
        })

    if "user" in scope_filter and user_id:
        conditions.append({
            "compound": {
                "must": [
                    {"equals": {"path": "scope", "value": "user"}},
                    {"equals": {"path": "agent_id", "value": agent_id}},
                    {"equals": {"path": "user_id", "value": user_id}},
                ]
            }
        })

    if len(conditions) == 0:
        # No conditions - just filter by agent_id
        return [{"equals": {"path": "agent_id", "value": agent_id}}]
    elif len(conditions) == 1:
        # Single condition - return its must clauses directly
        return conditions[0]["compound"]["must"]
    else:
        # Multiple conditions - wrap in compound.should (OR)
        return [{"compound": {"should": conditions, "minimumShouldMatch": 1}}]


async def _semantic_search(
    query: str,
    pre_filter: Dict,
    limit: int,
) -> List[Dict]:
    """
    Vector search with pre-filtering.

    Filter is applied BEFORE similarity computation for efficiency.
    """
    try:
        # Get query embedding
        embedding = await get_embedding(query)

        if not embedding:
            return []

        collection = Fact.get_collection()

        # MongoDB Atlas vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": get_vector_index_name(),
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": limit * 10,  # Over-fetch for better recall
                    "limit": limit,
                    "filter": pre_filter,  # PRE-FILTER: narrows search space first
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"},
                    "search_type": "semantic",
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "content": 1,
                    "score": 1,
                    "scope": 1,
                    "search_type": 1,
                    "formed_at": 1,
                }
            }
        ]

        try:
            # Run sync PyMongo call in thread pool for true async parallelism
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: list(collection.aggregate(pipeline))
            )

            # Filter by semantic score threshold (pre-fusion filtering)
            filtered_results = [
                r for r in results
                if r.get("score", 0) >= RAG_SEMANTIC_SCORE_THRESHOLD
            ]

            if LOCAL_DEV and len(results) != len(filtered_results):
                logger.debug(
                    f"Semantic search: filtered {len(results) - len(filtered_results)} "
                    f"results below threshold {RAG_SEMANTIC_SCORE_THRESHOLD}"
                )

            return filtered_results
        except Exception as e:
            # Fallback for environments without vector search
            if "no such index" in str(e).lower() or "vectorSearch" in str(e):
                logger.warning("Vector search not available, falling back to text search")
                return []
            raise

    except Exception as e:
        logger.error(f"Error in _semantic_search: {e}")
        return []


async def _text_search(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str],
    limit: int,
) -> List[Dict]:
    """
    Text search with pre-filtering.

    Uses MongoDB Atlas Search for keyword/fuzzy matching.
    """
    try:
        collection = Fact.get_collection()

        # Build Atlas Search filter clauses
        filter_clauses = _build_atlas_search_filter(agent_id, user_id, scope_filter)

        # MongoDB Atlas Search pipeline
        pipeline = [
            {
                "$search": {
                    "index": get_text_index_name(),
                    "compound": {
                        "must": [
                            {
                                "text": {
                                    "query": query,
                                    "path": "content",
                                    "fuzzy": {"maxEdits": 1},
                                }
                            }
                        ],
                        "filter": filter_clauses,
                    },
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "searchScore"},
                    "search_type": "text",
                }
            },
            {"$limit": limit},
            {
                "$project": {
                    "_id": 1,
                    "content": 1,
                    "score": 1,
                    "scope": 1,
                    "search_type": 1,
                    "formed_at": 1,
                }
            }
        ]

        try:
            # Run sync PyMongo call in thread pool for true async parallelism
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: list(collection.aggregate(pipeline))
            )

            # Filter by text score threshold (pre-fusion filtering)
            filtered_results = [
                r for r in results
                if r.get("score", 0) >= RAG_TEXT_SCORE_THRESHOLD
            ]

            if LOCAL_DEV and len(results) != len(filtered_results):
                logger.debug(
                    f"Text search: filtered {len(results) - len(filtered_results)} "
                    f"results below threshold {RAG_TEXT_SCORE_THRESHOLD}"
                )

            return filtered_results
        except Exception as e:
            # Fallback: simple regex search
            if "no such index" in str(e).lower() or "$search" in str(e):
                logger.warning("Text search not available, using regex fallback")
                # Build MQL filter for regex fallback
                pre_filter = _build_scope_filter(agent_id, user_id, scope_filter)
                return await _regex_fallback_search(query, pre_filter, limit)
            raise

    except Exception as e:
        logger.error(f"Error in _text_search: {e}")
        return []


async def _regex_fallback_search(
    query: str,
    pre_filter: Dict,
    limit: int,
) -> List[Dict]:
    """
    Fallback regex search for environments without Atlas Search.
    """
    try:
        import re

        collection = Fact.get_collection()

        # Build regex pattern (case insensitive)
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        # Combine pre_filter with content regex
        search_query = {**pre_filter, "content": {"$regex": pattern}}

        # Run sync PyMongo call in thread pool for true async parallelism
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(collection.find(
                search_query,
                {"_id": 1, "content": 1, "scope": 1, "formed_at": 1}
            ).limit(limit))
        )

        # Add mock scores
        for r in results:
            r["score"] = 0.5
            r["search_type"] = "regex"

        return results

    except Exception as e:
        logger.error(f"Error in _regex_fallback_search: {e}")
        return []


def _reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
    limit: int = 10,
) -> List[Dict]:
    """
    Merge ranked lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = Σ(1 / (k + rank))

    This is a proven technique for combining results from multiple
    retrieval methods (semantic + text search).

    Args:
        result_lists: List of result lists from different search methods
        k: RRF parameter (default 60, as recommended in literature)
        limit: Maximum results to return

    Returns:
        Merged and re-ranked results (filtered by RRF threshold, capped by limit)
    """
    rrf_scores = {}
    doc_map = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = str(doc["_id"])
            rrf_score = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = doc

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build result list with RRF scores, filtering by threshold and limit
    results = []
    filtered_count = 0
    for doc_id, rrf_score in sorted_ids:
        # Apply RRF score threshold (post-fusion filtering)
        if rrf_score < RAG_RRF_SCORE_THRESHOLD:
            filtered_count += 1
            continue

        # Apply top-k limit
        if len(results) >= limit:
            break

        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = rrf_score
        results.append(doc)

    if LOCAL_DEV and filtered_count > 0:
        logger.debug(
            f"RRF fusion: filtered {filtered_count} results "
            f"below threshold {RAG_RRF_SCORE_THRESHOLD}"
        )

    return results


async def _update_access_tracking(fact_ids: List[ObjectId]) -> None:
    """
    Update access tracking for retrieved facts.

    This data can be used for importance scoring and cleanup decisions.
    """
    if not fact_ids:
        return

    try:
        collection = Fact.get_collection()
        collection.update_many(
            {"_id": {"$in": fact_ids}},
            {
                "$inc": {"access_count": 1},
                "$set": {"last_accessed_at": datetime.now(timezone.utc)},
            },
        )

    except Exception as e:
        logger.error(f"Error updating access tracking: {e}")


async def get_relevant_facts_for_context(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    max_facts: int = 5,
) -> str:
    """
    Get relevant facts formatted for context injection.

    This is a convenience function for getting facts to inject into
    agent context alongside the always-in-context memory.

    Args:
        query: Query text (usually the user's message)
        agent_id: Agent ID
        user_id: User ID
        max_facts: Maximum facts to include

    Returns:
        Formatted string of relevant facts
    """
    try:
        facts = await search_facts(
            query=query,
            agent_id=agent_id,
            user_id=user_id,
            match_count=max_facts,
        )

        if not facts:
            return ""

        # Format facts
        fact_lines = []
        for fact in facts:
            scope = fact.get("scope", "")
            fact_lines.append(f"- [{scope}] {fact['content']}")

        return "\n".join(fact_lines)

    except Exception as e:
        logger.error(f"Error getting relevant facts: {e}")
        return ""


def format_facts_for_tool_response(facts: List[Dict]) -> str:
    """
    Format facts for a tool response to the agent.

    Args:
        facts: List of fact documents

    Returns:
        Formatted string for tool response
    """
    if not facts:
        return "No relevant facts found."

    lines = ["Retrieved facts:"]
    for i, fact in enumerate(facts, 1):
        scope = fact.get("scope", "")
        score = fact.get("rrf_score") or fact.get("score", 0)
        lines.append(f"{i}. [{scope}] (relevance: {score:.2f}) {fact['content']}")

    return "\n".join(lines)
