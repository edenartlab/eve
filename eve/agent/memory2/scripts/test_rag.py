#!/usr/bin/env python3
"""
Memory System v2 - RAG Test Script

Test script for validating RAG search functionality on the memory2_facts collection.

Usage:
    python -m eve.agent.memory2.scripts.test_rag
    python -m eve.agent.memory2.scripts.test_rag --check-indexes
    python -m eve.agent.memory2.scripts.test_rag --agent-id 507f1f77bcf86cd799439011 --query "multi-agent games"

"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

# Import RAG thresholds for display
from eve.agent.memory2.constants import (
    RAG_TOP_K,
    RAG_SEMANTIC_SCORE_THRESHOLD,
    RAG_TEXT_SCORE_THRESHOLD,
    RAG_RRF_SCORE_THRESHOLD,
)

async def run_semantic_search(
    embedding: List[float],
    pre_filter: Dict,
    limit: int,
) -> tuple[List[Dict], float, int]:
    """Run vector search with pre-computed embedding. Returns (results, elapsed_ms, filtered_count)."""
    from eve.agent.memory2.rag import get_vector_index_name
    from eve.agent.memory2.models import Fact

    start = time.perf_counter()
    collection = Fact.get_collection()

    pipeline = [
        {
            "$vectorSearch": {
                "index": get_vector_index_name(),
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": limit * 10,
                "limit": limit,
                "filter": pre_filter,
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}, "search_type": "semantic"}},
        {"$project": {"_id": 1, "content": 1, "scope": 1, "agent_id": 1,
                      "user_id": 1, "formed_at": 1, "score": 1, "search_type": 1}},
    ]
    # Run sync PyMongo call in thread pool for true async parallelism
    loop = asyncio.get_event_loop()
    raw_results = await loop.run_in_executor(
        None, lambda: list(collection.aggregate(pipeline))
    )

    # Apply semantic score threshold
    results = [r for r in raw_results if r.get("score", 0) >= RAG_SEMANTIC_SCORE_THRESHOLD]
    filtered_count = len(raw_results) - len(results)

    elapsed = (time.perf_counter() - start) * 1000
    return results, elapsed, filtered_count


async def run_text_search(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str],
    limit: int,
) -> tuple[List[Dict], float, int]:
    """Run text search. Returns (results, elapsed_ms, filtered_count).

    Note: _text_search already applies threshold filtering internally,
    so we report 0 for filtered_count here (filtering logged in rag.py).
    """
    from eve.agent.memory2.rag import _text_search

    start = time.perf_counter()
    results = await _text_search(query, agent_id, user_id, scope_filter, limit)
    elapsed = (time.perf_counter() - start) * 1000
    # Filtered count is handled internally by _text_search
    return results, elapsed, 0


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
    limit: int = 10,
) -> tuple[List[Dict], int]:
    """Merge results using Reciprocal Rank Fusion. Returns (results, filtered_count)."""
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = str(doc.get("_id", id(doc)))
            rrf_score = 1.0 / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in docs:
                docs[doc_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Apply RRF threshold and limit
    results = []
    filtered_count = 0
    for doc_id in sorted_ids:
        rrf_score = scores[doc_id]
        if rrf_score < RAG_RRF_SCORE_THRESHOLD:
            filtered_count += 1
            continue
        if len(results) >= limit:
            break
        results.append({**docs[doc_id], "rrf_score": rrf_score})

    return results, filtered_count


async def get_sample_agent_id():
    """Get a sample agent ID from existing facts."""
    from eve.agent.memory2.models import Fact
    sample = Fact.get_collection().find_one({}, {"agent_id": 1})
    return sample["agent_id"] if sample and "agent_id" in sample else None


async def main():
    parser = argparse.ArgumentParser(description="Test RAG search on memory2_facts")
    parser.add_argument("--agent-id", type=str, help="Agent ID (uses sample if not provided)")
    parser.add_argument("--user-id", type=str, help="User ID for user-scoped facts")
    parser.add_argument("--query", type=str, default="project deadline budget", help="Search query")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MEMORY2 RAG TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Resolve agent ID
    agent_id = ObjectId(args.agent_id) if args.agent_id else await get_sample_agent_id()
    if not agent_id:
        print("\n❌ No facts found. Provide --agent-id or ensure facts exist.")
        return

    user_id = ObjectId(args.user_id) if args.user_id else None
    scope_filter = ["user", "agent"] if user_id else ["agent"]

    print(f"\nAgent: {agent_id}")
    print(f"Query: '{args.query}'")
    print(f"Limit: {RAG_TOP_K}")

    print(f"\nRelevance Thresholds:")
    print(f"  Semantic: {RAG_SEMANTIC_SCORE_THRESHOLD} (cosine similarity)")
    print(f"  Text:     {RAG_TEXT_SCORE_THRESHOLD} (BM25 score)")
    print(f"  RRF:      {RAG_RRF_SCORE_THRESHOLD} (post-fusion)")

    # =========================================================================
    # STEP 1: Generate query embedding
    # =========================================================================
    from eve.agent.memory2.fact_storage import get_embedding
    from eve.agent.memory2.rag import _build_scope_filter

    print("\n" + "-" * 60)
    embed_start = time.perf_counter()
    embedding = await get_embedding(args.query)
    embed_time = (time.perf_counter() - embed_start) * 1000
    print(f"1. EMBEDDING:        {embed_time:7.1f}ms")

    if not embedding or embedding == [0.0] * len(embedding):
        print("   ❌ Failed to get embedding")
        return

    # =========================================================================
    # STEP 2: Run semantic + text search in parallel
    # =========================================================================
    pre_filter = _build_scope_filter(agent_id, user_id, scope_filter)

    search_start = time.perf_counter()
    semantic_task = run_semantic_search(embedding, pre_filter, RAG_TOP_K * 2)
    text_task = run_text_search(args.query, agent_id, user_id, scope_filter, RAG_TOP_K * 2)

    (semantic_results, semantic_time, semantic_filtered), (text_results, text_time, text_filtered) = await asyncio.gather(
        semantic_task, text_task
    )
    parallel_time = (time.perf_counter() - search_start) * 1000

    semantic_filter_info = f", {semantic_filtered} filtered" if semantic_filtered else ""
    text_filter_info = f", {text_filtered} filtered" if text_filtered else ""
    print(f"2. SEMANTIC SEARCH:  {semantic_time:7.1f}ms  ({len(semantic_results)} results{semantic_filter_info})")
    print(f"3. TEXT SEARCH:      {text_time:7.1f}ms  ({len(text_results)} results{text_filter_info})")
    print(f"   (parallel wall):  {parallel_time:7.1f}ms")

    # =========================================================================
    # STEP 3: Merge with RRF
    # =========================================================================
    rrf_start = time.perf_counter()
    merged_results, rrf_filtered = reciprocal_rank_fusion([semantic_results, text_results], k=60, limit=RAG_TOP_K)
    rrf_time = (time.perf_counter() - rrf_start) * 1000
    rrf_filter_info = f", {rrf_filtered} filtered" if rrf_filtered else ""
    print(f"4. RRF MERGE:        {rrf_time:7.1f}ms  ({len(merged_results)} results{rrf_filter_info})")

    # =========================================================================
    # STEP 4: Format tool response
    # =========================================================================
    from eve.agent.memory2.rag_tool import format_facts_for_tool_response

    format_start = time.perf_counter()
    tool_response = format_facts_for_tool_response(merged_results)
    format_time = (time.perf_counter() - format_start) * 1000
    print(f"5. FORMAT RESPONSE:  {format_time:7.1f}ms")

    total_time = embed_time + parallel_time + rrf_time + format_time
    print("-" * 60)
    print(f"   TOTAL:            {total_time:7.1f}ms")

    # =========================================================================
    # Individual Search Stage Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH RESULTS (cosine similarity)")
    print("=" * 60)
    if semantic_results:
        for i, r in enumerate(semantic_results, 1):
            score = r.get("score", 0)
            scope = r.get("scope", "?")
            content = r.get("content", "")[:100]
            print(f"{i:2}. [{scope}] score={score:.4f} {content}...")
    else:
        print("   No results above threshold")

    print("\n" + "=" * 60)
    print("TEXT SEARCH RESULTS (BM25)")
    print("=" * 60)
    if text_results:
        for i, r in enumerate(text_results, 1):
            score = r.get("score", 0)
            scope = r.get("scope", "?")
            content = r.get("content", "")[:100]
            print(f"{i:2}. [{scope}] score={score:.4f} {content}...")
    else:
        print("   No results above threshold")

    # =========================================================================
    # Hybrid Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("HYBRID RESULTS (RRF merged, threshold-filtered)")
    print("=" * 60)
    for i, r in enumerate(merged_results, 1):
        rrf_score = r.get("rrf_score", 0)
        raw_score = r.get("score", 0)
        search_type = r.get("search_type", "?")
        scope = r.get("scope", "?")
        content = r.get("content", "")[:100]
        print(f"{i:2}. [{scope}] rrf={rrf_score:.4f} ({search_type}: {raw_score:.3f}) {content}...")

    print("\n" + "=" * 60)
    print("TOOL RESPONSE")
    print("=" * 60)
    print(tool_response)


if __name__ == "__main__":
    asyncio.run(main())
