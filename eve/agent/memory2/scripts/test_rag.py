#!/usr/bin/env python3
"""
Memory System v2 - RAG Test Script

Test script for validating RAG search functionality on the memory2_facts collection.

Usage:
    python -m eve.agent.memory2.scripts.test_rag
    python -m eve.agent.memory2.scripts.test_rag --query "multi-agent games"
    python -m eve.agent.memory2.scripts.test_rag --agent-id 507f1f77bcf86cd799439011
    python -m eve.agent.memory2.scripts.test_rag --check-indexes
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


async def check_indexes_and_stats():
    """Check MongoDB indexes and collection statistics."""
    from eve.agent.memory2.models import Fact
    from eve.agent.memory2.rag import get_vector_index_name, get_text_index_name

    print("\n" + "=" * 60)
    print("COLLECTION STATUS: memory2_facts")
    print("=" * 60)

    collection = Fact.get_collection()

    total_count = collection.count_documents({})
    agent_count = collection.count_documents({"scope": "agent"})
    user_count = collection.count_documents({"scope": "user"})

    print(f"\nDocument counts:")
    print(f"  Total: {total_count} | Agent: {agent_count} | User: {user_count}")

    with_embedding = collection.count_documents({"embedding": {"$exists": True, "$ne": []}})
    print(f"  With embeddings: {with_embedding}")

    vector_index = get_vector_index_name()
    text_index = get_text_index_name()

    print(f"\nAtlas Search indexes (env={os.getenv('DB', 'STAGE')}):")

    # Test vector search index
    try:
        dummy_vector = [0.1] * 1536
        pipeline = [{"$vectorSearch": {"index": vector_index, "path": "embedding",
                     "queryVector": dummy_vector, "numCandidates": 1, "limit": 1}}]
        list(collection.aggregate(pipeline))
        print(f"  {vector_index}: ✅")
    except Exception as e:
        print(f"  {vector_index}: ❌ ({e})")

    # Test text search index
    try:
        pipeline = [{"$search": {"index": text_index, "text": {"query": "test", "path": "content"}}},
                    {"$limit": 1}]
        list(collection.aggregate(pipeline))
        print(f"  {text_index}: ✅")
    except Exception as e:
        print(f"  {text_index}: ❌ ({e})")


async def run_semantic_search(
    embedding: List[float],
    pre_filter: Dict,
    limit: int,
) -> tuple[List[Dict], float]:
    """Run vector search with pre-computed embedding. Returns (results, elapsed_ms)."""
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
    results = list(collection.aggregate(pipeline))
    elapsed = (time.perf_counter() - start) * 1000
    return results, elapsed


async def run_text_search(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    scope_filter: List[str],
    limit: int,
) -> tuple[List[Dict], float]:
    """Run text search. Returns (results, elapsed_ms)."""
    from eve.agent.memory2.rag import _text_search

    start = time.perf_counter()
    results = await _text_search(query, agent_id, user_id, scope_filter, limit)
    elapsed = (time.perf_counter() - start) * 1000
    return results, elapsed


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = 60,
    limit: int = 10,
) -> List[Dict]:
    """Merge results using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = str(doc.get("_id", id(doc)))
            rrf_score = 1.0 / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in docs:
                docs[doc_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
    return [{**docs[doc_id], "rrf_score": scores[doc_id]} for doc_id in sorted_ids]


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
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    parser.add_argument("--check-indexes", action="store_true", help="Check indexes only")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MEMORY2 RAG TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    if args.check_indexes:
        await check_indexes_and_stats()
        return

    # Resolve agent ID
    agent_id = ObjectId(args.agent_id) if args.agent_id else await get_sample_agent_id()
    if not agent_id:
        print("\n❌ No facts found. Provide --agent-id or ensure facts exist.")
        return

    user_id = ObjectId(args.user_id) if args.user_id else None
    scope_filter = ["user", "agent"] if user_id else ["agent"]

    print(f"\nAgent: {agent_id}")
    print(f"Query: '{args.query}'")
    print(f"Limit: {args.limit}")

    # =========================================================================
    # STEP 1: Generate embedding (this is the slow part)
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
    semantic_task = run_semantic_search(embedding, pre_filter, args.limit * 2)
    text_task = run_text_search(args.query, agent_id, user_id, scope_filter, args.limit * 2)

    (semantic_results, semantic_time), (text_results, text_time) = await asyncio.gather(
        semantic_task, text_task
    )
    parallel_time = (time.perf_counter() - search_start) * 1000

    print(f"2. SEMANTIC SEARCH:  {semantic_time:7.1f}ms  ({len(semantic_results)} results)")
    print(f"3. TEXT SEARCH:      {text_time:7.1f}ms  ({len(text_results)} results)")
    print(f"   (parallel wall):  {parallel_time:7.1f}ms")

    # =========================================================================
    # STEP 3: Merge with RRF
    # =========================================================================
    rrf_start = time.perf_counter()
    merged_results = reciprocal_rank_fusion([semantic_results, text_results], k=60, limit=args.limit)
    rrf_time = (time.perf_counter() - rrf_start) * 1000
    print(f"4. RRF MERGE:        {rrf_time:7.1f}ms  ({len(merged_results)} results)")

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
    # Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("HYBRID RESULTS (RRF merged)")
    print("=" * 60)
    for i, r in enumerate(merged_results, 1):
        rrf_score = r.get("rrf_score", 0)
        scope = r.get("scope", "?")
        content = r.get("content", "")[:75]
        print(f"{i:2}. [{scope}] (rrf: {rrf_score:.4f}) {content}...")

    print("\n" + "=" * 60)
    print("TOOL RESPONSE")
    print("=" * 60)
    print(tool_response)


if __name__ == "__main__":
    asyncio.run(main())
