#!/usr/bin/env python3
"""
Memory System v2 - RAG Test Script

Test script for validating RAG search functionality on the memory2_facts collection.
This bypasses the RAG_ENABLED flag to allow testing before full rollout.

Usage:
    # Basic test with default agent
    python -m eve.agent.memory2.scripts.test_rag

    # Test with specific agent and query
    python -m eve.agent.memory2.scripts.test_rag --agent-id 507f1f77bcf86cd799439011 --query "user preferences"

    # Test different search types
    python -m eve.agent.memory2.scripts.test_rag --search-type semantic
    python -m eve.agent.memory2.scripts.test_rag --search-type text
    python -m eve.agent.memory2.scripts.test_rag --search-type hybrid

    # Include user-scoped facts
    python -m eve.agent.memory2.scripts.test_rag --user-id 507f1f77bcf86cd799439012

    # Check index status and collection stats
    python -m eve.agent.memory2.scripts.test_rag --check-indexes
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


async def check_indexes_and_stats():
    """Check MongoDB indexes and collection statistics."""
    from eve.agent.memory2.models import Fact

    print("\n" + "=" * 60)
    print("COLLECTION STATUS: memory2_facts")
    print("=" * 60)

    collection = Fact.get_collection()

    # Count documents
    total_count = collection.count_documents({})
    agent_count = collection.count_documents({"scope": "agent"})
    user_count = collection.count_documents({"scope": "user"})

    print(f"\nDocument counts:")
    print(f"  Total facts: {total_count}")
    print(f"  Agent-scoped: {agent_count}")
    print(f"  User-scoped: {user_count}")

    # Check for embeddings
    with_embedding = collection.count_documents({"embedding": {"$exists": True, "$ne": []}})
    without_embedding = collection.count_documents(
        {"$or": [{"embedding": {"$exists": False}}, {"embedding": []}]}
    )

    print(f"\nEmbedding status:")
    print(f"  With embeddings: {with_embedding}")
    print(f"  Without embeddings: {without_embedding}")

    # List indexes
    print(f"\nMongoDB indexes:")
    for index in collection.list_indexes():
        print(f"  - {index['name']}: {index['key']}")

    # Check for Atlas Search indexes (these won't show in regular index list)
    from eve.agent.memory2.rag import get_vector_index_name, get_text_index_name

    vector_index = get_vector_index_name()
    text_index = get_text_index_name()

    print(f"\nAtlas Search indexes (checked via aggregation):")
    print(f"  Environment: DB={os.getenv('DB', 'STAGE')}")

    # Test vector search index
    try:
        # Use a non-zero dummy vector (cosine similarity fails on zero vectors)
        dummy_vector = [0.1] * 1536
        pipeline = [
            {
                "$vectorSearch": {
                    "index": vector_index,
                    "path": "embedding",
                    "queryVector": dummy_vector,
                    "numCandidates": 1,
                    "limit": 1,
                }
            }
        ]
        list(collection.aggregate(pipeline))
        print(f"  - {vector_index}: ✅ Active")
    except Exception as e:
        if "no such index" in str(e).lower() or "vectorSearch" in str(e).lower():
            print(f"  - {vector_index}: ❌ Not found or not ready")
        else:
            print(f"  - {vector_index}: ⚠️ Error: {e}")

    # Test text search index
    try:
        pipeline = [
            {
                "$search": {
                    "index": text_index,
                    "text": {"query": "test", "path": "content"},
                }
            },
            {"$limit": 1},
        ]
        list(collection.aggregate(pipeline))
        print(f"  - {text_index}: ✅ Active")
    except Exception as e:
        if "no such index" in str(e).lower() or "$search" in str(e).lower():
            print(f"  - {text_index}: ❌ Not found or not ready")
        else:
            print(f"  - {text_index}: ⚠️ Error: {e}")

    # Sample some facts
    print(f"\nSample facts (most recent 5):")
    sample_facts = list(
        collection.find({}, {"content": 1, "scope": 1, "formed_at": 1, "agent_id": 1})
        .sort("formed_at", -1)
        .limit(5)
    )

    if not sample_facts:
        print("  (no facts found)")
    else:
        for i, fact in enumerate(sample_facts, 1):
            content = fact.get("content", "")[:60]
            scope = fact.get("scope", "?")
            agent = str(fact.get("agent_id", ""))[-6:]
            print(f"  {i}. [{scope}] (agent:...{agent}) {content}...")


async def test_semantic_search(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    limit: int,
):
    """Test semantic (vector) search only."""
    from eve.agent.memory2.rag import _build_scope_filter, _semantic_search

    print("\n" + "-" * 40)
    print("SEMANTIC SEARCH (Vector)")
    print("-" * 40)

    scope_filter = ["user", "agent"] if user_id else ["agent"]
    pre_filter = _build_scope_filter(agent_id, user_id, scope_filter)

    print(f"Pre-filter: {pre_filter}")

    try:
        results = await _semantic_search(query, pre_filter, limit)

        if not results:
            print("No results found.")
            print("This could mean:")
            print("  - fact_vector_index doesn't exist yet")
            print("  - No facts match the filter criteria")
            print("  - No facts have embeddings")
        else:
            print(f"\nFound {len(results)} results:")
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                scope = r.get("scope", "?")
                content = r.get("content", "")[:80]
                print(f"  {i}. [{scope}] (score: {score:.4f}) {content}...")

        return results

    except Exception as e:
        print(f"Error: {e}")
        if "vectorSearch" in str(e) or "no such index" in str(e).lower():
            print("\n⚠️  Vector search index 'fact_vector_index' not found!")
            print("Create it in MongoDB Atlas → Atlas Search → Create Index")
        return []


async def test_text_search(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    limit: int,
):
    """Test text search only."""
    from eve.agent.memory2.rag import _build_atlas_search_filter, _text_search

    print("\n" + "-" * 40)
    print("TEXT SEARCH (Keyword)")
    print("-" * 40)

    scope_filter = ["user", "agent"] if user_id else ["agent"]
    atlas_filter = _build_atlas_search_filter(agent_id, user_id, scope_filter)

    print(f"Atlas Search filter: {atlas_filter}")

    try:
        results = await _text_search(query, agent_id, user_id, scope_filter, limit)

        if not results:
            print("No results found.")
            print("This could mean:")
            print("  - fact_text_index doesn't exist (will fallback to regex)")
            print("  - No facts contain the query terms")
        else:
            search_type = results[0].get("search_type", "unknown")
            print(f"Search method: {search_type}")
            print(f"\nFound {len(results)} results:")
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                scope = r.get("scope", "?")
                content = r.get("content", "")[:80]
                print(f"  {i}. [{scope}] (score: {score:.4f}) {content}...")

        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []


async def test_hybrid_search(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    limit: int,
):
    """Test hybrid search with RRF fusion."""
    from eve.agent.memory2.rag import search_facts

    print("\n" + "-" * 40)
    print("HYBRID SEARCH (Semantic + Text + RRF)")
    print("-" * 40)

    scope_filter = ["user", "agent"] if user_id else ["agent"]

    try:
        results = await search_facts(
            query=query,
            agent_id=agent_id,
            user_id=user_id,
            scope_filter=scope_filter,
            match_count=limit,
            search_type="hybrid",
        )

        if not results:
            print("No results found.")
        else:
            print(f"\nFound {len(results)} results (RRF fused):")
            for i, r in enumerate(results, 1):
                rrf_score = r.get("rrf_score", r.get("score", 0))
                scope = r.get("scope", "?")
                content = r.get("content", "")[:80]
                print(f"  {i}. [{scope}] (rrf: {rrf_score:.4f}) {content}...")

        return results

    except Exception as e:
        print(f"Error: {e}")
        return []


async def test_tool_handler(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    limit: int,
):
    """Test the tool handler (what agents would call)."""
    from eve.agent.memory2.rag import search_facts
    from eve.agent.memory2.rag_tool import format_facts_for_tool_response

    print("\n" + "-" * 40)
    print("TOOL HANDLER OUTPUT (Agent perspective)")
    print("-" * 40)

    # Bypass RAG_ENABLED check by calling search_facts directly
    try:
        facts = await search_facts(
            query=query,
            agent_id=agent_id,
            user_id=user_id,
            match_count=limit,
        )

        response = format_facts_for_tool_response(facts)
        print(f"\nTool response:\n{response}")

        return response

    except Exception as e:
        print(f"Error: {e}")
        return None


async def get_sample_agent_id():
    """Get a sample agent ID from existing facts."""
    from eve.agent.memory2.models import Fact

    collection = Fact.get_collection()
    sample = collection.find_one({}, {"agent_id": 1})

    if sample and "agent_id" in sample:
        return sample["agent_id"]

    return None


async def main():
    parser = argparse.ArgumentParser(
        description="Test RAG search on memory2_facts collection"
    )
    parser.add_argument(
        "--agent-id",
        type=str,
        help="Agent ID to search facts for (uses sample if not provided)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="User ID to include user-scoped facts",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="project deadline budget",
        help="Search query (default: 'project deadline budget')",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["hybrid", "semantic", "text", "all"],
        default="all",
        help="Search type to test (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max results to return (default: 10)",
    )
    parser.add_argument(
        "--check-indexes",
        action="store_true",
        help="Check index status and collection stats only",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MEMORY2 RAG TEST SCRIPT")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Check indexes if requested
    if args.check_indexes:
        await check_indexes_and_stats()
        return

    # Resolve agent ID
    if args.agent_id:
        agent_id = ObjectId(args.agent_id)
    else:
        agent_id = await get_sample_agent_id()
        if not agent_id:
            print("\n❌ No facts found in database. Cannot determine agent_id.")
            print("   Either provide --agent-id or ensure facts exist.")
            return

    user_id = ObjectId(args.user_id) if args.user_id else None

    print(f"\nTest parameters:")
    print(f"  Agent ID: {agent_id}")
    print(f"  User ID: {user_id or '(none - agent scope only)'}")
    print(f"  Query: '{args.query}'")
    print(f"  Search type: {args.search_type}")
    print(f"  Limit: {args.limit}")

    # Always show index status first
    await check_indexes_and_stats()

    # Run searches
    if args.search_type in ["semantic", "all"]:
        await test_semantic_search(args.query, agent_id, user_id, args.limit)

    if args.search_type in ["text", "all"]:
        await test_text_search(args.query, agent_id, user_id, args.limit)

    if args.search_type in ["hybrid", "all"]:
        await test_hybrid_search(args.query, agent_id, user_id, args.limit)

    # Always show tool handler output
    if args.search_type == "all":
        await test_tool_handler(args.query, agent_id, user_id, args.limit)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
