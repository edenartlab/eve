"""
Search Facts Tool Handler

Searches long-term memory facts using hybrid semantic and keyword search
with Reciprocal Rank Fusion (RRF) for result merging.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.constants import (
    RAG_RRF_SCORE_THRESHOLD,
    RAG_SEMANTIC_SCORE_THRESHOLD,
    RAG_TEXT_SCORE_THRESHOLD,
)
from eve.agent.memory2.fact_storage import get_embedding
from eve.agent.memory2.models import Fact
from eve.agent.memory2.rag import (
    _build_atlas_search_filter,
    _build_scope_filter,
    get_text_index_name,
    get_vector_index_name,
)
from eve.tool import ToolContext


def format_age(dt: Optional[datetime]) -> str:
    """Format a datetime as a human-readable age string."""
    if not dt:
        return "unknown age"

    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta = now - dt
    total_seconds = delta.total_seconds()

    if total_seconds < 0:
        return "just now"

    minutes = total_seconds / 60
    hours = minutes / 60
    days = hours / 24
    weeks = days / 7
    months = days / 30.44  # Average days per month
    years = days / 365.25

    if minutes < 1:
        return "just now"
    elif minutes < 60:
        m = int(minutes)
        return f"{m} minute{'s' if m != 1 else ''} old"
    elif hours < 24:
        h = int(hours)
        return f"{h} hour{'s' if h != 1 else ''} old"
    elif days < 7:
        d = int(days)
        return f"{d} day{'s' if d != 1 else ''} old"
    elif months < 1:
        w = int(weeks)
        return f"{w} week{'s' if w != 1 else ''} old"
    elif months < 12:
        m = int(months)
        return f"{m} month{'s' if m != 1 else ''} old"
    else:
        y = int(years)
        return f"{y} year{'s' if y != 1 else ''} old"


async def run_semantic_search(
    embedding: List[float],
    pre_filter: Dict,
    limit: int,
) -> tuple[List[Dict], float, int]:
    """Run vector search with pre-computed embedding. Returns (results, elapsed_ms, filtered_count)."""
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

    loop = asyncio.get_event_loop()
    raw_results = await loop.run_in_executor(
        None, lambda: list(collection.aggregate(pipeline))
    )

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
    """Run text search. Returns (results, elapsed_ms, filtered_count)."""
    start = time.perf_counter()
    collection = Fact.get_collection()

    filter_clauses = _build_atlas_search_filter(agent_id, user_id, scope_filter)

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
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(
            None, lambda: list(collection.aggregate(pipeline))
        )

        results = [
            r for r in raw_results
            if r.get("score", 0) >= RAG_TEXT_SCORE_THRESHOLD
        ]
        filtered_count = len(raw_results) - len(results)

        elapsed = (time.perf_counter() - start) * 1000
        return results, elapsed, filtered_count

    except Exception as e:
        # Fallback for environments without Atlas Search
        if "no such index" in str(e).lower() or "$search" in str(e):
            logger.warning("Text search not available")
            return [], (time.perf_counter() - start) * 1000, 0
        raise


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


async def search_single_query(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId],
    top_k: int,
    debug: bool,
) -> tuple[List[Dict], Optional[Dict]]:
    """
    Execute a single query search with hybrid retrieval.
    Returns (merged_results, debug_info).
    """
    scope_filter = ["user", "agent"] if user_id else ["agent"]
    pre_filter = _build_scope_filter(agent_id, user_id, scope_filter)

    debug_info = None
    if debug:
        debug_info = {"query": query, "stages": {}}

    # Generate embedding
    embed_start = time.perf_counter()
    embedding = await get_embedding(query)
    embed_time = (time.perf_counter() - embed_start) * 1000

    if debug:
        debug_info["stages"]["embedding"] = f"{embed_time:.1f}ms"

    if not embedding or embedding == [0.0] * len(embedding):
        logger.warning(f"Failed to get embedding for query: {query}")
        return [], debug_info

    # Run semantic + text search in parallel
    search_start = time.perf_counter()
    semantic_task = run_semantic_search(embedding, pre_filter, top_k * 2)
    text_task = run_text_search(query, agent_id, user_id, scope_filter, top_k * 2)

    (semantic_results, semantic_time, semantic_filtered), (text_results, text_time, text_filtered) = await asyncio.gather(
        semantic_task, text_task
    )
    parallel_time = (time.perf_counter() - search_start) * 1000

    if debug:
        debug_info["stages"]["semantic_search"] = {
            "time_ms": f"{semantic_time:.1f}",
            "results": len(semantic_results),
            "filtered": semantic_filtered,
            "top_results": [
                {"score": r.get("score", 0), "content": r.get("content", "")[:80]}
                for r in semantic_results[:5]
            ]
        }
        debug_info["stages"]["text_search"] = {
            "time_ms": f"{text_time:.1f}",
            "results": len(text_results),
            "filtered": text_filtered,
            "top_results": [
                {"score": r.get("score", 0), "content": r.get("content", "")[:80]}
                for r in text_results[:5]
            ]
        }
        debug_info["stages"]["parallel_wall_time"] = f"{parallel_time:.1f}ms"

    # RRF merge
    rrf_start = time.perf_counter()
    merged_results, rrf_filtered = reciprocal_rank_fusion(
        [semantic_results, text_results], k=60, limit=top_k
    )
    rrf_time = (time.perf_counter() - rrf_start) * 1000

    if debug:
        debug_info["stages"]["rrf_merge"] = {
            "time_ms": f"{rrf_time:.1f}",
            "results": len(merged_results),
            "filtered": rrf_filtered,
            "top_results": [
                {
                    "rrf_score": r.get("rrf_score", 0),
                    "search_type": r.get("search_type", "?"),
                    "content": r.get("content", "")[:80]
                }
                for r in merged_results[:5]
            ]
        }
        debug_info["total_time_ms"] = f"{embed_time + parallel_time + rrf_time:.1f}"

    return merged_results, debug_info


def format_facts_for_output(facts: List[Dict]) -> str:
    """Format facts with age suffix for tool response."""
    if not facts:
        return "No relevant facts found."

    lines = []
    seen_content = set()

    for fact in facts:
        content = fact.get("content", "").strip()
        if not content or content in seen_content:
            continue
        seen_content.add(content)

        formed_at = fact.get("formed_at")
        age_str = format_age(formed_at)
        lines.append(f"- {content} ({age_str})")

    if not lines:
        return "No relevant facts found."

    return "Retrieved facts:\n\n" + "\n".join(lines)


def print_debug_info(debug_infos: List[Dict]) -> None:
    """Print debug information to terminal."""
    print("\n" + "=" * 70)
    print("SEARCH FACTS DEBUG OUTPUT")
    print("=" * 70)

    for i, info in enumerate(debug_infos, 1):
        if not info:
            continue

        print(f"\n{'─' * 70}")
        print(f"Query {i}: \"{info.get('query', 'N/A')}\"")
        print("─" * 70)

        stages = info.get("stages", {})

        if "embedding" in stages:
            print(f"  1. EMBEDDING:        {stages['embedding']}")

        if "semantic_search" in stages:
            ss = stages["semantic_search"]
            filtered_info = f", {ss['filtered']} filtered" if ss.get('filtered') else ""
            print(f"  2. SEMANTIC SEARCH:  {ss['time_ms']}ms  ({ss['results']} results{filtered_info})")
            for j, r in enumerate(ss.get("top_results", []), 1):
                print(f"      {j}. score={r['score']:.4f} {r['content']}...")

        if "text_search" in stages:
            ts = stages["text_search"]
            filtered_info = f", {ts['filtered']} filtered" if ts.get('filtered') else ""
            print(f"  3. TEXT SEARCH:      {ts['time_ms']}ms  ({ts['results']} results{filtered_info})")
            for j, r in enumerate(ts.get("top_results", []), 1):
                print(f"      {j}. score={r['score']:.4f} {r['content']}...")

        if "parallel_wall_time" in stages:
            print(f"     (parallel wall):  {stages['parallel_wall_time']}")

        if "rrf_merge" in stages:
            rrf = stages["rrf_merge"]
            filtered_info = f", {rrf['filtered']} filtered" if rrf.get('filtered') else ""
            print(f"  4. RRF MERGE:        {rrf['time_ms']}ms  ({rrf['results']} results{filtered_info})")
            for j, r in enumerate(rrf.get("top_results", []), 1):
                print(f"      {j}. rrf={r['rrf_score']:.4f} ({r['search_type']}) {r['content']}...")

        if "total_time_ms" in info:
            print(f"     TOTAL:            {info['total_time_ms']}ms")

    print("\n" + "=" * 70 + "\n")


async def handler(context: ToolContext):
    """Handler for the search_facts tool."""
    user_id = context.user
    agent_id = context.agent

    if not agent_id:
        raise Exception("Agent ID is required")

    agent_oid = ObjectId(agent_id)
    user_oid = ObjectId(user_id) if user_id else None

    queries = context.args.get("query", [])
    top_k = context.args.get("top_k", 10)
    debug = context.args.get("debug", False)

    # Ensure queries is a list
    if isinstance(queries, str):
        queries = [queries]

    if not queries:
        return {"output": "No search queries provided."}

    # Execute all queries in parallel
    tasks = [
        search_single_query(q, agent_oid, user_oid, top_k, debug)
        for q in queries
    ]
    results = await asyncio.gather(*tasks)

    # Collect debug info if enabled
    if debug:
        debug_infos = [r[1] for r in results if r[1]]
        if debug_infos:
            print_debug_info(debug_infos)

    # Combine all facts from all queries
    all_facts = []
    seen_ids = set()

    for facts, _ in results:
        for fact in facts:
            fact_id = str(fact.get("_id", ""))
            if fact_id and fact_id not in seen_ids:
                seen_ids.add(fact_id)
                all_facts.append(fact)

    # Sort by RRF score (highest first) and take top results
    all_facts.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

    # Limit total facts if we had multiple queries
    max_total = top_k * min(len(queries), 3)  # Cap at 3x top_k
    all_facts = all_facts[:max_total]

    output = format_facts_for_output(all_facts)

    return {"output": output}
