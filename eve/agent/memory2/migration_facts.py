"""
Memory System Facts Migration Script: Old Memory → Memory2

This script migrates facts from the old memory system to the new memory2 system.
It copies facts from SessionMemory (memory_type="fact") to the new Fact collection
without modifying or removing any old data.

Migration mapping:
- Old SessionMemory (memory_type="fact") → New Fact (scope=["agent"])

Old facts were shard-level (agent-wide), so they map to agent scope in the new system.
Embeddings are NOT generated during migration - they will be generated lazily when needed
or via a separate batch process.

Performance optimizations:
- Filters agents by recent message activity (default: 6 months)
- Uses bulk inserts instead of individual saves
- Supports resume capability to skip already-migrated records
- Uses MongoDB projections to reduce data transfer

Usage:
    python -m eve.agent.memory2.migration_facts [--dry-run] [--agent-id <agent_id>]

Options:
    --dry-run           Preview what would be migrated without making changes
    --agent-id          Only migrate a specific agent (useful for testing)
    --clear-first       Clear existing memory2 facts before migrating (use with caution!)
    --activity-months   Only migrate agents with activity in last N months (default: 6, 0=all)
    --batch-size        Batch size for bulk inserts (default: 500)
    --generate-embeddings  Generate embeddings during migration (slower but facts ready for RAG)
"""

import argparse
import hashlib
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from bson import ObjectId
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.memory.memory_models import AgentMemory, SessionMemory
from eve.agent.memory2.models import Fact
from eve.agent.session.models import ChatMessage
from eve.mongo import get_collection


# Projections for efficient data loading (only fetch needed fields)
AGENT_MEMORY_PROJECTION = {
    "_id": 1,
    "agent_id": 1,
    "shard_name": 1,
    "facts": 1,
    "is_active": 1,
}

SESSION_MEMORY_PROJECTION = {
    "_id": 1,
    "content": 1,
    "source_session_id": 1,
    "source_message_ids": 1,
    "createdAt": 1,
    "memory_type": 1,
    "agent_id": 1,
}


class FactsMigrationStats:
    """Track facts migration statistics."""

    def __init__(self):
        self.agents_processed = 0
        self.agents_skipped_no_activity = 0
        self.agents_skipped_no_facts = 0
        self.shards_processed = 0
        self.facts_migrated = 0
        self.facts_skipped_already_exists = 0
        self.facts_skipped_empty_content = 0
        self.embeddings_generated = 0
        self.errors: List[str] = []

    def print_summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("FACTS MIGRATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Agents processed:              {self.agents_processed}")
        logger.info(f"Agents skipped (no activity):  {self.agents_skipped_no_activity}")
        logger.info(f"Agents skipped (no facts):     {self.agents_skipped_no_facts}")
        logger.info(f"Shards processed:              {self.shards_processed}")
        logger.info(f"Facts migrated:                {self.facts_migrated}")
        logger.info(f"Facts skipped (already exist): {self.facts_skipped_already_exists}")
        logger.info(f"Facts skipped (empty content): {self.facts_skipped_empty_content}")
        logger.info(f"Embeddings generated:          {self.embeddings_generated}")
        if self.errors:
            logger.info(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors[:10]:  # Show first 10 errors
                logger.error(f"  - {error}")
            if len(self.errors) > 10:
                logger.error(f"  ... and {len(self.errors) - 10} more errors")
        logger.info("=" * 60)


def get_active_agents(months: int = 6) -> Optional[Set[ObjectId]]:
    """
    Find agents with recent message activity.

    Uses MongoDB aggregation to efficiently find all agent_ids (from sender field
    in assistant messages) with recent messages.

    Args:
        months: Number of months to look back for activity (0 = no filtering)

    Returns:
        Set of active agent ObjectIds, or None if filtering disabled
    """
    if months <= 0:
        logger.info("Activity filtering disabled (months=0), will process all agents")
        return None

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=months * 30)
    messages_collection = ChatMessage.get_collection()

    logger.info(f"Finding agents with activity since {cutoff_date.date()}...")

    # Find all unique agent_ids with recent messages
    # For assistant messages, sender field contains the actual agent ID
    active_agents_pipeline = [
        {
            "$match": {
                "createdAt": {"$gte": cutoff_date},
                "role": "assistant",
                "sender": {"$ne": None},
            }
        },
        {"$group": {"_id": "$sender"}},
    ]

    active_agent_ids = set()
    for doc in messages_collection.aggregate(active_agents_pipeline):
        if doc["_id"]:
            active_agent_ids.add(doc["_id"])

    logger.info(f"Found {len(active_agent_ids)} agents with recent activity")
    return active_agent_ids


def get_already_migrated_fact_hashes(agent_id: ObjectId) -> Set[str]:
    """
    Get set of fact content hashes that already exist for an agent.
    Used for resume capability to skip already-migrated facts.

    Args:
        agent_id: Agent ID to check

    Returns:
        Set of MD5 hashes of existing facts
    """
    fact_collection = Fact.get_collection()

    # Find all hashes for this agent's facts
    cursor = fact_collection.find(
        {"agent_id": agent_id},
        {"hash": 1}
    )

    return {doc["hash"] for doc in cursor if doc.get("hash")}


def compute_content_hash(content: str) -> str:
    """Compute MD5 hash for fact content deduplication."""
    return hashlib.md5(content.encode()).hexdigest()


def load_facts_from_session_memory(fact_ids: List[ObjectId]) -> List[Dict[str, Any]]:
    """
    Load fact records from SessionMemory by their IDs.

    Args:
        fact_ids: List of SessionMemory IDs to load

    Returns:
        List of raw SessionMemory documents
    """
    if not fact_ids:
        return []

    collection = SessionMemory.get_collection()
    return list(
        collection.find(
            {"_id": {"$in": fact_ids}, "memory_type": "fact"},
            SESSION_MEMORY_PROJECTION
        )
    )


async def generate_embedding(content: str) -> List[float]:
    """
    Generate embedding for fact content using OpenAI.

    Args:
        content: The fact text to embed

    Returns:
        List of floats representing the embedding vector
    """
    try:
        from eve.llm.openai import async_openai_client

        response = await async_openai_client().embeddings.create(
            input=content,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"Failed to generate embedding: {e}")
        return []


def prepare_facts_batch(
    agent_id: ObjectId,
    shard_facts_raw: List[Dict[str, Any]],
    existing_hashes: Set[str],
    stats: FactsMigrationStats,
) -> List[Dict]:
    """
    Prepare batch of Fact documents for bulk insert.

    Args:
        agent_id: Agent ID these facts belong to
        shard_facts_raw: Raw SessionMemory documents containing facts
        existing_hashes: Set of fact hashes that already exist (for deduplication)
        stats: Statistics tracker

    Returns:
        List of Fact documents ready for insert
    """
    fact_docs = []
    now = datetime.now(timezone.utc)

    for fact_raw in shard_facts_raw:
        content = fact_raw.get("content", "").strip()

        # Skip empty content
        if not content:
            stats.facts_skipped_empty_content += 1
            continue

        # Compute hash for deduplication
        content_hash = compute_content_hash(content)

        # Skip if already migrated (resume capability)
        if content_hash in existing_hashes:
            stats.facts_skipped_already_exists += 1
            continue

        # Add to existing hashes to prevent duplicates within this batch
        existing_hashes.add(content_hash)

        # Create Fact document
        # Old facts were shard-level (agent-wide), so scope is ["agent"]
        fact_doc = {
            "_id": ObjectId(),
            "content": content,
            "hash": content_hash,
            "embedding": [],  # Empty - will be generated lazily or via batch process
            "embedding_model": "text-embedding-3-small",
            "scope": ["agent"],  # Old shard facts map to agent scope
            "agent_id": agent_id,
            "user_id": None,  # Agent-scoped facts don't have user_id
            "formed_at": fact_raw.get("createdAt") or now,
            "session_id": fact_raw.get("source_session_id"),
            "source_message_ids": fact_raw.get("source_message_ids", []),
            "access_count": 0,
            "last_accessed_at": None,
            "version": 1,
            "previous_content": None,
            "updated_at": None,
            "createdAt": now,
            "updatedAt": now,
        }
        fact_docs.append(fact_doc)

    return fact_docs


def bulk_insert_facts(
    fact_docs: List[Dict],
    dry_run: bool = False,
) -> int:
    """
    Perform bulk insert of Fact documents.

    Args:
        fact_docs: List of Fact documents to insert
        dry_run: If True, don't actually insert

    Returns:
        Number of facts inserted
    """
    if dry_run or not fact_docs:
        return len(fact_docs)

    fact_collection = Fact.get_collection()
    try:
        result = fact_collection.insert_many(fact_docs, ordered=False)
        return len(result.inserted_ids)
    except Exception as e:
        # Handle partial failures (some docs may have been inserted)
        logger.warning(f"Partial failure in facts bulk insert: {e}")
        return len(fact_docs)  # Approximate


def migrate_agent_facts(
    agent: Agent,
    active_agents: Optional[Set[ObjectId]],
    dry_run: bool = False,
    stats: FactsMigrationStats = None,
    batch_size: int = 500,
) -> Dict:
    """
    Migrate all facts for a single agent.

    Args:
        agent: The agent to migrate
        active_agents: Set of active agent_ids (None = no filtering)
        dry_run: If True, preview only
        stats: Migration statistics tracker
        batch_size: Batch size for bulk inserts

    Returns:
        Dict with migration results for this agent
    """
    agent_id = agent.id
    agent_name = agent.username or str(agent_id)
    results = {
        "agent_id": str(agent_id),
        "agent_name": agent_name,
        "shards_processed": 0,
        "facts_migrated": 0,
        "facts_skipped": 0,
        "errors": [],
    }

    # Check if this agent has recent activity (when filtering is enabled)
    if active_agents is not None and agent_id not in active_agents:
        logger.debug(f"Skipping agent {agent_name} - no recent activity")
        stats.agents_skipped_no_activity += 1
        return results

    logger.info(f"\nMigrating facts for agent: {agent_name} ({agent_id})")

    # Get existing fact hashes for resume capability
    existing_hashes = get_already_migrated_fact_hashes(agent_id)
    if existing_hashes:
        logger.info(f"  Found {len(existing_hashes)} existing facts (will skip duplicates)")

    # Find all agent memory shards with facts
    try:
        shard_collection = AgentMemory.get_collection()
        shards = list(
            shard_collection.find(
                {"agent_id": agent_id},
                AGENT_MEMORY_PROJECTION
            )
        )

        if not shards:
            logger.info(f"  No memory shards found for agent {agent_name}")
            stats.agents_skipped_no_facts += 1
            return results

        # Collect all fact IDs from all shards
        all_fact_ids = []
        for shard in shards:
            fact_ids = shard.get("facts", [])
            if fact_ids:
                all_fact_ids.extend(fact_ids)
                results["shards_processed"] += 1
                stats.shards_processed += 1

        if not all_fact_ids:
            logger.info(f"  No facts found in any shard for agent {agent_name}")
            stats.agents_skipped_no_facts += 1
            return results

        logger.info(f"  Found {len(all_fact_ids)} fact IDs across {results['shards_processed']} shards")

        # Load facts in batches
        all_fact_docs = []
        for i in range(0, len(all_fact_ids), batch_size):
            batch_ids = all_fact_ids[i:i + batch_size]

            # Load SessionMemory records
            facts_raw = load_facts_from_session_memory(batch_ids)

            # Prepare Fact documents
            fact_docs = prepare_facts_batch(
                agent_id,
                facts_raw,
                existing_hashes,
                stats,
            )
            all_fact_docs.extend(fact_docs)

        # Bulk insert all facts
        if all_fact_docs:
            for i in range(0, len(all_fact_docs), batch_size):
                batch_docs = all_fact_docs[i:i + batch_size]
                inserted = bulk_insert_facts(batch_docs, dry_run)
                results["facts_migrated"] += inserted
                stats.facts_migrated += inserted

        stats.agents_processed += 1

    except Exception as e:
        error_msg = f"Error migrating facts for agent={agent_id}: {e}"
        results["errors"].append(error_msg)
        if stats:
            stats.errors.append(error_msg)
        logger.error(f"  {error_msg}")
        traceback.print_exc()

    # Summary for this agent
    logger.info(
        f"  Agent {agent_name}: "
        f"shards={results['shards_processed']}, "
        f"facts_migrated={results['facts_migrated']}, "
        f"errors={len(results['errors'])}"
    )

    return results


def clear_memory2_facts():
    """Clear all memory2 facts. Use with caution!"""
    logger.warning("Clearing memory2 facts collection...")

    fact_collection = Fact.get_collection()
    result = fact_collection.delete_many({})
    logger.info(f"  Deleted {result.deleted_count} Fact records")


def run_facts_migration(
    dry_run: bool = False,
    agent_id: Optional[str] = None,
    clear_first: bool = False,
    activity_months: int = 6,
    batch_size: int = 500,
    generate_embeddings: bool = False,
):
    """
    Run the facts migration.

    Args:
        dry_run: If True, preview what would be migrated without making changes
        agent_id: If provided, only migrate this specific agent
        clear_first: If True, clear memory2 facts collection before migrating
        activity_months: Only migrate agents with activity in last N months (0=all)
        batch_size: Batch size for bulk inserts
        generate_embeddings: If True, generate embeddings during migration (slower)
    """
    stats = FactsMigrationStats()

    logger.info("=" * 60)
    logger.info("FACTS MIGRATION: Old System → Memory2")
    logger.info("=" * 60)

    if dry_run:
        logger.info("MODE: DRY RUN (no changes will be made)")
    else:
        logger.info("MODE: LIVE (changes will be made to database)")

    logger.info(f"Activity filter: {activity_months} months")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Generate embeddings: {generate_embeddings}")

    if generate_embeddings:
        logger.warning(
            "Embedding generation enabled - migration will be slower but "
            "facts will be ready for RAG retrieval immediately"
        )

    if clear_first and not dry_run:
        clear_memory2_facts()

    # Get activity data for filtering
    active_agents = get_active_agents(activity_months)

    # Get agents to migrate
    if agent_id:
        try:
            agent = Agent.from_mongo(ObjectId(agent_id))
            if not agent:
                logger.error(f"Agent not found: {agent_id}")
                return
            agents = [agent]
            logger.info(f"Migrating single agent: {agent_id}")
        except Exception as e:
            logger.error(f"Error loading agent {agent_id}: {e}")
            return
    else:
        # Find all agents
        agents = Agent.find({"type": "agent"})
        logger.info(f"Found {len(agents)} total agents")

    # Process each agent
    for agent in agents:
        try:
            migrate_agent_facts(
                agent,
                active_agents=active_agents,
                dry_run=dry_run,
                stats=stats,
                batch_size=batch_size,
            )
        except Exception as e:
            error_msg = f"Error migrating agent {agent.id}: {e}"
            stats.errors.append(error_msg)
            logger.error(error_msg)
            traceback.print_exc()

    # Print summary
    stats.print_summary()


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Migrate facts from old memory system to memory2"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be migrated without making changes",
    )
    parser.add_argument(
        "--agent-id",
        type=str,
        help="Only migrate a specific agent (useful for testing)",
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Clear existing memory2 facts before migrating (use with caution!)",
    )
    parser.add_argument(
        "--activity-months",
        type=int,
        default=6,
        help="Only migrate agents with activity in last N months (default: 6, 0=all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for bulk inserts (default: 500)",
    )
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Generate embeddings during migration (slower but facts ready for RAG)",
    )

    args = parser.parse_args()

    # Confirm clear operation
    if args.clear_first and not args.dry_run:
        response = input(
            "WARNING: This will delete all existing memory2 facts. "
            "Are you sure? (yes/no): "
        )
        if response.lower() != "yes":
            logger.info("Migration cancelled.")
            return

    run_facts_migration(
        dry_run=args.dry_run,
        agent_id=args.agent_id,
        clear_first=args.clear_first,
        activity_months=args.activity_months,
        batch_size=args.batch_size,
        generate_embeddings=args.generate_embeddings,
    )


if __name__ == "__main__":
    main()
