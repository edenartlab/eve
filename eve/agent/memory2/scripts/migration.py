"""
Memory System Migration Script: Old Memory → Memory2

This script migrates memory data from the old memory system to the new memory2 system.
It copies consolidated memory blobs and unabsorbed reflections without modifying the old data.

Migration mapping:
- Old UserMemory → New ConsolidatedMemory (scope="user")
- Old AgentMemory shards → New ConsolidatedMemory (scope="agent") [merged into one blob]
- Old SessionMemory (directive) → New Reflection (scope="user")
- Old SessionMemory (suggestion) → New Reflection (scope="agent")

Note: Session-level memories are NOT migrated (they will be lost, as per requirements).
Note: Facts are NOT migrated (those will follow later).

Performance optimizations:
- Filters agents/users by recent message activity (default: 6 months)
- Uses bulk inserts instead of individual saves
- Supports resume capability to skip already-migrated records
- Uses MongoDB projections to reduce data transfer

Usage:
    python -m eve.agent.memory2.migration [--dry-run] [--agent-id <agent_id>]

Options:
    --dry-run           Preview what would be migrated without making changes
    --agent-id          Only migrate a specific agent (useful for testing)
    --clear-first       Clear existing memory2 collections before migrating (use with caution!)
    --activity-months   Only migrate agents/users with activity in last N months (default: 6, 0=all)
    --batch-size        Batch size for bulk inserts (default: 500)
"""

import argparse
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from bson import ObjectId
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.memory.memory_models import AgentMemory, SessionMemory, UserMemory
from eve.agent.memory2.constants import (
    CONSOLIDATED_WORD_LIMITS,
)
from eve.agent.memory2.models import ConsolidatedMemory, Reflection
from eve.agent.session.models import ChatMessage, Session
from eve.mongo import get_collection


# Projections for efficient data loading (only fetch needed fields)
USER_MEMORY_PROJECTION = {
    "_id": 1,
    "agent_id": 1,
    "user_id": 1,
    "content": 1,
    "unabsorbed_memory_ids": 1,
    "last_updated_at": 1,
}

AGENT_MEMORY_PROJECTION = {
    "_id": 1,
    "agent_id": 1,
    "shard_name": 1,
    "content": 1,
    "unabsorbed_memory_ids": 1,
    "last_updated_at": 1,
    "is_active": 1,
}

SESSION_MEMORY_PROJECTION = {
    "_id": 1,
    "content": 1,
    "source_session_id": 1,
    "source_message_ids": 1,
    "createdAt": 1,
    "memory_type": 1,
}


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.agents_processed = 0
        self.agents_skipped_no_activity = 0
        self.agents_skipped_already_migrated = 0
        self.user_memories_migrated = 0
        self.user_memories_skipped = 0
        self.agent_memories_migrated = 0
        self.user_reflections_created = 0
        self.agent_reflections_created = 0
        self.errors: List[str] = []

    def print_summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Agents processed:              {self.agents_processed}")
        logger.info(f"Agents skipped (no activity):  {self.agents_skipped_no_activity}")
        logger.info(
            f"Agents skipped (already done): {self.agents_skipped_already_migrated}"
        )
        logger.info(f"User memories migrated:        {self.user_memories_migrated}")
        logger.info(f"User memories skipped:         {self.user_memories_skipped}")
        logger.info(f"Agent memories migrated:       {self.agent_memories_migrated}")
        logger.info(f"User reflections created:      {self.user_reflections_created}")
        logger.info(f"Agent reflections created:     {self.agent_reflections_created}")
        if self.errors:
            logger.info(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors[:10]:  # Show first 10 errors
                logger.error(f"  - {error}")
            if len(self.errors) > 10:
                logger.error(f"  ... and {len(self.errors) - 10} more errors")
        logger.info("=" * 60)


def get_active_agents_and_users(
    months: int = 6,
) -> Tuple[Set[ObjectId], Dict[ObjectId, Set[ObjectId]]]:
    """
    Find agents and users with recent message activity.

    Uses MongoDB aggregation to efficiently find:
    1. All agent_ids (from agent_owner field) with recent messages
    2. All (agent_id, user_id) pairs with recent messages

    Args:
        months: Number of months to look back for activity (0 = no filtering)

    Returns:
        Tuple of:
        - Set of active agent ObjectIds
        - Dict mapping agent_id -> Set of active user_ids for that agent
    """
    if months <= 0:
        logger.info("Activity filtering disabled (months=0), will process all agents")
        return None, None

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=months * 30)
    messages_collection = ChatMessage.get_collection()

    logger.info(f"Finding agents/users with activity since {cutoff_date.date()}...")

    # Find all unique agent_ids with recent messages
    # For assistant messages, sender field contains the actual agent ID
    # (agent_owner field contains the owner USER, not the agent)
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

    # Find all unique (agent, user) pairs from sessions with recent messages
    # Messages have session field, sessions have agents and users fields
    # Use aggregation to join messages -> sessions and extract pairs
    sessions_collection = Session.get_collection()

    # First find session IDs with recent messages
    sessions_pipeline = [
        {"$match": {"createdAt": {"$gte": cutoff_date}, "session": {"$ne": []}}},
        {"$unwind": "$session"},
        {"$group": {"_id": "$session"}},
    ]

    active_session_ids = set()
    for doc in messages_collection.aggregate(sessions_pipeline):
        if doc["_id"]:
            active_session_ids.add(doc["_id"])

    logger.info(f"Found {len(active_session_ids)} sessions with recent activity")

    # Now get (agent, user) pairs from those sessions
    agent_to_users: Dict[ObjectId, Set[ObjectId]] = {}
    if active_session_ids:
        # Query sessions in batches to avoid cursor timeout
        session_ids_list = list(active_session_ids)
        batch_size = 1000
        for i in range(0, len(session_ids_list), batch_size):
            batch_ids = session_ids_list[i : i + batch_size]
            for session_doc in sessions_collection.find(
                {"_id": {"$in": batch_ids}},
                {"agents": 1, "users": 1}
            ):
                agents = session_doc.get("agents") or []
                users = session_doc.get("users") or []
                # Create pairs of each agent with each user in the session
                for agent_id in agents:
                    if agent_id not in agent_to_users:
                        agent_to_users[agent_id] = set()
                    for user_id in users:
                        agent_to_users[agent_id].add(user_id)

    total_pairs = sum(len(users) for users in agent_to_users.values())
    logger.info(f"Found {total_pairs} active (agent, user) pairs")

    return active_agent_ids, agent_to_users


def get_already_migrated_agents() -> Set[ObjectId]:
    """
    Get set of agent_ids that already have agent-level ConsolidatedMemory.
    Used for resume capability.
    """
    consolidated_collection = ConsolidatedMemory.get_collection()

    # Find all agent_ids with scope="agent" (indicates fully migrated)
    migrated = consolidated_collection.distinct(
        "agent_id", {"scope": "agent"}
    )

    return set(migrated)


def get_already_migrated_user_pairs() -> Set[Tuple[ObjectId, ObjectId]]:
    """
    Get set of (agent_id, user_id) pairs that already have user-level ConsolidatedMemory.
    Used for resume capability.
    """
    consolidated_collection = ConsolidatedMemory.get_collection()

    # Find all (agent_id, user_id) pairs with scope="user"
    cursor = consolidated_collection.find(
        {"scope": "user"},
        {"agent_id": 1, "user_id": 1}
    )

    migrated = set()
    for doc in cursor:
        if doc.get("agent_id") and doc.get("user_id"):
            migrated.add((doc["agent_id"], doc["user_id"]))

    return migrated


def load_unabsorbed_memories_raw(
    memory_ids: List[ObjectId], memory_type: str
) -> List[Dict[str, Any]]:
    """Load unabsorbed memories by their IDs using projection."""
    if not memory_ids:
        return []

    collection = SessionMemory.get_collection()
    query = {"_id": {"$in": memory_ids}, "memory_type": memory_type}
    return list(collection.find(query, SESSION_MEMORY_PROJECTION))


def prepare_user_memory_batch(
    user_memories_raw: List[Dict[str, Any]],
    migrated_pairs: Set[Tuple[ObjectId, ObjectId]],
    active_users: Optional[Set[ObjectId]],
    stats: MigrationStats,
) -> Tuple[List[Dict], List[Dict], Dict[str, List[ObjectId]]]:
    """
    Prepare batch of ConsolidatedMemory and Reflection documents for bulk insert.

    Args:
        user_memories_raw: Raw user memory documents from MongoDB
        migrated_pairs: Set of already migrated (agent_id, user_id) pairs
        active_users: Set of active user_ids (None = no filtering)
        stats: Statistics tracker

    Returns:
        Tuple of:
        - List of ConsolidatedMemory documents ready for insert
        - List of Reflection documents ready for insert
        - Dict mapping consolidated_memory temp_id -> list of reflection temp_ids
    """
    consolidated_docs = []
    reflection_docs = []
    consolidated_to_reflections: Dict[str, List[str]] = {}

    for um in user_memories_raw:
        agent_id = um.get("agent_id")
        user_id = um.get("user_id")

        if not agent_id or not user_id:
            continue

        # Skip if already migrated (resume capability)
        if (agent_id, user_id) in migrated_pairs:
            stats.user_memories_skipped += 1
            continue

        # Skip if user not in active set (when filtering is enabled)
        if active_users is not None and user_id not in active_users:
            stats.user_memories_skipped += 1
            continue

        # Generate temporary ID for tracking relationships
        temp_consolidated_id = ObjectId()

        # Create ConsolidatedMemory document
        now = datetime.now(timezone.utc)
        consolidated_doc = {
            "_id": temp_consolidated_id,
            "scope": "user",
            "agent_id": agent_id,
            "user_id": user_id,
            "consolidated_content": um.get("content") or "",
            "word_limit": CONSOLIDATED_WORD_LIMITS.get("user", 400),
            "unabsorbed_ids": [],  # Will be populated after reflection insert
            "last_consolidated_at": um.get("last_updated_at"),
            "createdAt": now,
            "updatedAt": now,
        }

        # Load and prepare reflection documents for unabsorbed directives
        unabsorbed_ids = um.get("unabsorbed_memory_ids", [])
        reflection_temp_ids = []

        if unabsorbed_ids:
            directives = load_unabsorbed_memories_raw(unabsorbed_ids, "directive")

            for directive in directives:
                temp_reflection_id = ObjectId()
                reflection_temp_ids.append(temp_reflection_id)

                reflection_doc = {
                    "_id": temp_reflection_id,
                    "content": directive.get("content", ""),
                    "scope": "user",
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "session_id": directive.get("source_session_id"),
                    "source_message_ids": directive.get("source_message_ids", []),
                    "formed_at": directive.get("createdAt") or now,
                    "absorbed": False,
                    "createdAt": now,
                    "updatedAt": now,
                }
                reflection_docs.append(reflection_doc)

        # Update consolidated doc with reflection IDs
        consolidated_doc["unabsorbed_ids"] = reflection_temp_ids
        consolidated_docs.append(consolidated_doc)

        stats.user_memories_migrated += 1
        stats.user_reflections_created += len(reflection_temp_ids)

    return consolidated_docs, reflection_docs, consolidated_to_reflections


def prepare_agent_memory_batch(
    agent_id: ObjectId,
    migrated_agents: Set[ObjectId],
    stats: MigrationStats,
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Prepare agent-level ConsolidatedMemory and Reflection documents.

    Args:
        agent_id: The agent to migrate
        migrated_agents: Set of already migrated agent_ids
        stats: Statistics tracker

    Returns:
        Tuple of:
        - ConsolidatedMemory document (or None if skipped)
        - List of Reflection documents
    """
    # Skip if already migrated (resume capability)
    if agent_id in migrated_agents:
        return None, []

    # Find all active shards using projection
    collection = AgentMemory.get_collection()
    shards = list(
        collection.find(
            {"agent_id": agent_id, "is_active": True},
            AGENT_MEMORY_PROJECTION
        )
    )

    if not shards:
        return None, []

    # Merge all shard contents into one blob
    merged_content_parts = []
    all_suggestion_ids = []
    latest_updated_at = None

    for shard in shards:
        shard_name = shard.get("shard_name") or "Unknown Shard"

        # Add shard content with header
        if shard.get("content"):
            merged_content_parts.append(f"## {shard_name}\n\n{shard['content']}")

        # Collect unabsorbed suggestion IDs
        unabsorbed_ids = shard.get("unabsorbed_memory_ids", [])
        if unabsorbed_ids:
            all_suggestion_ids.extend(unabsorbed_ids)

        # Track latest update time
        shard_updated = shard.get("last_updated_at")
        if shard_updated:
            if latest_updated_at is None or shard_updated > latest_updated_at:
                latest_updated_at = shard_updated

    # Create consolidated memory document
    now = datetime.now(timezone.utc)
    merged_content = "\n\n".join(merged_content_parts)
    temp_consolidated_id = ObjectId()

    consolidated_doc = {
        "_id": temp_consolidated_id,
        "scope": "agent",
        "agent_id": agent_id,
        "consolidated_content": merged_content,
        "word_limit": CONSOLIDATED_WORD_LIMITS.get("agent", 1000),
        "unabsorbed_ids": [],
        "last_consolidated_at": latest_updated_at,
        "createdAt": now,
        "updatedAt": now,
    }

    # Prepare reflection documents for unabsorbed suggestions
    reflection_docs = []

    if all_suggestion_ids:
        suggestions = load_unabsorbed_memories_raw(all_suggestion_ids, "suggestion")

        for suggestion in suggestions:
            temp_reflection_id = ObjectId()
            consolidated_doc["unabsorbed_ids"].append(temp_reflection_id)

            reflection_doc = {
                "_id": temp_reflection_id,
                "content": suggestion.get("content", ""),
                "scope": "agent",
                "agent_id": agent_id,
                "session_id": suggestion.get("source_session_id"),
                "source_message_ids": suggestion.get("source_message_ids", []),
                "formed_at": suggestion.get("createdAt") or now,
                "absorbed": False,
                "createdAt": now,
                "updatedAt": now,
            }
            reflection_docs.append(reflection_doc)

    stats.agent_memories_migrated += 1
    stats.agent_reflections_created += len(reflection_docs)

    return consolidated_doc, reflection_docs


def bulk_insert_documents(
    consolidated_docs: List[Dict],
    reflection_docs: List[Dict],
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Perform bulk insert of ConsolidatedMemory and Reflection documents.

    Args:
        consolidated_docs: List of ConsolidatedMemory documents
        reflection_docs: List of Reflection documents
        dry_run: If True, don't actually insert

    Returns:
        Tuple of (consolidated_count, reflection_count) inserted
    """
    if dry_run:
        return len(consolidated_docs), len(reflection_docs)

    consolidated_count = 0
    reflection_count = 0

    # Insert reflections first (consolidated docs reference them)
    if reflection_docs:
        reflection_collection = Reflection.get_collection()
        try:
            result = reflection_collection.insert_many(reflection_docs, ordered=False)
            reflection_count = len(result.inserted_ids)
        except Exception as e:
            # Handle partial failures (some docs may have been inserted)
            logger.warning(f"Partial failure in reflection bulk insert: {e}")
            # Try to count what was inserted
            reflection_count = len(reflection_docs)  # Approximate

    # Insert consolidated memories
    if consolidated_docs:
        consolidated_collection = ConsolidatedMemory.get_collection()
        try:
            result = consolidated_collection.insert_many(
                consolidated_docs, ordered=False
            )
            consolidated_count = len(result.inserted_ids)
        except Exception as e:
            logger.warning(f"Partial failure in consolidated bulk insert: {e}")
            consolidated_count = len(consolidated_docs)  # Approximate

    return consolidated_count, reflection_count


def migrate_agent_optimized(
    agent: Agent,
    active_users: Optional[Set[ObjectId]],
    migrated_agents: Set[ObjectId],
    migrated_user_pairs: Set[Tuple[ObjectId, ObjectId]],
    dry_run: bool = False,
    stats: MigrationStats = None,
    batch_size: int = 500,
) -> Dict:
    """
    Migrate all memory for a single agent using optimized bulk operations.

    Args:
        agent: The agent to migrate
        active_users: Set of active user_ids for this agent (None = no filtering)
        migrated_agents: Set of already migrated agent_ids
        migrated_user_pairs: Set of already migrated (agent_id, user_id) pairs
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
        "user_memories": 0,
        "agent_memories": 0,
        "user_reflections": 0,
        "agent_reflections": 0,
        "errors": [],
    }

    logger.info(f"\nMigrating agent: {agent_name} ({agent_id})")

    # 1. Migrate UserMemory records using bulk operations
    try:
        # Load user memories using projection
        user_memory_collection = UserMemory.get_collection()
        user_memories_raw = list(
            user_memory_collection.find(
                {"agent_id": agent_id},
                USER_MEMORY_PROJECTION
            )
        )

        if user_memories_raw:
            # Prepare batch
            consolidated_docs, reflection_docs, _ = prepare_user_memory_batch(
                user_memories_raw,
                migrated_user_pairs,
                active_users,
                stats,
            )

            # Bulk insert in batches
            for i in range(0, len(consolidated_docs), batch_size):
                batch_consolidated = consolidated_docs[i : i + batch_size]
                # Collect all reflection IDs referenced by this batch
                batch_reflection_ids = set()
                for doc in batch_consolidated:
                    batch_reflection_ids.update(doc.get("unabsorbed_ids", []))
                batch_reflections = [
                    r for r in reflection_docs if r["_id"] in batch_reflection_ids
                ]

                c_count, r_count = bulk_insert_documents(
                    batch_consolidated, batch_reflections, dry_run
                )
                results["user_memories"] += c_count
                results["user_reflections"] += r_count

    except Exception as e:
        error_msg = f"Error migrating user memories for agent={agent_id}: {e}"
        results["errors"].append(error_msg)
        if stats:
            stats.errors.append(error_msg)
        logger.error(f"  {error_msg}")
        traceback.print_exc()

    # 2. Migrate AgentMemory shards
    try:
        consolidated_doc, reflection_docs = prepare_agent_memory_batch(
            agent_id, migrated_agents, stats
        )

        if consolidated_doc:
            c_count, r_count = bulk_insert_documents(
                [consolidated_doc], reflection_docs, dry_run
            )
            results["agent_memories"] = c_count
            results["agent_reflections"] = r_count

    except Exception as e:
        error_msg = f"Error migrating agent memories for agent={agent_id}: {e}"
        results["errors"].append(error_msg)
        if stats:
            stats.errors.append(error_msg)
        logger.error(f"  {error_msg}")
        traceback.print_exc()

    if stats:
        stats.agents_processed += 1

    # Summary for this agent
    logger.info(
        f"  Agent {agent_name}: "
        f"user_memories={results['user_memories']}, "
        f"agent_memories={results['agent_memories']}, "
        f"user_reflections={results['user_reflections']}, "
        f"agent_reflections={results['agent_reflections']}, "
        f"errors={len(results['errors'])}"
    )

    return results


def clear_memory2_collections():
    """Clear all memory2 collections. Use with caution!"""
    logger.warning("Clearing memory2 collections...")

    # Clear ConsolidatedMemory
    consolidated_collection = ConsolidatedMemory.get_collection()
    result = consolidated_collection.delete_many({})
    logger.info(f"  Deleted {result.deleted_count} ConsolidatedMemory records")

    # Clear Reflection
    reflection_collection = Reflection.get_collection()
    result = reflection_collection.delete_many({})
    logger.info(f"  Deleted {result.deleted_count} Reflection records")


def run_migration(
    dry_run: bool = False,
    agent_id: Optional[str] = None,
    clear_first: bool = False,
    activity_months: int = 6,
    batch_size: int = 500,
):
    """
    Run the memory migration with optimizations.

    Args:
        dry_run: If True, preview what would be migrated without making changes
        agent_id: If provided, only migrate this specific agent
        clear_first: If True, clear memory2 collections before migrating
        activity_months: Only migrate agents/users with activity in last N months (0=all)
        batch_size: Batch size for bulk inserts
    """
    stats = MigrationStats()

    logger.info("=" * 60)
    logger.info("MEMORY MIGRATION: Old System → Memory2 (Optimized)")
    logger.info("=" * 60)

    if dry_run:
        logger.info("MODE: DRY RUN (no changes will be made)")
    else:
        logger.info("MODE: LIVE (changes will be made to database)")

    logger.info(f"Activity filter: {activity_months} months")
    logger.info(f"Batch size: {batch_size}")

    if clear_first and not dry_run:
        clear_memory2_collections()

    # Get activity data for filtering
    active_agent_ids, agent_to_users = get_active_agents_and_users(activity_months)

    # Get already migrated data for resume capability
    logger.info("Checking for already migrated data (resume capability)...")
    migrated_agents = get_already_migrated_agents()
    migrated_user_pairs = get_already_migrated_user_pairs()
    logger.info(
        f"Found {len(migrated_agents)} agents and {len(migrated_user_pairs)} "
        f"user memories already migrated"
    )

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
            # Check if this agent has recent activity (when filtering is enabled)
            if active_agent_ids is not None and agent.id not in active_agent_ids:
                logger.debug(f"Skipping agent {agent.username} - no recent activity")
                stats.agents_skipped_no_activity += 1
                continue

            # Check if already fully migrated
            if agent.id in migrated_agents:
                logger.debug(
                    f"Skipping agent {agent.username} - already migrated (resume)"
                )
                stats.agents_skipped_already_migrated += 1
                continue

            # Get active users for this agent
            active_users = None
            if agent_to_users is not None:
                active_users = agent_to_users.get(agent.id)

            migrate_agent_optimized(
                agent,
                active_users=active_users,
                migrated_agents=migrated_agents,
                migrated_user_pairs=migrated_user_pairs,
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
        description="Migrate memory from old system to memory2 (optimized)"
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
        help="Clear existing memory2 collections before migrating (use with caution!)",
    )
    parser.add_argument(
        "--activity-months",
        type=int,
        default=6,
        help="Only migrate agents/users with activity in last N months (default: 6, 0=all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for bulk inserts (default: 500)",
    )

    args = parser.parse_args()

    # Confirm clear operation
    if args.clear_first and not args.dry_run:
        response = input(
            "WARNING: This will delete all existing memory2 data. "
            "Are you sure? (yes/no): "
        )
        if response.lower() != "yes":
            logger.info("Migration cancelled.")
            return

    run_migration(
        dry_run=args.dry_run,
        agent_id=args.agent_id,
        clear_first=args.clear_first,
        activity_months=args.activity_months,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
