"""
Memory System Migration Script: Old Memory → Memory2

This script migrates memory data from the old memory system to the new memory2 system.
It copies consolidated memory blobs and unabsorbed reflections without modifying the old data.

Migration mapping:
- Old UserMemory → New ConsolidatedMemory (scope_type="user")
- Old AgentMemory shards → New ConsolidatedMemory (scope_type="agent") [merged into one blob]
- Old SessionMemory (directive) → New Reflection (scope="user")
- Old SessionMemory (suggestion) → New Reflection (scope="agent")

Note: Session-level memories are NOT migrated (they will be lost, as per requirements).
Note: Facts are NOT migrated (those will follow later).

Usage:
    python -m eve.agent.memory2.migration [--dry-run] [--agent-id <agent_id>]

Options:
    --dry-run       Preview what would be migrated without making changes
    --agent-id      Only migrate a specific agent (useful for testing)
    --clear-first   Clear existing memory2 collections before migrating (use with caution!)
"""

import argparse
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from bson import ObjectId
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.memory.memory_models import AgentMemory, SessionMemory, UserMemory
from eve.agent.memory2.constants import (
    CONSOLIDATED_WORD_LIMITS,
)
from eve.agent.memory2.models import ConsolidatedMemory, Reflection


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.agents_processed = 0
        self.user_memories_migrated = 0
        self.agent_memories_migrated = 0
        self.user_reflections_created = 0
        self.agent_reflections_created = 0
        self.errors: List[str] = []

    def print_summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Agents processed:           {self.agents_processed}")
        logger.info(f"User memories migrated:     {self.user_memories_migrated}")
        logger.info(f"Agent memories migrated:    {self.agent_memories_migrated}")
        logger.info(f"User reflections created:   {self.user_reflections_created}")
        logger.info(f"Agent reflections created:  {self.agent_reflections_created}")
        if self.errors:
            logger.info(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors[:10]:  # Show first 10 errors
                logger.error(f"  - {error}")
            if len(self.errors) > 10:
                logger.error(f"  ... and {len(self.errors) - 10} more errors")
        logger.info("=" * 60)


def load_unabsorbed_memories(
    memory_ids: List[ObjectId], memory_type: str
) -> List[SessionMemory]:
    """Load unabsorbed memories by their IDs."""
    if not memory_ids:
        return []

    query = {"_id": {"$in": memory_ids}, "memory_type": memory_type}
    return SessionMemory.find(query)


def migrate_user_memory(
    user_memory: UserMemory, dry_run: bool = False, stats: MigrationStats = None
) -> Tuple[Optional[ConsolidatedMemory], List[Reflection]]:
    """
    Migrate a single UserMemory record to memory2.

    Creates:
    - ConsolidatedMemory (scope_type="user") with the consolidated content
    - Reflection records for each unabsorbed directive

    Args:
        user_memory: The old UserMemory record
        dry_run: If True, don't actually create records
        stats: Migration statistics tracker

    Returns:
        Tuple of (consolidated_memory, list_of_reflections)
    """
    reflections = []

    # Check if this user memory already exists in memory2
    existing = ConsolidatedMemory.find_one(
        {
            "scope_type": "user",
            "agent_id": user_memory.agent_id,
            "user_id": user_memory.user_id,
        }
    )

    if existing:
        logger.debug(
            f"  User memory already exists for agent={user_memory.agent_id}, user={user_memory.user_id}, skipping"
        )
        return None, []

    # Create consolidated memory
    consolidated = ConsolidatedMemory(
        scope_type="user",
        agent_id=user_memory.agent_id,
        user_id=user_memory.user_id,
        consolidated_content=user_memory.content or "",
        word_limit=CONSOLIDATED_WORD_LIMITS.get("user", 400),
        unabsorbed_ids=[],  # Will be populated with new reflection IDs
        last_consolidated_at=user_memory.last_updated_at,
    )

    if not dry_run:
        consolidated.save()
        if stats:
            stats.user_memories_migrated += 1

    # Migrate unabsorbed directives as reflections
    unabsorbed_ids = getattr(user_memory, "unabsorbed_memory_ids", [])
    if unabsorbed_ids:
        unabsorbed_directives = load_unabsorbed_memories(unabsorbed_ids, "directive")

        for directive in unabsorbed_directives:
            reflection = Reflection(
                content=directive.content,
                scope="user",
                agent_id=user_memory.agent_id,
                user_id=user_memory.user_id,
                session_id=directive.source_session_id,  # From old model's field
                source_message_ids=directive.source_message_ids,
                formed_at=directive.createdAt or datetime.now(timezone.utc),
                absorbed=False,
            )

            if not dry_run:
                reflection.save()
                # Track the reflection ID in consolidated memory
                consolidated.unabsorbed_ids.append(reflection.id)
                if stats:
                    stats.user_reflections_created += 1

            reflections.append(reflection)

        # Update consolidated memory with reflection IDs
        if not dry_run and reflections:
            consolidated.save()

    return consolidated, reflections


def migrate_agent_memories(
    agent_id: ObjectId, dry_run: bool = False, stats: MigrationStats = None
) -> Tuple[Optional[ConsolidatedMemory], List[Reflection]]:
    """
    Migrate all AgentMemory shards for an agent to memory2.

    In the old system, each agent can have multiple "shards" of collective memory.
    In memory2, there's a single agent-level consolidated memory.
    This function merges all shard contents into one consolidated blob.

    Creates:
    - ConsolidatedMemory (scope_type="agent") with merged content from all shards
    - Reflection records for each unabsorbed suggestion across all shards

    Args:
        agent_id: The agent to migrate
        dry_run: If True, don't actually create records
        stats: Migration statistics tracker

    Returns:
        Tuple of (consolidated_memory, list_of_reflections)
    """
    reflections = []

    # Check if agent memory already exists in memory2
    existing = ConsolidatedMemory.find_one(
        {
            "scope_type": "agent",
            "agent_id": agent_id,
        }
    )

    if existing:
        logger.debug(f"  Agent memory already exists for agent={agent_id}, skipping")
        return None, []

    # Find all active shards for this agent
    shards = AgentMemory.find({"agent_id": agent_id, "is_active": True})

    if not shards:
        logger.debug(f"  No active shards found for agent={agent_id}")
        return None, []

    # Merge all shard contents into one blob
    merged_content_parts = []
    all_suggestion_ids = []
    latest_updated_at = None

    for shard in shards:
        shard_name = shard.shard_name or "Unknown Shard"

        # Add shard content with header
        if shard.content:
            merged_content_parts.append(f"## {shard_name}\n\n{shard.content}")

        # Collect unabsorbed suggestion IDs
        unabsorbed_ids = getattr(shard, "unabsorbed_memory_ids", [])
        if unabsorbed_ids:
            all_suggestion_ids.extend(unabsorbed_ids)

        # Track latest update time
        if shard.last_updated_at:
            if latest_updated_at is None or shard.last_updated_at > latest_updated_at:
                latest_updated_at = shard.last_updated_at

    # Create consolidated memory with merged content
    merged_content = "\n\n".join(merged_content_parts)
    consolidated = ConsolidatedMemory(
        scope_type="agent",
        agent_id=agent_id,
        consolidated_content=merged_content,
        word_limit=CONSOLIDATED_WORD_LIMITS.get("agent", 1000),
        unabsorbed_ids=[],  # Will be populated with new reflection IDs
        last_consolidated_at=latest_updated_at,
    )

    if not dry_run:
        consolidated.save()
        if stats:
            stats.agent_memories_migrated += 1

    # Migrate unabsorbed suggestions as reflections
    if all_suggestion_ids:
        unabsorbed_suggestions = load_unabsorbed_memories(
            all_suggestion_ids, "suggestion"
        )

        for suggestion in unabsorbed_suggestions:
            reflection = Reflection(
                content=suggestion.content,
                scope="agent",
                agent_id=agent_id,
                session_id=suggestion.source_session_id,  # From old model's field
                source_message_ids=suggestion.source_message_ids,
                formed_at=suggestion.createdAt or datetime.now(timezone.utc),
                absorbed=False,
            )

            if not dry_run:
                reflection.save()
                # Track the reflection ID in consolidated memory
                consolidated.unabsorbed_ids.append(reflection.id)
                if stats:
                    stats.agent_reflections_created += 1

            reflections.append(reflection)

        # Update consolidated memory with reflection IDs
        if not dry_run and reflections:
            consolidated.save()

    return consolidated, reflections


def migrate_agent(
    agent: Agent, dry_run: bool = False, stats: MigrationStats = None
) -> Dict:
    """
    Migrate all memory for a single agent.

    Args:
        agent: The agent to migrate
        dry_run: If True, preview only
        stats: Migration statistics tracker

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

    # 1. Migrate UserMemory records (user-level consolidated memories)
    try:
        user_memories = UserMemory.find({"agent_id": agent_id})
        for user_memory in user_memories:
            try:
                consolidated, reflections = migrate_user_memory(
                    user_memory, dry_run=dry_run, stats=stats
                )
                if consolidated:
                    results["user_memories"] += 1
                    results["user_reflections"] += len(reflections)
                    logger.debug(
                        f"  Migrated user memory for user={user_memory.user_id}: "
                        f"content={len(consolidated.consolidated_content)} chars, "
                        f"reflections={len(reflections)}"
                    )
            except Exception as e:
                error_msg = f"Error migrating user memory for user={user_memory.user_id}: {e}"
                results["errors"].append(error_msg)
                if stats:
                    stats.errors.append(error_msg)
                logger.error(f"  {error_msg}")
                traceback.print_exc()

    except Exception as e:
        error_msg = f"Error finding user memories for agent={agent_id}: {e}"
        results["errors"].append(error_msg)
        if stats:
            stats.errors.append(error_msg)
        logger.error(f"  {error_msg}")
        traceback.print_exc()

    # 2. Migrate AgentMemory shards (agent-level consolidated memory)
    try:
        consolidated, reflections = migrate_agent_memories(
            agent_id, dry_run=dry_run, stats=stats
        )
        if consolidated:
            results["agent_memories"] = 1
            results["agent_reflections"] = len(reflections)
            logger.debug(
                f"  Migrated agent memory: "
                f"content={len(consolidated.consolidated_content)} chars, "
                f"reflections={len(reflections)}"
            )

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
):
    """
    Run the memory migration.

    Args:
        dry_run: If True, preview what would be migrated without making changes
        agent_id: If provided, only migrate this specific agent
        clear_first: If True, clear memory2 collections before migrating
    """
    stats = MigrationStats()

    logger.info("=" * 60)
    logger.info("MEMORY MIGRATION: Old System → Memory2")
    logger.info("=" * 60)

    if dry_run:
        logger.info("MODE: DRY RUN (no changes will be made)")
    else:
        logger.info("MODE: LIVE (changes will be made to database)")

    if clear_first and not dry_run:
        clear_memory2_collections()

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
        logger.info(f"Found {len(agents)} agents to migrate")

    # Process each agent
    for agent in agents:
        try:
            migrate_agent(agent, dry_run=dry_run, stats=stats)
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
        description="Migrate memory from old system to memory2"
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
    )


if __name__ == "__main__":
    main()
