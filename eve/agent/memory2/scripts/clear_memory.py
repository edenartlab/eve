"""
Memory Clearing Script for Memory2 System

This script allows selective clearing of an agent's memories by type and scope.

Memory Types:
- Facts: Atomic factual memories stored for RAG retrieval
  - Scopes: user, agent (no session scope for facts)
- Reflections: Interpreted memories that evolve agent behavior
  - Scopes: session, user, agent
- Consolidated: Merged reflection blobs
  - Scope types: session, user, agent

Usage:
    python -m eve.agent.memory2.scripts.clear_memory --agent-id <agent_id> [options]

IMPORTANT: By default, this script runs in dry-run mode (preview only).
           You must pass --execute to actually delete records.

Examples:
    # Preview what would be deleted (default behavior)
    python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --all

    # Actually delete user-scoped reflections
    python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --reflections --scopes user --execute

    # Actually delete all facts (both user and agent scoped)
    python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --facts --scopes user agent --execute

    # Actually delete everything for an agent
    python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --all --execute

    # Preview clearing only consolidated memories for agent scope
    python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --consolidated --scopes agent
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Set

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.models import ConsolidatedMemory, Fact, Reflection


@dataclass
class ClearStats:
    """Track clearing statistics."""

    # Facts stats
    facts_user_scope: int = 0
    facts_agent_scope: int = 0

    # Reflections stats
    reflections_session_scope: int = 0
    reflections_user_scope: int = 0
    reflections_agent_scope: int = 0

    # Consolidated stats
    consolidated_session_scope: int = 0
    consolidated_user_scope: int = 0
    consolidated_agent_scope: int = 0

    # Errors
    errors: List[str] = field(default_factory=list)

    @property
    def total_facts(self) -> int:
        return self.facts_user_scope + self.facts_agent_scope

    @property
    def total_reflections(self) -> int:
        return (
            self.reflections_session_scope
            + self.reflections_user_scope
            + self.reflections_agent_scope
        )

    @property
    def total_consolidated(self) -> int:
        return (
            self.consolidated_session_scope
            + self.consolidated_user_scope
            + self.consolidated_agent_scope
        )

    @property
    def total(self) -> int:
        return self.total_facts + self.total_reflections + self.total_consolidated

    def print_summary(self, action: str = "deleted"):
        """Print a summary of the clearing operation."""
        logger.info("\n" + "=" * 60)
        logger.info(f"MEMORY CLEARING SUMMARY ({action.upper()})")
        logger.info("=" * 60)

        if self.total_facts > 0 or self.facts_user_scope == 0 and self.facts_agent_scope == 0:
            logger.info("\nFacts:")
            logger.info(f"  User-scoped:   {self.facts_user_scope}")
            logger.info(f"  Agent-scoped:  {self.facts_agent_scope}")
            logger.info(f"  Total:         {self.total_facts}")

        if self.total_reflections > 0 or (
            self.reflections_session_scope == 0
            and self.reflections_user_scope == 0
            and self.reflections_agent_scope == 0
        ):
            logger.info("\nReflections:")
            logger.info(f"  Session-scoped: {self.reflections_session_scope}")
            logger.info(f"  User-scoped:    {self.reflections_user_scope}")
            logger.info(f"  Agent-scoped:   {self.reflections_agent_scope}")
            logger.info(f"  Total:          {self.total_reflections}")

        if self.total_consolidated > 0 or (
            self.consolidated_session_scope == 0
            and self.consolidated_user_scope == 0
            and self.consolidated_agent_scope == 0
        ):
            logger.info("\nConsolidated Memories:")
            logger.info(f"  Session-scoped: {self.consolidated_session_scope}")
            logger.info(f"  User-scoped:    {self.consolidated_user_scope}")
            logger.info(f"  Agent-scoped:   {self.consolidated_agent_scope}")
            logger.info(f"  Total:          {self.total_consolidated}")

        logger.info("\n" + "-" * 60)
        logger.info(f"GRAND TOTAL: {self.total} records {action}")
        logger.info("=" * 60)

        if self.errors:
            logger.error(f"\nErrors encountered: {len(self.errors)}")
            for error in self.errors[:10]:
                logger.error(f"  - {error}")
            if len(self.errors) > 10:
                logger.error(f"  ... and {len(self.errors) - 10} more errors")


def count_facts_by_scope(
    agent_id: ObjectId,
    scopes: Set[str],
) -> Dict[str, int]:
    """
    Count facts by scope for an agent.

    Note: Facts can have multiple scopes (e.g., both "user" and "agent"),
    so we count facts that have each scope in their scope list.

    Args:
        agent_id: Agent ID
        scopes: Set of scopes to count ("user", "agent")

    Returns:
        Dict mapping scope -> count
    """
    collection = Fact.get_collection()
    counts = {}

    for scope in scopes:
        if scope == "session":
            # Facts don't have session scope
            counts[scope] = 0
            continue

        # Count facts where this scope is in the scope list
        count = collection.count_documents({
            "agent_id": agent_id,
            "scope": scope,  # MongoDB matches if scope is in the array
        })
        counts[scope] = count

    return counts


def count_reflections_by_scope(
    agent_id: ObjectId,
    scopes: Set[str],
) -> Dict[str, int]:
    """
    Count reflections by scope for an agent.

    Args:
        agent_id: Agent ID
        scopes: Set of scopes to count ("session", "user", "agent")

    Returns:
        Dict mapping scope -> count
    """
    collection = Reflection.get_collection()
    counts = {}

    for scope in scopes:
        count = collection.count_documents({
            "agent_id": agent_id,
            "scope": scope,
        })
        counts[scope] = count

    return counts


def count_consolidated_by_scope(
    agent_id: ObjectId,
    scopes: Set[str],
) -> Dict[str, int]:
    """
    Count consolidated memories by scope type for an agent.

    Args:
        agent_id: Agent ID
        scopes: Set of scope types to count ("session", "user", "agent")

    Returns:
        Dict mapping scope_type -> count
    """
    collection = ConsolidatedMemory.get_collection()
    counts = {}

    for scope in scopes:
        count = collection.count_documents({
            "agent_id": agent_id,
            "scope_type": scope,
        })
        counts[scope] = count

    return counts


def delete_facts_by_scope(
    agent_id: ObjectId,
    scopes: Set[str],
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Delete facts by scope for an agent.

    Note: Since facts can have multiple scopes, we need to handle this carefully:
    - If a fact has ONLY the specified scope, delete it
    - If a fact has multiple scopes and we're only clearing one, we could
      either delete the whole fact or update it to remove that scope.
    - For simplicity, we delete facts that contain the specified scope.

    Args:
        agent_id: Agent ID
        scopes: Set of scopes to clear ("user", "agent")
        dry_run: If True, only count without deleting

    Returns:
        Dict mapping scope -> deleted count
    """
    collection = Fact.get_collection()
    deleted = {}

    for scope in scopes:
        if scope == "session":
            # Facts don't have session scope
            deleted[scope] = 0
            continue

        query = {
            "agent_id": agent_id,
            "scope": scope,
        }

        if dry_run:
            deleted[scope] = collection.count_documents(query)
        else:
            result = collection.delete_many(query)
            deleted[scope] = result.deleted_count

    return deleted


def delete_reflections_by_scope(
    agent_id: ObjectId,
    scopes: Set[str],
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Delete reflections by scope for an agent.

    Args:
        agent_id: Agent ID
        scopes: Set of scopes to clear ("session", "user", "agent")
        dry_run: If True, only count without deleting

    Returns:
        Dict mapping scope -> deleted count
    """
    collection = Reflection.get_collection()
    deleted = {}

    for scope in scopes:
        query = {
            "agent_id": agent_id,
            "scope": scope,
        }

        if dry_run:
            deleted[scope] = collection.count_documents(query)
        else:
            result = collection.delete_many(query)
            deleted[scope] = result.deleted_count

    return deleted


def delete_consolidated_by_scope(
    agent_id: ObjectId,
    scopes: Set[str],
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Delete consolidated memories by scope type for an agent.

    Args:
        agent_id: Agent ID
        scopes: Set of scope types to clear ("session", "user", "agent")
        dry_run: If True, only count without deleting

    Returns:
        Dict mapping scope_type -> deleted count
    """
    collection = ConsolidatedMemory.get_collection()
    deleted = {}

    for scope in scopes:
        query = {
            "agent_id": agent_id,
            "scope_type": scope,
        }

        if dry_run:
            deleted[scope] = collection.count_documents(query)
        else:
            result = collection.delete_many(query)
            deleted[scope] = result.deleted_count

    return deleted


def clear_agent_memory(
    agent_id: ObjectId,
    clear_facts: bool = False,
    clear_reflections: bool = False,
    clear_consolidated: bool = False,
    scopes: Optional[Set[str]] = None,
    dry_run: bool = False,
) -> ClearStats:
    """
    Clear specified memories for an agent.

    Args:
        agent_id: The agent ID to clear memories for
        clear_facts: Whether to clear facts
        clear_reflections: Whether to clear reflections
        clear_consolidated: Whether to clear consolidated memories
        scopes: Set of scopes to clear. For facts: "user", "agent".
                For reflections/consolidated: "session", "user", "agent".
                If None, clears all applicable scopes.
        dry_run: If True, only count without actually deleting

    Returns:
        ClearStats with counts of deleted (or would-be-deleted) records
    """
    stats = ClearStats()

    # Default to all scopes if none specified
    if scopes is None:
        scopes = {"session", "user", "agent"}

    # Validate scopes
    valid_scopes = {"session", "user", "agent"}
    invalid_scopes = scopes - valid_scopes
    if invalid_scopes:
        raise ValueError(f"Invalid scopes: {invalid_scopes}. Valid scopes: {valid_scopes}")

    action = "would be deleted" if dry_run else "deleted"
    logger.info(f"\n{'DRY RUN: ' if dry_run else ''}Clearing memories for agent {agent_id}")
    logger.info(f"Scopes: {', '.join(sorted(scopes))}")
    logger.info(f"Types: facts={clear_facts}, reflections={clear_reflections}, consolidated={clear_consolidated}")

    # Clear facts
    if clear_facts:
        fact_scopes = scopes - {"session"}  # Facts don't have session scope
        if fact_scopes:
            try:
                deleted = delete_facts_by_scope(agent_id, fact_scopes, dry_run)
                stats.facts_user_scope = deleted.get("user", 0)
                stats.facts_agent_scope = deleted.get("agent", 0)
                logger.info(f"Facts {action}: user={stats.facts_user_scope}, agent={stats.facts_agent_scope}")
            except Exception as e:
                error_msg = f"Error clearing facts: {e}"
                stats.errors.append(error_msg)
                logger.error(error_msg)

    # Clear reflections
    if clear_reflections:
        try:
            deleted = delete_reflections_by_scope(agent_id, scopes, dry_run)
            stats.reflections_session_scope = deleted.get("session", 0)
            stats.reflections_user_scope = deleted.get("user", 0)
            stats.reflections_agent_scope = deleted.get("agent", 0)
            logger.info(
                f"Reflections {action}: session={stats.reflections_session_scope}, "
                f"user={stats.reflections_user_scope}, agent={stats.reflections_agent_scope}"
            )
        except Exception as e:
            error_msg = f"Error clearing reflections: {e}"
            stats.errors.append(error_msg)
            logger.error(error_msg)

    # Clear consolidated memories
    if clear_consolidated:
        try:
            deleted = delete_consolidated_by_scope(agent_id, scopes, dry_run)
            stats.consolidated_session_scope = deleted.get("session", 0)
            stats.consolidated_user_scope = deleted.get("user", 0)
            stats.consolidated_agent_scope = deleted.get("agent", 0)
            logger.info(
                f"Consolidated {action}: session={stats.consolidated_session_scope}, "
                f"user={stats.consolidated_user_scope}, agent={stats.consolidated_agent_scope}"
            )
        except Exception as e:
            error_msg = f"Error clearing consolidated memories: {e}"
            stats.errors.append(error_msg)
            logger.error(error_msg)

    return stats


def show_current_memory_counts(agent_id: ObjectId):
    """Display current memory counts for an agent."""
    logger.info("\n" + "=" * 60)
    logger.info("CURRENT MEMORY COUNTS")
    logger.info("=" * 60)

    # Facts
    fact_counts = count_facts_by_scope(agent_id, {"user", "agent"})
    logger.info("\nFacts:")
    logger.info(f"  User-scoped:  {fact_counts.get('user', 0)}")
    logger.info(f"  Agent-scoped: {fact_counts.get('agent', 0)}")

    # Reflections
    ref_counts = count_reflections_by_scope(agent_id, {"session", "user", "agent"})
    logger.info("\nReflections:")
    logger.info(f"  Session-scoped: {ref_counts.get('session', 0)}")
    logger.info(f"  User-scoped:    {ref_counts.get('user', 0)}")
    logger.info(f"  Agent-scoped:   {ref_counts.get('agent', 0)}")

    # Consolidated
    cons_counts = count_consolidated_by_scope(agent_id, {"session", "user", "agent"})
    logger.info("\nConsolidated Memories:")
    logger.info(f"  Session-scoped: {cons_counts.get('session', 0)}")
    logger.info(f"  User-scoped:    {cons_counts.get('user', 0)}")
    logger.info(f"  Agent-scoped:   {cons_counts.get('agent', 0)}")

    logger.info("=" * 60)


def run_clear(
    agent_id: str,
    clear_facts: bool = False,
    clear_reflections: bool = False,
    clear_consolidated: bool = False,
    scopes: Optional[List[str]] = None,
    dry_run: bool = False,
    show_counts: bool = False,
    skip_confirmation: bool = False,
):
    """
    Run the memory clearing operation.

    Args:
        agent_id: Agent ID string
        clear_facts: Whether to clear facts
        clear_reflections: Whether to clear reflections
        clear_consolidated: Whether to clear consolidated memories
        scopes: List of scopes to clear
        dry_run: If True, preview only
        show_counts: If True, show current memory counts before clearing
        skip_confirmation: If True, skip confirmation prompt
    """
    try:
        agent_oid = ObjectId(agent_id)
    except Exception as e:
        logger.error(f"Invalid agent_id: {agent_id}. Error: {e}")
        return

    # Show current counts if requested
    if show_counts or dry_run:
        show_current_memory_counts(agent_oid)

    # Convert scopes list to set
    scopes_set = set(scopes) if scopes else {"session", "user", "agent"}

    # Determine what will be cleared
    clear_all = not (clear_facts or clear_reflections or clear_consolidated)
    if clear_all:
        clear_facts = clear_reflections = clear_consolidated = True

    logger.info("\n" + "=" * 60)
    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    else:
        logger.info("LIVE MODE - Records will be deleted")
    logger.info("=" * 60)

    # Run clearing operation
    stats = clear_agent_memory(
        agent_id=agent_oid,
        clear_facts=clear_facts,
        clear_reflections=clear_reflections,
        clear_consolidated=clear_consolidated,
        scopes=scopes_set,
        dry_run=dry_run,
    )

    # Print summary
    action = "would be deleted" if dry_run else "deleted"
    stats.print_summary(action)

    # If dry run and there are records to delete, show hint about --execute
    if dry_run and stats.total > 0:
        logger.info("\nThis was a DRY RUN. To actually delete these records, add --execute to the command.")

    # Show final counts after deletion (if not dry run)
    if not dry_run and stats.total > 0:
        logger.info("\nFinal memory counts after deletion:")
        show_current_memory_counts(agent_oid)


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Clear agent memories from the memory2 system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be deleted (default behavior - dry run)
  python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --all

  # Actually delete user-scoped reflections
  python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --reflections --scopes user --execute

  # Actually delete all facts (both user and agent scoped)
  python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --facts --scopes user agent --execute

  # Actually delete everything for an agent
  python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --all --execute

  # Preview clearing only consolidated memories for agent scope
  python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --consolidated --scopes agent

  # Just show current memory counts without deleting
  python -m eve.agent.memory2.scripts.clear_memory --agent-id 123abc --show-counts
        """,
    )

    parser.add_argument(
        "--agent-id",
        type=str,
        required=True,
        help="Agent ID to clear memories for (required)",
    )

    # Memory types to clear
    type_group = parser.add_argument_group("Memory Types")
    type_group.add_argument(
        "--facts",
        action="store_true",
        help="Clear facts (scopes: user, agent - no session scope for facts)",
    )
    type_group.add_argument(
        "--reflections",
        action="store_true",
        help="Clear reflections (scopes: session, user, agent)",
    )
    type_group.add_argument(
        "--consolidated",
        action="store_true",
        help="Clear consolidated memories (scopes: session, user, agent)",
    )
    type_group.add_argument(
        "--all",
        action="store_true",
        dest="all_types",
        help="Clear all memory types (facts, reflections, consolidated)",
    )

    # Scopes to clear
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=["session", "user", "agent"],
        help="Scopes to clear. If not specified, clears all scopes. "
             "Note: facts don't have session scope.",
    )

    # Options
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the deletion. Without this flag, runs in dry-run mode (preview only).",
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Show current memory counts (useful for verification)",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        dest="skip_confirmation",
        help="Skip confirmation prompts (only applies when --execute is used)",
    )

    args = parser.parse_args()

    # Determine what to clear
    clear_facts = args.facts or args.all_types
    clear_reflections = args.reflections or args.all_types
    clear_consolidated = args.consolidated or args.all_types

    # If no types specified and not just showing counts, require explicit choice
    if not (clear_facts or clear_reflections or clear_consolidated) and not args.show_counts:
        parser.error(
            "Please specify what to clear: --facts, --reflections, --consolidated, "
            "or --all. Use --show-counts to just view current counts."
        )

    # Determine if this is a dry run (default) or actual execution
    dry_run = not args.execute

    # Confirmation for live deletion
    if args.execute and not args.skip_confirmation and not args.show_counts:
        types_to_clear = []
        if clear_facts:
            types_to_clear.append("facts")
        if clear_reflections:
            types_to_clear.append("reflections")
        if clear_consolidated:
            types_to_clear.append("consolidated")

        scopes_str = ", ".join(args.scopes) if args.scopes else "all"

        response = input(
            f"\nWARNING: This will permanently delete {', '.join(types_to_clear)} "
            f"with scopes [{scopes_str}] for agent {args.agent_id}.\n"
            "This action cannot be undone. Are you sure? (yes/no): "
        )
        if response.lower() != "yes":
            logger.info("Operation cancelled.")
            return

    # Handle show-counts only mode
    if args.show_counts and not (clear_facts or clear_reflections or clear_consolidated):
        try:
            agent_oid = ObjectId(args.agent_id)
            show_current_memory_counts(agent_oid)
        except Exception as e:
            logger.error(f"Invalid agent_id: {args.agent_id}. Error: {e}")
        return

    run_clear(
        agent_id=args.agent_id,
        clear_facts=clear_facts,
        clear_reflections=clear_reflections,
        clear_consolidated=clear_consolidated,
        scopes=args.scopes,
        dry_run=dry_run,
        show_counts=args.show_counts,
        skip_confirmation=args.skip_confirmation,
    )


if __name__ == "__main__":
    main()
