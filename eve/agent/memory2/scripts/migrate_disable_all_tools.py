"""
Migration Script - Disable all_tools Flag

This script migrates the users3 collection to disable the tools.all_tools flag.

It finds all documents where tools.all_tools exists and is true, and sets it to false.

Usage:
    python -m eve.agent.memory2.scripts.migrate_disable_all_tools [--dry-run]

Options:
    --dry-run     Preview changes without modifying the database
    --batch-size  Batch size for bulk operations (default: 100)
"""

import argparse
import os
import sys
import time
from typing import Dict, List

from pymongo import MongoClient, UpdateOne

# Ensure eve module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Default batch size for bulk operations
DEFAULT_BATCH_SIZE = 100


def get_users_collection():
    """Get the users3 collection from MongoDB."""
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI environment variable not set")

    client = MongoClient(mongo_uri)
    db_name = os.environ.get("MONGO_DB_NAME", "eden")
    db = client[db_name]
    return db["users3"]


def migrate_all_tools(dry_run: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, int]:
    """
    Migrate users3 collection: set tools.all_tools to false where it is currently true.

    Args:
        dry_run: If True, preview changes without modifying database
        batch_size: Number of operations to batch together

    Returns:
        Dict with migration stats
    """
    collection = get_users_collection()
    start_time = time.time()

    stats = {
        "total_users": 0,
        "with_all_tools_true": 0,
        "migrated": 0,
        "already_false": 0,
        "no_tools_field": 0,
        "errors": 0,
    }

    # Get total count for context
    stats["total_users"] = collection.count_documents({})
    print(f"  Total users in collection: {stats['total_users']}")

    # Count documents where tools.all_tools is true
    stats["with_all_tools_true"] = collection.count_documents({"tools.all_tools": True})
    print(f"  Users with tools.all_tools=true: {stats['with_all_tools_true']}")

    # Count documents where tools.all_tools is already false
    stats["already_false"] = collection.count_documents({"tools.all_tools": False})
    print(f"  Users with tools.all_tools=false: {stats['already_false']}")

    # Count documents without tools field or without all_tools
    stats["no_tools_field"] = collection.count_documents({
        "$or": [
            {"tools": {"$exists": False}},
            {"tools.all_tools": {"$exists": False}}
        ]
    })
    print(f"  Users without tools.all_tools field: {stats['no_tools_field']}")

    if stats["with_all_tools_true"] == 0:
        print("  No documents need migration")
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s")
        return stats

    if dry_run:
        # In dry run, show sample of affected documents
        print(f"\n  [DRY RUN] Would update {stats['with_all_tools_true']} documents")
        print("\n  Sample of affected documents:")

        cursor = collection.find(
            {"tools.all_tools": True},
            {"_id": 1, "userId": 1, "username": 1, "tools.all_tools": 1}
        ).limit(10)

        for doc in cursor:
            user_id = doc.get("userId", doc.get("username", str(doc["_id"])))
            print(f"    - {user_id}: tools.all_tools={doc.get('tools', {}).get('all_tools')}")

        if stats["with_all_tools_true"] > 10:
            print(f"    ... and {stats['with_all_tools_true'] - 10} more")

        stats["migrated"] = stats["with_all_tools_true"]
    else:
        # Perform the actual migration using update_many for efficiency
        print(f"\n  Updating {stats['with_all_tools_true']} documents...")

        try:
            result = collection.update_many(
                {"tools.all_tools": True},
                {"$set": {"tools.all_tools": False}}
            )
            stats["migrated"] = result.modified_count
            print(f"  Successfully updated {result.modified_count} documents")
        except Exception as e:
            print(f"  ERROR in bulk update: {e}")
            stats["errors"] += stats["with_all_tools_true"]

            # Fallback to batch processing
            print("  Falling back to batch processing...")
            cursor = collection.find(
                {"tools.all_tools": True},
                {"_id": 1}
            )

            batch_operations: List[UpdateOne] = []
            processed = 0

            for doc in cursor:
                batch_operations.append(
                    UpdateOne(
                        {"_id": doc["_id"]},
                        {"$set": {"tools.all_tools": False}}
                    )
                )

                if len(batch_operations) >= batch_size:
                    try:
                        result = collection.bulk_write(batch_operations, ordered=False)
                        stats["migrated"] += result.modified_count
                        stats["errors"] -= result.modified_count
                        processed += len(batch_operations)
                        print(f"    Batch processed: {processed} documents")
                    except Exception as batch_e:
                        print(f"    ERROR in batch: {batch_e}")
                    batch_operations = []

            # Process remaining
            if batch_operations:
                try:
                    result = collection.bulk_write(batch_operations, ordered=False)
                    stats["migrated"] += result.modified_count
                    stats["errors"] -= result.modified_count
                except Exception as batch_e:
                    print(f"    ERROR in final batch: {batch_e}")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    return stats


def verify_migration() -> Dict[str, int]:
    """
    Verify the migration was successful.

    Returns:
        Dict with verification stats
    """
    collection = get_users_collection()

    stats = {
        "still_true": 0,
        "now_false": 0,
    }

    stats["still_true"] = collection.count_documents({"tools.all_tools": True})
    stats["now_false"] = collection.count_documents({"tools.all_tools": False})

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Disable tools.all_tools flag in users3 collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for bulk operations (default: {DEFAULT_BATCH_SIZE})"
    )
    args = parser.parse_args()

    dry_run = args.dry_run
    batch_size = args.batch_size
    overall_start = time.time()

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 60)

    print(f"\nConnecting to database...")
    print(f"DB: {os.environ.get('MONGO_DB_NAME', 'eden')}")
    print(f"Batch size: {batch_size}")

    # Run migration
    print("\n" + "=" * 60)
    print("MIGRATING users3 (tools.all_tools: true -> false)")
    print("=" * 60)

    migration_stats = migrate_all_tools(dry_run, batch_size)

    # Print migration stats
    print(f"\nMigration stats:")
    print(f"  Total users:              {migration_stats['total_users']}")
    print(f"  Had all_tools=true:       {migration_stats['with_all_tools_true']}")
    print(f"  Migrated:                 {migration_stats['migrated']}")
    print(f"  Already false:            {migration_stats['already_false']}")
    print(f"  No tools.all_tools field: {migration_stats['no_tools_field']}")
    print(f"  Errors:                   {migration_stats['errors']}")

    # Verify if not dry run
    if not dry_run and migration_stats['migrated'] > 0:
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        verify_stats = verify_migration()
        print(f"  Documents with tools.all_tools=true:  {verify_stats['still_true']}")
        print(f"  Documents with tools.all_tools=false: {verify_stats['now_false']}")

        if verify_stats['still_true'] > 0:
            print(f"\n  WARNING: {verify_stats['still_true']} documents still have all_tools=true")

    # Summary
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)

    if dry_run:
        print(f"[DRY RUN] Would migrate {migration_stats['migrated']} documents")
    else:
        print(f"Successfully migrated {migration_stats['migrated']} documents")
        if migration_stats['errors'] > 0:
            print(f"WARNING: {migration_stats['errors']} documents had errors")

    print(f"\nTotal time: {overall_elapsed:.1f}s")
    print("Migration complete!")

    return 0 if migration_stats['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
