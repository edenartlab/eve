"""
Memory System v2 - Scope Field Migration Script

This script migrates the scope fields across all memory collections to use
a consistent `scope: string` format:

1. memory2_facts: Convert `scope: array` to `scope: string`
   - If array has 1 element: use that element
   - If array has >1 elements: warn, print ID, skip document

2. memory2_consolidated: Rename `scope_type` to `scope`

3. memory2_reflections: Already uses `scope: string` (no changes needed)

Usage:
    python -m eve.agent.memory2.scripts.migrate_scope_fields [--dry-run]

Options:
    --dry-run   Preview changes without modifying the database
    --batch-size  Batch size for bulk operations (default: 500)
"""

import argparse
import os
import sys
import time
from typing import Dict, List

from pymongo import UpdateOne

# Ensure eve module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from eve.agent.memory2.models import Fact, ConsolidatedMemory, Reflection

# Default batch size for bulk operations
DEFAULT_BATCH_SIZE = 100


def migrate_facts(dry_run: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, int]:
    """
    Migrate memory2_facts collection: convert scope array to string.

    Uses batch updates for better performance.

    Args:
        dry_run: If True, preview changes without modifying database
        batch_size: Number of operations to batch together

    Returns:
        Dict with migration stats
    """
    collection = Fact.get_collection()

    stats = {
        "total": 0,
        "migrated": 0,
        "skipped_multi_scope": 0,
        "already_string": 0,
        "errors": 0,
    }

    # First, get a quick count for progress tracking
    total_count = collection.count_documents({})
    print(f"  Scanning {total_count} documents...")

    # Find all documents where scope is an array (use $type filter for efficiency)
    # Type 4 = array, Type 2 = string
    array_cursor = collection.find(
        {"scope": {"$type": "array"}},
        {"_id": 1, "scope": 1}  # Only fetch needed fields
    )

    skipped_ids: List[str] = []
    batch_operations: List[UpdateOne] = []
    scanned = 0
    start_time = time.time()

    for doc in array_cursor:
        scanned += 1
        stats["total"] += 1
        doc_id = doc["_id"]
        scope = doc.get("scope", [])

        # Progress update every 1000 documents
        if scanned % 1000 == 0:
            elapsed = time.time() - start_time
            rate = scanned / elapsed if elapsed > 0 else 0
            print(f"  Progress: {scanned} array-scope docs scanned, {len(batch_operations)} queued for update ({rate:.0f}/sec)")

        if len(scope) == 0:
            print(f"WARNING: Document {doc_id} has empty scope array - skipping")
            stats["errors"] += 1
            skipped_ids.append(str(doc_id))
            continue

        if len(scope) > 1:
            print(f"WARNING: Document {doc_id} has multiple scopes {scope} - skipping")
            stats["skipped_multi_scope"] += 1
            skipped_ids.append(str(doc_id))
            continue

        # Single element array - queue update
        new_scope = scope[0]
        batch_operations.append(
            UpdateOne({"_id": doc_id}, {"$set": {"scope": new_scope}})
        )

        # Execute batch when it reaches the batch size
        if len(batch_operations) >= batch_size and not dry_run:
            try:
                result = collection.bulk_write(batch_operations, ordered=False)
                stats["migrated"] += result.modified_count
                print(f"  Batch committed: {result.modified_count} documents updated")
            except Exception as e:
                print(f"ERROR in batch update: {e}")
                stats["errors"] += len(batch_operations)
            batch_operations = []

    # Execute remaining operations
    if batch_operations:
        if dry_run:
            stats["migrated"] += len(batch_operations)
        else:
            try:
                result = collection.bulk_write(batch_operations, ordered=False)
                stats["migrated"] += result.modified_count
                print(f"  Final batch committed: {result.modified_count} documents updated")
            except Exception as e:
                print(f"ERROR in final batch update: {e}")
                stats["errors"] += len(batch_operations)

    # Count documents that already have string scope
    stats["already_string"] = collection.count_documents({"scope": {"$type": "string"}})

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    if skipped_ids:
        print(f"\nSkipped document IDs (multi-scope or errors):")
        for id_str in skipped_ids[:20]:  # Limit output
            print(f"  - {id_str}")
        if len(skipped_ids) > 20:
            print(f"  ... and {len(skipped_ids) - 20} more")

    return stats


def migrate_consolidated(dry_run: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, int]:
    """
    Migrate memory2_consolidated collection: rename scope_type to scope.

    Uses MongoDB's $rename operator with update_many for maximum efficiency.

    Args:
        dry_run: If True, preview changes without modifying database
        batch_size: Number of operations to batch together (for edge cases)

    Returns:
        Dict with migration stats
    """
    collection = ConsolidatedMemory.get_collection()
    start_time = time.time()

    stats = {
        "total": 0,
        "migrated": 0,
        "already_migrated": 0,
        "has_both_fields": 0,
        "errors": 0,
    }

    # Count documents that need migration (have scope_type but not scope)
    needs_migration_count = collection.count_documents({
        "scope_type": {"$exists": True},
        "scope": {"$exists": False}
    })
    stats["total"] = needs_migration_count
    print(f"  Found {needs_migration_count} documents needing migration")

    # Check for edge case: documents with both fields
    has_both_count = collection.count_documents({
        "scope_type": {"$exists": True},
        "scope": {"$exists": True}
    })
    if has_both_count > 0:
        print(f"  WARNING: {has_both_count} documents have both scope and scope_type fields")
        stats["has_both_fields"] = has_both_count

    # Count already migrated (have scope but not scope_type)
    stats["already_migrated"] = collection.count_documents({
        "scope": {"$exists": True},
        "scope_type": {"$exists": False}
    })
    print(f"  Already migrated: {stats['already_migrated']} documents")

    if needs_migration_count == 0:
        print("  No documents need migration")
        return stats

    if dry_run:
        stats["migrated"] = needs_migration_count
        print(f"  [DRY RUN] Would migrate {needs_migration_count} documents")
    else:
        # Use $rename operator with update_many - very efficient single operation
        print(f"  Renaming scope_type -> scope for {needs_migration_count} documents...")
        try:
            result = collection.update_many(
                {
                    "scope_type": {"$exists": True},
                    "scope": {"$exists": False}
                },
                {"$rename": {"scope_type": "scope"}}
            )
            stats["migrated"] = result.modified_count
            print(f"  Successfully migrated {result.modified_count} documents")
        except Exception as e:
            print(f"ERROR in bulk rename: {e}")
            stats["errors"] += needs_migration_count

            # Fallback: batch processing for edge cases
            print("  Falling back to batch processing...")
            cursor = collection.find(
                {"scope_type": {"$exists": True}, "scope": {"$exists": False}},
                {"_id": 1, "scope_type": 1}
            )
            batch_operations: List[UpdateOne] = []

            for doc in cursor:
                batch_operations.append(
                    UpdateOne(
                        {"_id": doc["_id"]},
                        {"$rename": {"scope_type": "scope"}}
                    )
                )
                if len(batch_operations) >= batch_size:
                    try:
                        result = collection.bulk_write(batch_operations, ordered=False)
                        stats["migrated"] += result.modified_count
                        stats["errors"] -= result.modified_count
                    except Exception as batch_e:
                        print(f"ERROR in fallback batch: {batch_e}")
                    batch_operations = []

            if batch_operations:
                try:
                    result = collection.bulk_write(batch_operations, ordered=False)
                    stats["migrated"] += result.modified_count
                    stats["errors"] -= result.modified_count
                except Exception as batch_e:
                    print(f"ERROR in final fallback batch: {batch_e}")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    return stats


def verify_reflections() -> Dict[str, int]:
    """
    Verify memory2_reflections collection uses scope as string.

    Uses count queries for efficiency where possible.

    Returns:
        Dict with verification stats
    """
    collection = Reflection.get_collection()
    start_time = time.time()

    stats = {
        "total": 0,
        "valid_string": 0,
        "invalid": 0,
    }

    # Use count queries for efficiency
    stats["total"] = collection.count_documents({})
    print(f"  Total reflections: {stats['total']}")

    stats["valid_string"] = collection.count_documents({"scope": {"$type": "string"}})
    print(f"  Valid (string scope): {stats['valid_string']}")

    # Only iterate through invalid documents to report them
    invalid_cursor = collection.find(
        {"scope": {"$not": {"$type": "string"}}},
        {"_id": 1, "scope": 1}
    )

    invalid_docs = list(invalid_cursor)
    stats["invalid"] = len(invalid_docs)

    if invalid_docs:
        print(f"\n  Invalid reflections found:")
        for doc in invalid_docs[:20]:  # Limit output
            print(f"    - {doc['_id']}: scope={doc.get('scope')} (type={type(doc.get('scope')).__name__})")
        if len(invalid_docs) > 20:
            print(f"    ... and {len(invalid_docs) - 20} more")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    return stats


def update_indexes(dry_run: bool = False):
    """Update indexes after migration."""
    if dry_run:
        print("\n[DRY RUN] Would update indexes...")
        return

    print("\nUpdating indexes...")

    # Drop old index on memory2_consolidated if exists
    consolidated = ConsolidatedMemory.get_collection()
    try:
        # Try to drop old scope_type index
        consolidated.drop_index("consolidated_scope_idx")
        print("  Dropped old consolidated_scope_idx")
    except Exception as e:
        print(f"  Note: Could not drop old index (may not exist): {e}")

    # Create new index with scope field
    try:
        consolidated.create_index(
            [("agent_id", 1), ("user_id", 1), ("session_id", 1), ("scope", 1)],
            name="consolidated_scope_idx",
            unique=True,
            background=True,
        )
        print("  Created new consolidated_scope_idx with scope field")
    except Exception as e:
        print(f"  WARNING: Could not create index: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate memory system scope fields to consistent format"
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

    print(f"\nConnecting to database: {os.environ.get('DB', 'STAGE')}...")
    print(f"Batch size: {batch_size}")

    # Migrate facts
    print("\n" + "=" * 60)
    print("MIGRATING memory2_facts (scope array -> string)")
    print("=" * 60)
    facts_stats = migrate_facts(dry_run, batch_size)
    print(f"\nFacts migration stats:")
    print(f"  Total with array scope: {facts_stats['total']}")
    print(f"  Already string:         {facts_stats['already_string']}")
    print(f"  Migrated:               {facts_stats['migrated']}")
    print(f"  Skipped (multi-scope):  {facts_stats['skipped_multi_scope']}")
    print(f"  Errors:                 {facts_stats['errors']}")

    # Migrate consolidated
    print("\n" + "=" * 60)
    print("MIGRATING memory2_consolidated (scope_type -> scope)")
    print("=" * 60)
    consolidated_stats = migrate_consolidated(dry_run, batch_size)
    print(f"\nConsolidated migration stats:")
    print(f"  With scope_type:     {consolidated_stats['total']}")
    print(f"  Migrated:            {consolidated_stats['migrated']}")
    print(f"  Already migrated:    {consolidated_stats['already_migrated']}")
    print(f"  Has both fields:     {consolidated_stats['has_both_fields']}")
    print(f"  Errors:              {consolidated_stats['errors']}")

    # Verify reflections
    print("\n" + "=" * 60)
    print("VERIFYING memory2_reflections (should already use scope: string)")
    print("=" * 60)
    reflections_stats = verify_reflections()
    print(f"\nReflections verification stats:")
    print(f"  Total documents:     {reflections_stats['total']}")
    print(f"  Valid (string):      {reflections_stats['valid_string']}")
    print(f"  Invalid:             {reflections_stats['invalid']}")

    # Update indexes
    update_indexes(dry_run)

    # Summary
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)

    total_migrated = facts_stats['migrated'] + consolidated_stats['migrated']
    total_errors = facts_stats['errors'] + consolidated_stats['errors'] + facts_stats['skipped_multi_scope']

    if dry_run:
        print(f"[DRY RUN] Would migrate {total_migrated} documents")
        print(f"[DRY RUN] Would skip {total_errors} documents with issues")
    else:
        print(f"Successfully migrated {total_migrated} documents")
        if total_errors > 0:
            print(f"WARNING: {total_errors} documents had issues (see above)")

    if reflections_stats['invalid'] > 0:
        print(f"WARNING: {reflections_stats['invalid']} reflections have invalid scope format")

    print(f"\nTotal time: {overall_elapsed:.1f}s")
    print("Migration complete!")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
