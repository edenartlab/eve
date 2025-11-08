#!/usr/bin/env python3
"""
Migration Script: Remove 'freeTools' from featureFlags array in users3 collection

This script safely removes the 'freeTools' string from the 'featureFlags' array field
in users3 documents where it exists. The 'freeTools' feature flag was accidentally
added to some agent/user documents and needs to be removed.

Usage:
    # Dry run against staging, agents only (default)
    python scripts/remove_freetools_preview.py

    # Dry run against production database, agents only
    python scripts/remove_freetools_preview.py --database prod

    # Dry run against production, filter by users
    python scripts/remove_freetools_preview.py --database prod --type user

    # Dry run with verbose output
    python scripts/remove_freetools_preview.py --database prod --verbose

    # Actually perform the migration on production agents (requires explicit confirmation)
    python scripts/remove_freetools_preview.py --database prod --execute

Safety Features:
    - Dry run mode by default (requires --execute flag to make changes)
    - Database selection flag (defaults to staging for safety)
    - Type filter (defaults to agents only)
    - Verbose logging with document IDs and usernames
    - Backup recommendations before execution
    - Uses atomic MongoDB operations ($pull)
    - Only modifies documents that actually contain 'freeTools' in featureFlags array
"""

import argparse
import os
import sys
from typing import List, Dict, Any
from loguru import logger

# Add parent directory to path to import eve modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def find_documents_with_freetools(collection, doc_type: str = "agent") -> List[Dict[str, Any]]:
    """
    Find all documents in users3 that have 'freeTools' in their featureFlags array.

    Args:
        collection: MongoDB collection object
        doc_type: Filter by document type ("agent", "user", or None for all)

    Returns:
        List of documents containing 'freeTools' in featureFlags array
    """
    query = {
        "featureFlags": "freeTools"
    }

    # Add type filter if specified
    if doc_type:
        query["type"] = doc_type

    # Project only fields we need for logging and verification
    projection = {
        "_id": 1,
        "username": 1,
        "type": 1,
        "featureFlags": 1
    }

    documents = list(collection.find(query, projection))
    return documents


def remove_freetools_from_featureflags(collection, doc_type: str = "agent", dry_run: bool = True, verbose: bool = False) -> Dict[str, int]:
    """
    Remove 'freeTools' from the featureFlags array in affected documents.

    Args:
        collection: MongoDB collection object
        doc_type: Filter by document type ("agent", "user", or None for all)
        dry_run: If True, only report changes without executing them
        verbose: If True, print detailed information about each document

    Returns:
        Dictionary with statistics about the operation
    """
    stats = {
        "total_found": 0,
        "total_modified": 0,
        "errors": 0
    }

    logger.info("=" * 80)
    logger.info(f"Starting migration: Remove 'freeTools' from featureFlags array")
    if doc_type:
        logger.info(f"Filtering by type: {doc_type}")
    logger.info("=" * 80)

    # Find all affected documents
    logger.info("Searching for documents with 'freeTools' in featureFlags array...")
    affected_docs = find_documents_with_freetools(collection, doc_type)
    stats["total_found"] = len(affected_docs)

    logger.info(f"Found {stats['total_found']} document(s) with 'freeTools' in featureFlags array")

    if stats["total_found"] == 0:
        logger.info("No documents to modify. Migration complete.")
        return stats

    logger.info("")
    logger.info("-" * 80)

    # Process each document
    for idx, doc in enumerate(affected_docs, 1):
        doc_id = doc["_id"]
        username = doc.get("username", "N/A")
        document_type = doc.get("type", "N/A")
        feature_flags = doc.get("featureFlags", [])

        logger.info(f"Document {idx}/{stats['total_found']}:")
        logger.info(f"  ID: {doc_id}")
        logger.info(f"  Username: {username}")
        logger.info(f"  Type: {document_type}")
        logger.info(f"  Current featureFlags: {feature_flags}")

        if verbose:
            logger.info(f"  Full document: {doc}")

        if dry_run:
            # Calculate what the new array would be
            new_flags = [item for item in feature_flags if item != "freeTools"]
            logger.info(f"  [DRY RUN] Would update featureFlags to: {new_flags}")
        else:
            # Actually perform the update using $pull operator
            try:
                result = collection.update_one(
                    {"_id": doc_id},
                    {"$pull": {"featureFlags": "freeTools"}}
                )

                if result.modified_count > 0:
                    stats["total_modified"] += 1
                    logger.success(f"  ✓ Successfully removed 'freeTools' from featureFlags")
                else:
                    logger.warning(f"  ⚠ No modification made (document may have been updated already)")

            except Exception as e:
                stats["errors"] += 1
                logger.error(f"  ✗ Error updating document: {e}")

        logger.info("-" * 80)

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Migration Summary")
    logger.info("=" * 80)
    logger.info(f"Documents found with 'freeTools' in featureFlags: {stats['total_found']}")

    if dry_run:
        logger.info(f"Mode: DRY RUN (no changes made)")
        logger.info(f"Documents that would be modified: {stats['total_found']}")
    else:
        logger.info(f"Mode: EXECUTION")
        logger.info(f"Documents successfully modified: {stats['total_modified']}")
        logger.info(f"Errors encountered: {stats['errors']}")

    logger.info("=" * 80)

    return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Remove 'freeTools' from preview array in users3 collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run against staging, agents only (default - shows what would change)
  python scripts/remove_freetools_preview.py

  # Dry run against production database, agents only
  python scripts/remove_freetools_preview.py --database prod

  # Dry run against production, users only
  python scripts/remove_freetools_preview.py --database prod --type user

  # Dry run against production, both agents and users
  python scripts/remove_freetools_preview.py --database prod --type all

  # Dry run with verbose output on production agents
  python scripts/remove_freetools_preview.py --database prod --dry-run --verbose

  # Execute the migration on production agents (makes actual changes)
  python scripts/remove_freetools_preview.py --database prod --execute

  # Execute on production users with verbose output
  python scripts/remove_freetools_preview.py --database prod --type user --execute --verbose

IMPORTANT: Always run with --dry-run first to verify the changes!
        """
    )

    parser.add_argument(
        "--database",
        "--db",
        choices=["staging", "prod"],
        default="staging",
        help="Database to run against (default: staging). Use 'prod' for production database."
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["agent", "user", "all"],
        default="agent",
        help="Filter by document type (default: agent). Use 'all' to process both agents and users."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration (default is dry-run mode)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (show changes without applying them) - this is the default"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output including full document details"
    )

    args = parser.parse_args()

    # Normalize database name to uppercase for environment variable
    # "staging" -> "STAGE", "prod" -> "PROD"
    database = "STAGE" if args.database == "staging" else "PROD"

    # CRITICAL: Set the DB environment variable BEFORE importing eve.mongo
    # The get_collection function reads this at call time and caches the collection
    os.environ["DB"] = database

    # Import eve.mongo AFTER setting the environment variable
    from eve.mongo import get_collection
    from bson import ObjectId

    # Determine document type filter
    doc_type = None if args.type == "all" else args.type

    # Determine if we're in dry run mode
    dry_run = not args.execute

    # Display database info
    logger.info("=" * 80)
    logger.info(f"TARGET DATABASE: {database}")
    logger.info(f"DOCUMENT TYPE: {args.type}")
    logger.info(f"MODE: {'DRY RUN' if dry_run else 'EXECUTION'}")
    logger.info("=" * 80)
    logger.info("")

    # Safety check: if --execute is specified, ask for confirmation
    if args.execute:
        logger.warning("⚠️  WARNING: You are about to modify documents in the database!")
        logger.warning(f"⚠️  TARGET DATABASE: {database}")
        logger.warning(f"⚠️  DOCUMENT TYPE: {args.type}")
        logger.warning("⚠️  This will remove 'freeTools' from the featureFlags array in affected documents.")
        logger.warning("")
        logger.warning("⚠️  Recommended: Create a database backup before proceeding!")
        logger.warning("")

        response = input(f"Are you sure you want to proceed on {database} database? Type 'yes' to continue: ")
        if response.lower() != "yes":
            logger.info("Migration cancelled by user.")
            return 1
        logger.info("")

    try:
        # Get the users3 collection
        collection = get_collection("users3")

        # Run the migration
        stats = remove_freetools_from_featureflags(
            collection=collection,
            doc_type=doc_type,
            dry_run=dry_run,
            verbose=args.verbose
        )

        # Exit with appropriate code
        if not dry_run and stats["errors"] > 0:
            logger.error("Migration completed with errors!")
            return 1

        logger.success("Migration completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Fatal error during migration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
