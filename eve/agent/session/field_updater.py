#!/usr/bin/env python3
"""
Database Field Update Utility

This script allows you to easily add/update fields in MongoDB collections.
It can iterate over all documents in a collection, check if a field exists,
and create/update it with a specified value.

Usage examples:
    # Add field 'new_field' with value 'default_value' to all docs missing it (dry run)
    python field_updater.py --collection users --field new_field --value default_value

    # Update existing field 'status' to 'active' for all documents (actually perform)
    python field_updater.py --collection users --field status --value active --update-existing --dry-run=false

    # Add field only where another field matches a condition (dry run)
    python field_updater.py --collection users --field role --value user --filter '{"type": "normal"}'

Run with:
cd /Users/xandersteenbrugge/Documents/GitHub/Eden/eve
DB=PROD PYTHONPATH=/Users/xandersteenbrugge/Documents/GitHub/Eden python -m eve.agent.session.field_updater

"""

import sys
import json
import argparse
import traceback
from typing import Dict, Any, Optional, Union

from loguru import logger

# Import MongoDB utilities
from eve.mongo import get_collection


def update_field_in_collection(
    collection_name: str,
    field_name: str,
    field_value: Union[str, int, float, bool, dict, list],
    update_existing: bool = False,
    filter_query: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
) -> None:
    """
    Update or add a field in all documents of a collection.

    Args:
        collection_name: Name of the MongoDB collection
        field_name: Name of the field to add/update
        field_value: Value to set for the field
        update_existing: Whether to update existing fields (default: only add missing fields)
        filter_query: Optional query to filter which documents to update
        dry_run: If True, only show what would be updated without making changes
    """
    try:
        logger.info(f"Processing collection: {collection_name}")
        logger.info(f"Field: {field_name} = {field_value}")
        logger.info(f"Update existing: {update_existing}")
        logger.info(f"Filter query: {filter_query}")
        logger.info(f"Dry run: {dry_run}")
        logger.info("-" * 50)

        # Get the collection
        collection = get_collection(collection_name)

        # Build the query
        query = filter_query or {}

        if not update_existing:
            # Only update documents where the field doesn't exist
            query[field_name] = {"$exists": False}

        # Count documents that match
        total_docs = collection.count_documents(query)
        logger.info(f"Found {total_docs} documents to update")

        if total_docs == 0:
            logger.info("No documents match the criteria. Nothing to update.")
            return

        if dry_run:
            # Show sample documents that would be updated
            logger.info("\nSample documents that would be updated:")
            sample_docs = collection.find(query).limit(5)
            for i, doc in enumerate(sample_docs, 1):
                logger.info(
                    f"  {i}. _id: {doc.get('_id')}, current {field_name}: {doc.get(field_name, 'NOT SET')}"
                )
            logger.info(f"\n[DRY RUN] Would update {total_docs} documents")
            return

        # Perform the update
        update_operation = {"$set": {field_name: field_value}}

        logger.info(f"Updating {total_docs} documents...")
        result = collection.update_many(query, update_operation)

        logger.info(f"Successfully updated {result.modified_count} documents")
        logger.info(f"Matched {result.matched_count} documents")

        if result.modified_count != result.matched_count:
            logger.info(
                f"Note: {result.matched_count - result.modified_count} documents already had the same value"
            )

    except Exception as e:
        logger.info(f"Error updating field in collection {collection_name}: {e}")
        traceback.logger.info_exc()
        raise


def parse_value(value_str: str) -> Union[str, int, float, bool, dict, list]:
    """
    Parse string value to appropriate Python type.
    Tries JSON parsing first, then falls back to string.
    """
    # Try to parse as JSON first (handles dict, list, bool, null, numbers)
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        # If JSON parsing fails, treat as string
        return value_str


def main():
    parser = argparse.ArgumentParser(
        description="Add or update fields in MongoDB collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add 'active' field with value true to documents missing it (dry run by default)
  python field_updater.py --collection users3 --field active --value true

  # Actually update all documents' status to 'verified'
  python field_updater.py --collection users3 --field status --value verified --update-existing --dry-run=false

  # Add role='user' only to documents where type='normal' (dry run)
  python field_updater.py --collection users3 --field role --value user --filter '{"type": "normal"}'

  # Set complex object value (actually perform the update)
  python field_updater.py --collection users3 --field metadata --value '{"version": 1, "migrated": true}' --dry-run=false

        """,
    )

    parser.add_argument(
        "--collection", "-c", required=True, help="MongoDB collection name"
    )

    parser.add_argument("--field", "-f", required=True, help="Field name to add/update")

    parser.add_argument(
        "--value",
        "-v",
        required=True,
        help="Value to set (supports JSON format for complex types)",
    )

    parser.add_argument(
        "--update-existing",
        "-u",
        action="store_true",
        help="Update existing fields (default: only add missing fields)",
    )

    parser.add_argument(
        "--filter",
        help='MongoDB query filter as JSON string (e.g., \'{"type": "user"}\')',
    )

    parser.add_argument(
        "--dry-run",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Run in dry-run mode (default: true). Use --dry-run=false to actually perform updates",
    )

    args = parser.parse_args()

    # Parse the value
    field_value = parse_value(args.value)

    # Parse filter query if provided
    filter_query = None
    if args.filter:
        try:
            filter_query = json.loads(args.filter)
        except json.JSONDecodeError as e:
            logger.info(f"Error: Invalid JSON in filter query: {e}")
            sys.exit(1)

    logger.info(f"Starting field update process...")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Field: {args.field}")
    logger.info(f"Value: {field_value} (type: {type(field_value).__name__})")

    # Confirm operation unless it's a dry run
    if not args.dry_run:
        response = input("\nProceed with the update? [y/N]: ")
        if response.lower() != "y":
            logger.info("Operation cancelled.")
            sys.exit(0)

    # Execute the update
    update_field_in_collection(
        collection_name=args.collection,
        field_name=args.field,
        field_value=field_value,
        update_existing=args.update_existing,
        filter_query=filter_query,
        dry_run=args.dry_run,
    )

    logger.info("Field update completed!")


if __name__ == "__main__":
    main()
