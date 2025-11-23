#!/usr/bin/env python3
"""
Script: Update User Memories from Markdown Files

This script processes markdown files from a user profiles directory and updates
the corresponding user memory documents in the database with the content from
the markdown files.

Usage:
    # Dry run - print changes without saving (default)
    python scripts/update_user_memories.py

    # Actually execute the updates
    python scripts/update_user_memories.py --execute

Process:
    1. Loop over all .md files in the specified directory
    2. Extract username from filename (without .md extension)
    3. Query users3 collection to find user by username
    4. Extract user_id from the matched document
    5. Query memory_user collection for document with matching user_id
    6. Read content from the markdown file
    7. Update the memory document's 'content' field (if --execute is set)

Safety Features:
    - Dry run mode by default (--execute flag required for actual updates)
    - Detailed logging of each step
    - Shows what will change vs what actually changed
    - Handles missing users and memories gracefully
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# CRITICAL: Set the DB environment variable BEFORE adding eve to path
# The eve.__init__ module reads os.getenv("DB") at import time
os.environ["DB"] = "PROD"

# Add parent directory to path to import eve modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def find_user_by_username(users_collection, username: str) -> Optional[Dict[str, Any]]:
    """
    Find a user document in the users3 collection by username.

    Args:
        users_collection: MongoDB users3 collection object
        username: The username to search for

    Returns:
        User document if found, None otherwise
    """
    query = {"username": username}
    projection = {"_id": 1, "username": 1, "type": 1}

    user_doc = users_collection.find_one(query, projection)
    return user_doc


def find_memory_by_user_id(
    memory_collection, user_id, agent_id: str
) -> Optional[Dict[str, Any]]:
    """
    Find a memory document in the memory_user collection by user_id and agent_id.

    Args:
        memory_collection: MongoDB memory_user collection object
        user_id: The user_id to search for (ObjectId)
        agent_id: The agent_id to search for (string that will be converted to ObjectId)

    Returns:
        Memory document if found, None otherwise
    """
    from bson import ObjectId

    # Convert agent_id string to ObjectId to match database schema
    agent_id_obj = ObjectId(agent_id) if isinstance(agent_id, str) else agent_id
    query = {"user_id": user_id, "agent_id": agent_id_obj}

    memory_doc = memory_collection.find_one(query)
    return memory_doc


def process_user_profiles(
    users_collection,
    memory_collection,
    profiles_dir: Path,
    agent_id: str,
    execute: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Process all .md files in the profiles directory and update memory content.

    Args:
        users_collection: MongoDB users3 collection object
        memory_collection: MongoDB memory_user collection object
        profiles_dir: Directory containing .md profile files
        agent_id: The agent_id to filter memories by (mandatory)
        execute: If True, actually perform database updates. If False, dry run.
        verbose: If True, print detailed information

    Returns:
        Dictionary with statistics about the operation
    """
    stats = {
        "total_files": 0,
        "users_found": 0,
        "users_not_found": 0,
        "memories_found": 0,
        "memories_not_found": 0,
        "updated": 0,
        "would_update": 0,
        "errors": 0,
    }

    logger.info("=" * 80)
    if execute:
        logger.info("Starting User Memory Content Update (EXECUTE MODE)")
    else:
        logger.info("Starting User Memory Content Update (DRY RUN MODE)")
    logger.info("=" * 80)
    logger.info(f"Profiles directory: {profiles_dir}")
    logger.info(f"Execute mode: {execute}")
    logger.info("")

    # Get all .md files in the directory
    md_files = list(profiles_dir.glob("*.md"))
    stats["total_files"] = len(md_files)

    logger.info(f"Found {stats['total_files']} .md file(s) to process")
    logger.info("")
    logger.info("-" * 80)

    # Process each .md file
    for idx, md_file in enumerate(md_files, 1):
        username = md_file.stem  # Filename without extension

        logger.info(f"Processing {idx}/{stats['total_files']}: {md_file.name}")
        logger.info(f"  Username: {username}")

        try:
            # Step 1: Find user in users3 collection
            user_doc = find_user_by_username(users_collection, username)

            if not user_doc:
                logger.warning("  ⚠ User not found in users3 collection")
                stats["users_not_found"] += 1
                logger.info("-" * 80)
                continue

            stats["users_found"] += 1
            user_id = user_doc["_id"]
            user_type = user_doc.get("type", "N/A")

            logger.info(f"  ✓ User found: ID={user_id}, Type={user_type}")

            if verbose:
                logger.info(f"  Full user document: {user_doc}")

            # Step 2: Find memory in memory_user collection
            memory_doc = find_memory_by_user_id(memory_collection, user_id, agent_id)

            if not memory_doc:
                logger.warning("  ⚠ Memory not found in memory_user collection")
                stats["memories_not_found"] += 1
                logger.info("-" * 80)
                continue

            stats["memories_found"] += 1
            old_content = memory_doc.get("content", "")
            memory_id = memory_doc.get("_id")

            logger.info(f"  ✓ Memory found: Memory ID={memory_id}")
            logger.info(f"  Old content length: {len(old_content)} characters")

            # Print current content
            logger.info("  Current memory_user.content:")
            logger.info(f"  {'-' * 60}")
            if old_content:
                logger.info(f"{old_content}")
            else:
                logger.info("  (empty)")
            logger.info(f"  {'-' * 60}")

            # Step 3: Read content from markdown file
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    new_content = f.read()

                logger.info(f"  ✓ Read markdown file: {len(new_content)} characters")

                # Print new content
                logger.info("  New content from markdown file:")
                logger.info(f"  {'-' * 60}")
                if new_content:
                    logger.info(f"{new_content}")
                else:
                    logger.info("  (empty)")
                logger.info(f"  {'-' * 60}")

                # Step 4: Compare and update if different
                if old_content == new_content:
                    logger.info("  ℹ Content is identical, no update needed")
                else:
                    logger.info("  ⚠ Content differs:")
                    logger.info(f"    Old: {len(old_content)} chars")
                    logger.info(f"    New: {len(new_content)} chars")

                    if execute:
                        # Actually perform the update

                        result = memory_collection.update_one(
                            {"_id": memory_id}, {"$set": {"content": new_content}}
                        )

                        if result.modified_count > 0:
                            logger.success("  ✓ UPDATED memory document")
                            stats["updated"] += 1

                            # Verify the update by reading back from database
                            updated_doc = memory_collection.find_one({"_id": memory_id})
                            if updated_doc:
                                updated_content = updated_doc.get("content", "")
                                logger.info("  Post-update memory_user.content:")
                                logger.info(f"  {'-' * 60}")
                                if updated_content:
                                    logger.info(f"{updated_content}")
                                else:
                                    logger.info("  (empty)")
                                logger.info(f"  {'-' * 60}")

                                # Verify it matches what we intended to write
                                if updated_content == new_content:
                                    logger.success(
                                        "  ✓ Verified: Database content matches markdown file"
                                    )
                                else:
                                    logger.error(
                                        "  ✗ Verification failed: Database content does not match!"
                                    )
                            else:
                                logger.error(
                                    "  ✗ Could not read back document for verification"
                                )
                        else:
                            logger.warning(
                                "  ⚠ Update command executed but no document was modified"
                            )
                    else:
                        logger.info("  [DRY RUN] Would update memory document")
                        stats["would_update"] += 1

            except Exception as read_error:
                logger.error(f"  ✗ Error reading markdown file: {read_error}")
                stats["errors"] += 1
                logger.info("-" * 80)
                continue

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"  ✗ Error processing {username}: {e}")
            if verbose:
                import traceback

                traceback.print_exc()

        logger.info("-" * 80)

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Processing Summary")
    logger.info("=" * 80)
    logger.info(f"Total .md files processed: {stats['total_files']}")
    logger.info(f"Users found in database: {stats['users_found']}")
    logger.info(f"Users not found: {stats['users_not_found']}")
    logger.info(f"Memories found: {stats['memories_found']}")
    logger.info(f"Memories not found: {stats['memories_not_found']}")
    if execute:
        logger.info(f"Memory documents updated: {stats['updated']}")
    else:
        logger.info(f"Memory documents that would be updated: {stats['would_update']}")
    logger.info(f"Errors encountered: {stats['errors']}")
    logger.info("=" * 80)

    return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract and save user memory content from markdown profile files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - see what would be updated (default)
  python scripts/update_user_memories.py

  # Actually execute the updates
  python scripts/update_user_memories.py --execute

  # Dry run with verbose output
  python scripts/update_user_memories.py --verbose

  # Execute with custom profiles directory
  python scripts/update_user_memories.py --execute --profiles-dir /path/to/profiles

  # Execute with verbose output
  python scripts/update_user_memories.py --execute --verbose
        """,
    )

    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="/Users/xandersteenbrugge/Downloads/0b1744872b9d9160a0154aa571debe024fd9136dc9f79168562e80a15726c7e3/user_profiles",
        help="Directory containing .md profile files (default: specified in script)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output including full document details",
    )

    parser.add_argument(
        "--agent-id",
        type=str,
        default="690235d0231996f69255e900",
        help="Agent ID to filter memories by (default: 690235d0231996f69255e900)",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually perform database updates. Without this flag, runs in dry-run mode (default: False)",
    )

    args = parser.parse_args()

    # Set database to PROD (hardcoded as per requirements)
    database = "PROD"

    # CRITICAL: Set the DB environment variable BEFORE importing eve.mongo
    # The get_collection function reads this at call time and caches the collection
    os.environ["DB"] = database

    # Import eve.mongo AFTER setting the environment variable
    from eve.mongo import get_collection

    # Convert paths to Path objects
    profiles_dir = Path(args.profiles_dir)

    # Validate profiles directory exists
    if not profiles_dir.exists():
        logger.error(f"Profiles directory does not exist: {profiles_dir}")
        return 1

    if not profiles_dir.is_dir():
        logger.error(f"Profiles path is not a directory: {profiles_dir}")
        return 1

    # Display configuration
    logger.info("=" * 80)
    logger.info(f"TARGET DATABASE: {database}")
    logger.info(f"PROFILES DIRECTORY: {profiles_dir}")
    logger.info(f"AGENT ID: {args.agent_id}")
    logger.info(f"EXECUTE MODE: {args.execute}")
    if not args.execute:
        logger.warning("DRY RUN MODE - No changes will be saved to database")
    logger.info("=" * 80)
    logger.info("")

    # If execute mode, ask for confirmation
    if args.execute:
        logger.warning("=" * 80)
        logger.warning("⚠ WARNING: EXECUTE MODE ENABLED ⚠")
        logger.warning("This will UPDATE the PRODUCTION database!")
        logger.warning("=" * 80)
        logger.info("")

        response = input("Are you sure you want to proceed? Type 'yes' to continue: ")
        if response.lower() != "yes":
            logger.info("Operation cancelled by user.")
            return 0

        logger.info("Proceeding with database updates...")
        logger.info("")

    try:
        # Get the collections
        users_collection = get_collection("users3")
        memory_collection = get_collection("memory_user")

        # Process user profiles
        stats = process_user_profiles(
            users_collection=users_collection,
            memory_collection=memory_collection,
            profiles_dir=profiles_dir,
            agent_id=args.agent_id,
            execute=args.execute,
            verbose=args.verbose,
        )

        # Exit with appropriate code
        if stats["errors"] > 0:
            logger.warning("Processing completed with errors!")
            return 1

        logger.success("Processing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
