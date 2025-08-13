"""
Migration script to transition from Modal Dict based memory system to Session-based memory system.
This script can be run to migrate existing sessions to the new memory context structure.
"""

import asyncio
from bson import ObjectId
from datetime import datetime, timezone
from eve.agent.session.models import Session
from eve.mongo import get_db


async def migrate_sessions():
    """
    Migrate existing sessions to use the new SessionMemoryContext structure.
    This adds default values for new fields that didn't exist before.
    """
    print("Starting session memory context migration...")
    
    db = get_db()
    sessions_collection = db.sessions
    
    # Find all sessions that don't have the new memory_context structure
    sessions_to_migrate = sessions_collection.find({
        "$or": [
            {"memory_context": {"$exists": False}},
            {"memory_context.messages_since_memory_formation": {"$exists": False}}
        ]
    })
    
    migrated_count = 0
    error_count = 0
    
    for session_data in sessions_to_migrate:
        try:
            session_id = session_data["_id"]
            
            # Build the new memory_context structure
            memory_context = {
                "cached_memory_context": session_data.get("context", {}).get("cached_memory_context"),
                "memory_context_timestamp": session_data.get("context", {}).get("memory_context_timestamp"),
                "last_activity": session_data.get("updatedAt", datetime.now(timezone.utc)),
                "last_memory_message_id": session_data.get("last_memory_message_id"),
                "messages_since_memory_formation": 0,  # Start fresh
                "agent_memory_timestamp": None,
                "user_memory_timestamp": None,
                "cached_episode_memories": None,
                "episode_memories_timestamp": None
            }
            
            # Update the session
            sessions_collection.update_one(
                {"_id": session_id},
                {"$set": {"memory_context": memory_context}}
            )
            
            migrated_count += 1
            print(f"✓ Migrated session {session_id}")
            
        except Exception as e:
            error_count += 1
            print(f"❌ Error migrating session {session_data.get('_id')}: {e}")
    
    print(f"\nMigration complete: {migrated_count} sessions migrated, {error_count} errors")
    
    # Create indexes for efficient querying
    print("\nCreating indexes for cold session processing...")
    sessions_collection.create_index([
        ("memory_context.last_activity", 1),
        ("memory_context.messages_since_memory_formation", 1),
        ("status", 1)
    ])
    print("✓ Indexes created")


async def cleanup_modal_dicts():
    """
    Optional: Clean up Modal Dict state after migration.
    Only run this after confirming the new system is working correctly.
    """
    print("\n⚠️  WARNING: This will delete all Modal Dict state for the memory system.")
    response = input("Are you sure you want to proceed? (yes/no): ")
    
    if response.lower() != "yes":
        print("Cleanup cancelled.")
        return
    
    try:
        import modal
        import os
        
        db = os.getenv("DB", "STAGE").upper()
        
        # Delete the old Modal Dict objects
        dicts_to_delete = [
            f"session_state-{db.lower()}",
            f"agent-memory-status-{db.lower()}",
            f"user-memory-status-{db.lower()}"
        ]
        
        for dict_name in dicts_to_delete:
            try:
                modal_dict = modal.Dict.from_name(dict_name)
                modal_dict.clear()
                print(f"✓ Cleared Modal Dict: {dict_name}")
            except Exception as e:
                print(f"⚠️  Could not clear {dict_name}: {e}")
        
        print("\n✓ Modal Dict cleanup complete")
        
    except ImportError:
        print("Modal not installed, skipping Modal Dict cleanup")


async def verify_migration():
    """
    Verify that the migration was successful by checking a few sessions.
    """
    print("\nVerifying migration...")
    
    db = get_db()
    sessions_collection = db.sessions
    
    # Check a few recent sessions
    recent_sessions = sessions_collection.find({
        "status": "active"
    }).sort("updatedAt", -1).limit(5)
    
    all_valid = True
    for session_data in recent_sessions:
        session_id = session_data["_id"]
        memory_context = session_data.get("memory_context", {})
        
        # Check required fields
        required_fields = [
            "messages_since_memory_formation",
            "last_activity"
        ]
        
        missing_fields = [f for f in required_fields if f not in memory_context]
        
        if missing_fields:
            print(f"⚠️  Session {session_id} missing fields: {missing_fields}")
            all_valid = False
        else:
            print(f"✓ Session {session_id} has valid memory_context")
    
    if all_valid:
        print("\n✓ Migration verification passed!")
    else:
        print("\n⚠️  Some sessions may need manual review")


async def main():
    """
    Main migration function.
    """
    print("=" * 60)
    print("Memory System Migration Script")
    print("=" * 60)
    
    # Step 1: Migrate sessions
    await migrate_sessions()
    
    # Step 2: Verify migration
    await verify_migration()
    
    # Step 3: Optional cleanup (commented out by default for safety)
    # await cleanup_modal_dicts()
    
    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())