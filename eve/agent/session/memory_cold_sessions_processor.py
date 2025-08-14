"""
Memory state management for cold session processing.
This module handles background processing of sessions that need memory formation.
"""

import os
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
import modal
import sentry_sdk

from eve.agent.session.memory_constants import NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES, CONSIDER_COLD_AFTER_MINUTES, CLEANUP_COLD_SESSIONS_EVERY_MINUTES

async def process_cold_sessions():
    """
    Process cold sessions (last activity > CONSIDER_COLD_AFTER_MINUTES minutes ago) and trigger memory formation.
    Uses MongoDB queries to find sessions needing processing.
    """
    print("🧠 Processing cold sessions for memory formation...")
    
    try:
        from eve.agent.session.models import Session
        from eve.agent.session.memory import form_memories
        
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(minutes=CONSIDER_COLD_AFTER_MINUTES)
        hard_filter_date = current_time - timedelta(days=3)
        
        # Query for cold sessions that need memory processing with pagination
        # Handle cases where memory_context may not exist
        BATCH_SIZE = 50  # Process in batches to avoid memory issues
        
        # Simplified query with compound index optimization
        base_query = {
            "updatedAt": {"$gte": hard_filter_date, "$lt": cutoff_time},
            "status": "active"
        }
        
        # First batch: Sessions with memory_context that need processing
        query_with_context = {
            **base_query,
            "memory_context.last_activity": {"$lt": cutoff_time},
            "memory_context.messages_since_memory_formation": {"$gt": NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES}
        }
        
        # Second batch: Sessions without memory_context
        query_without_context = {
            **base_query,
            "memory_context": {"$exists": False}
        }
        
        # Process both query results with limits
        cold_sessions_with_context = Session.find(query_with_context, limit=BATCH_SIZE)
        cold_sessions_without_context = Session.find(query_without_context, limit=BATCH_SIZE)
        
        # Combine results
        cold_sessions = list(cold_sessions_with_context) + list(cold_sessions_without_context)
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for session in cold_sessions:
            try:
                # Get the primary agent for this session
                if not session.agents or len(session.agents) == 0:
                    skipped_count += 1
                    continue
                
                agent_id = session.agents[0]
                
                # Process memory formation
                success = await form_memories(agent_id, session)
                
                if success:
                    processed_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                print(f"❌ Error processing session {session.id}: {e}")
                traceback.print_exc()
                error_count += 1
        
        total_sessions = processed_count + skipped_count + error_count
        print(f"✓ Cold session processing complete: {processed_count} processed, {skipped_count} skipped, {error_count} errors, {total_sessions} total")
        
    except Exception as e:
        print(f"❌ Error in process_cold_sessions: {e}")
        traceback.print_exc()
        sentry_sdk.capture_exception(e)


# Modal app setup for background processing
db = os.getenv("DB", "STAGE").upper()

root_dir = Path(__file__).parent.parent.parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db, "MODAL_SERVE": os.getenv("MODAL_SERVE", "False")})
    .apt_install(
        "git",
        "libmagic1",
        "ffmpeg",
        "wget",
    )
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .add_local_file(str(root_dir / "pyproject.toml"), "/eve/pyproject.toml")
    .add_local_python_source("eve", ignore=[])
)

app = modal.App(
    name="update_agent_memories",
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)

@app.function(
    image=image, 
    min_containers=0,
    max_containers=1,
    scaledown_window=10,
    schedule=modal.Period(minutes=CLEANUP_COLD_SESSIONS_EVERY_MINUTES), 
    timeout=3600
)
async def process_cold_sessions_fn():
    """Scheduled function to process cold sessions every CLEANUP_COLD_SESSIONS_EVERY_MINUTES minutes"""
    try:
        await process_cold_sessions()
    except Exception as e:
        print(f"Error processing cold sessions: {e}")
        sentry_sdk.capture_exception(e)


# Utility function for manual triggering (for debugging)
async def manually_process_cold_sessions():
    """Manually trigger cold session processing for debugging"""
    print("Manually triggering cold session processing...")
    await process_cold_sessions()


if __name__ == "__main__":
    import asyncio
    asyncio.run(manually_process_cold_sessions())