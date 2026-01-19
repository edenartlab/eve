"""
Memory System v2 - Cold Session Processing.

This module handles background processing of sessions that need memory formation.
It finds "cold" sessions (inactive for 10+ minutes with sufficient messages) and
triggers memory formation for them.

Run locally:
cd /Users/xandersteenbrugge/Documents/GitHub/Eden/eve
DB=STAGE PYTHONPATH=/Users/xandersteenbrugge/Documents/GitHub/Eden python -m eve.agent.memory2.memory2_cold_sessions_processor

# Monitor deployments
modal app list
modal app logs memory2_process_cold_sessions

# Stop deployment
modal app stop memory2_process_cold_sessions
"""

import os
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import modal
import sentry_sdk
from loguru import logger

# Cleanup interval - how often this job runs
CLEANUP_COLD_SESSIONS_EVERY_MINUTES = 10


async def process_cold_sessions():
    """
    Process cold sessions (last activity > CONSIDER_COLD_AFTER_MINUTES minutes ago)
    and trigger memory formation using memory system v2.

    Uses MongoDB queries to find sessions needing processing.
    """
    # Import here to avoid issues with Modal image building
    from eve.agent.memory2.constants import (
        CONSIDER_COLD_AFTER_MINUTES,
        LOCAL_DEV,
        NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES,
    )

    if LOCAL_DEV:
        logger.debug(
            "Cold session processing is disabled in local development mode."
        )
        return

    logger.debug("🧠 [Memory2] Processing cold sessions for memory formation...")

    try:
        from eve.agent.memory2.formation import process_cold_session
        from eve.agent.session.models import ChatMessage, Session

        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(minutes=CONSIDER_COLD_AFTER_MINUTES)
        hard_filter_date = current_time - timedelta(days=2)

        # Query for cold sessions that need memory processing with pagination
        # Handle cases where memory_context may not exist
        MAX_SESSIONS_TO_PROCESS = 100  # Process in batches to avoid memory issues

        # Simplified query with compound index optimization
        base_query = {
            "updatedAt": {"$gte": hard_filter_date, "$lt": cutoff_time},
            "status": "active",
            "extras.incognito": {
                "$ne": True
            },  # Exclude incognito sessions from memory formation
        }

        # First batch: Sessions with memory_context that need processing
        query_with_context = {
            **base_query,
            "memory_context.last_activity": {"$lt": cutoff_time},
            "memory_context.messages_since_memory_formation": {
                "$gte": NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES
            },
        }

        logger.debug("Running queries...")
        cold_sessions_with_context = Session.find(
            query_with_context, limit=MAX_SESSIONS_TO_PROCESS // 2
        )

        # Second batch: Sessions without memory_context (legacy sessions)
        messages_collection = ChatMessage.get_collection()
        pipeline = [
            {
                "$match": {
                    "session": {"$ne": None},
                    "role": {"$ne": "eden"},
                    "createdAt": {"$gte": hard_filter_date, "$lt": cutoff_time},
                }
            },
            # Unwind session array (since ChatMessage.session is now an array)
            {"$unwind": "$session"},
            {
                "$group": {
                    "_id": "$session",
                    "count": {"$sum": 1},
                    "last_message_at": {"$max": "$createdAt"},
                }
            },
            {
                "$match": {
                    "count": {"$gte": NEVER_FORM_MEMORIES_LESS_THAN_N_MESSAGES},
                    "last_message_at": {"$lt": cutoff_time},
                }
            },
            {"$sort": {"last_message_at": -1}},
            {"$limit": MAX_SESSIONS_TO_PROCESS},
        ]

        candidate_session_ids = [
            doc["_id"]
            for doc in messages_collection.aggregate(pipeline, allowDiskUse=True)
            if doc.get("_id")
        ]

        logger.debug(
            f"Aggregation found {len(candidate_session_ids)} candidate sessions without context"
        )

        processed_session_ids = {session.id for session in cold_sessions_with_context}
        candidate_session_ids = [
            session_id
            for session_id in candidate_session_ids
            if session_id not in processed_session_ids
        ][: MAX_SESSIONS_TO_PROCESS // 2]

        logger.debug(
            f"Considering {len(candidate_session_ids)} candidate sessions after filtering processed ones"
        )

        cold_sessions_without_context = []
        if candidate_session_ids:
            query_without_context = {
                **base_query,
                "_id": {"$in": candidate_session_ids},
                "$or": [
                    {"memory_context": {"$exists": False}},
                    {"memory_context": None},
                ],
            }
            cold_sessions_without_context = Session.find(
                query_without_context, limit=len(candidate_session_ids)
            )

        # Combine results
        cold_sessions = list(cold_sessions_with_context) + list(
            cold_sessions_without_context
        )
        logger.debug(
            f"Found {len(cold_sessions_with_context)} cold sessions with context"
        )
        logger.debug(
            f"Found {len(cold_sessions_without_context)} cold sessions without context"
        )
        logger.debug(f"Found {len(cold_sessions)} total cold sessions to process")

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

                # Process memory formation using memory2
                success = await process_cold_session(
                    session=session,
                    agent_id=agent_id,
                )

                if success:
                    processed_count += 1
                else:
                    error_count += 1

            except Exception as e:
                logger.error(f"❌ Error processing session {session.id}: {e}")
                traceback.print_exc()
                error_count += 1

        total_sessions = processed_count + skipped_count + error_count
        logger.debug(
            f"✓ [Memory2] Cold session processing complete: {processed_count} processed, "
            f"{skipped_count} skipped, {error_count} errors, {total_sessions} total"
        )

    except Exception as e:
        logger.error(f"❌ Error in process_cold_sessions: {e}")
        traceback.print_exc()
        sentry_sdk.capture_exception(e)


# Modal app setup for background processing
db = os.getenv("DB", "STAGE").upper()

root_dir = Path(__file__).parent.parent.parent.parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db, "MODAL_SERVE": "1"})
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
    name=f"memory2_process_cold_sessions-{db.upper()}",
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
    timeout=3600,
)
async def process_cold_sessions_fn():
    """Scheduled function to process cold sessions every 10 minutes using memory2"""
    try:
        await process_cold_sessions()
    except Exception as e:
        logger.error(f"Error processing cold sessions: {e}")
        sentry_sdk.capture_exception(e)


# Utility function for manual triggering (for debugging)
async def manually_process_cold_sessions():
    """Manually trigger cold session processing for debugging"""
    logger.debug("Manually triggering cold session processing (memory2)...")
    await process_cold_sessions()


if __name__ == "__main__":
    import asyncio

    asyncio.run(manually_process_cold_sessions())
