import modal
from bson import ObjectId
from typing import Dict, Any
import os
from pathlib import Path
import traceback
from datetime import datetime, timezone, timedelta
import sentry_sdk

# Import safe functions and wrapper classes
from eve.agent.session.modal_dict_state import (
    ModalDictState, MultiModalDictState, 
    safe_modal_get, safe_modal_set
)

# Global state dict to track session/memory state per agent_id using modal.Dict
# This acts like redis to store session state and avoid frequent MongoDB queries
db = os.getenv("DB", "STAGE").upper()
pending_session_memories = modal.Dict.from_name(f"pending-session-memories-{db.lower()}", create_if_missing=True)

# Global state dict to track agent memory update timestamps
# Key: agent_id (str), Value: {"last_updated_at": timestamp_string}
agent_memory_status = modal.Dict.from_name(f"agent-memory-status-{db.lower()}", create_if_missing=True)

# Global state dict to track user memory update timestamps per agent
# Key: agent_id (str), Value: {user_id (str): {"last_updated_at": timestamp_string}}
user_memory_status = modal.Dict.from_name(f"user-memory-status-{db.lower()}", create_if_missing=True)

# Create ModalDictState wrapper instances for better state management
session_state_manager = ModalDictState(pending_session_memories, "pending_session_memories")
agent_memory_state_manager = ModalDictState(agent_memory_status, "agent_memory_status")
user_memory_state_manager = ModalDictState(user_memory_status, "user_memory_status")

# Combined multi-dict state manager for batch operations
memory_state_manager = MultiModalDictState({
    "sessions": session_state_manager,
    "agent_memory": agent_memory_state_manager,
    "user_memory": user_memory_state_manager
})

# Default session state structure - defined once to avoid duplication
DEFAULT_SESSION_STATE = {
    "last_activity": None,
    "last_memory_message_id": None,
    "message_count_since_memory": 0,
    "cached_memory_context": None,
    "should_refresh_memory": True,
    "agent_collective_memory_timestamp": None, # Timestamp of the last time the agent's collective memory was fetched in this session
    "user_memory_timestamp": None # Timestamp of the last time the user memory was fetched in this session
}

async def _get_or_create_agent_sessions_dict(agent_key: str) -> Dict[str, Any]:
    """Get agent dictionary from modal.Dict, creating if missing."""
    agent_dict = await safe_modal_get(pending_session_memories, agent_key, {})
    if not agent_dict:
        print(f"No agent dict found for agent {agent_key}, creating empty dict")
        await safe_modal_set(pending_session_memories, agent_key, {})
        return {}
    return agent_dict

async def get_session_state(agent_id: ObjectId, session_id: ObjectId) -> Dict[str, Any]:
    """Get session state from modal.Dict, initializing if needed"""
    session_key = str(session_id)
    
    # Use ModalDictState wrapper for cleaner operations
    agent_dict = await session_state_manager.get_agent_dict(agent_id)
    
    if not agent_dict.get(session_key):
        print(f"No session state found for session {session_key}, creating default session state")
        agent_dict[session_key] = DEFAULT_SESSION_STATE.copy()
        await session_state_manager.update_agent_dict(agent_id, agent_dict)
    
    return agent_dict[session_key]

async def update_session_state(agent_id: ObjectId, session_id: ObjectId, updates: Dict[str, Any]) -> None:
    """Update session state in modal.Dict with minimal network calls"""
    session_key = str(session_id)
    
    # Use ModalDictState wrapper for cleaner operations
    agent_dict = await session_state_manager.get_agent_dict(agent_id)
    session_state = agent_dict.get(session_key, DEFAULT_SESSION_STATE.copy())
    
    session_state.update(updates)
    agent_dict[session_key] = session_state
    await session_state_manager.update_agent_dict(agent_id, agent_dict)

    # print("-----------------------------------")
    # print("Updated session_state state:")
    # print(json.dumps(session_state, indent=4))
    # print("-----------------------------------")

######## Background task to process cold sessions #########

# Configuration for cold session processing
CONSIDER_COLD_AFTER_MINUTES         = 5  # Consider a session cold if no activity for this many minutes
CLEANUP_COLD_SESSIONS_EVERY_MINUTES = 10  # Run the background task every N minutes

async def _process_session_for_memory(agent_id: ObjectId, session_id: ObjectId, reason: str = "cold session"):
    """Process a single session for memory formation and return processing results."""
    from eve.agent.session.memory import form_memories, should_form_memories
    from eve.agent.session.models import Session
    
    print(f"Processing {reason} {session_id} for agent {agent_id}")
    
    try:
        session = Session.from_mongo(session_id)
        if not session:
            print(f"⚠️ Could not load session {session_id} from MongoDB")
            return {"processed": False, "skipped": False, "should_remove": True}
        
        if should_form_memories(agent_id, session):
            success = await form_memories(agent_id, session)
            if success:
                print(f"✓ Memory formation completed for session {session_id}")
                return {"processed": True, "skipped": False, "should_remove": True}
            else:
                print(f"⚠️ Memory formation failed for session {session_id}")
                return {"processed": False, "skipped": False, "should_remove": True}
        else:
            print(f"⏭️ Memory formation not needed for session {session_id} (insufficient messages/tokens)")
            return {"processed": False, "skipped": True, "should_remove": True}
            
    except Exception as e:
        print(f"❌ Error processing session {session_id}: {e}")
        traceback.print_exc()
        return {"processed": False, "skipped": False, "should_remove": True}

def _is_session_cold(session_state: Dict[str, Any], cutoff_time: datetime) -> bool:
    """Check if a session is cold based on last activity timestamp."""
    last_activity_str = session_state.get("last_activity", None)
    if not last_activity_str:
        return True  # No timestamp means cold session
    
    try:
        last_activity = datetime.fromisoformat(last_activity_str.replace('Z', '+00:00'))
        return last_activity < cutoff_time
    except Exception:
        return True  # Invalid timestamp means cold session

async def process_cold_sessions():
    """
    Process cold sessions (last activity > CONSIDER_COLD_AFTER_MINUTES minutes ago) and trigger memory formation.
    Removes sessions from pending_session_memories after processing to avoid reprocessing.
    """
    print("🧠 Processing cold sessions for memory formation...")
    
    try:
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(minutes=CONSIDER_COLD_AFTER_MINUTES)
        
        processed_sessions = 0
        skipped_sessions = 0
        total_sessions = 0

        agent_ids_to_check = list(pending_session_memories.keys())
        print(f"Iterating over {len(agent_ids_to_check)} agents for potential memory formation...")
        
        for agent_key in agent_ids_to_check:
            agent_id = ObjectId(agent_key)
            agent_dict = pending_session_memories[agent_key]
            
            sessions_to_remove = []
            for session_key in list(agent_dict.keys()):
                session_id = ObjectId(session_key)
                session_state = agent_dict[session_key]
                total_sessions += 1
                
                if _is_session_cold(session_state, cutoff_time):
                    last_activity_str = session_state.get("last_activity")
                    reason = f"cold session (last activity: {last_activity_str})" if last_activity_str else "session without last_activity timestamp"
                    
                    result = await _process_session_for_memory(agent_id, session_id, reason)
                    
                    if result["processed"]:
                        processed_sessions += 1
                    elif result["skipped"]:
                        skipped_sessions += 1
                    
                    if result["should_remove"]:
                        sessions_to_remove.append(session_key)
            
            if sessions_to_remove:
                for session_key in sessions_to_remove:
                    del agent_dict[session_key]
                pending_session_memories[agent_key] = agent_dict
                print(f"Removed {len(sessions_to_remove)} processed sessions from pending_session_memories for agent {agent_id}")
        
        print(f"✓ Cold session processing complete: {processed_sessions} processed, {skipped_sessions} skipped, {total_sessions} total sessions")
        
    except Exception as e:
        print(f"❌ Error in process_cold_sessions: {e}")
        traceback.print_exc()

# Define periodical background tasks for triggering memory formation
# Modal app setup
from eve import db

## Todo: reuse image from api.py
## from eve.api.api import image

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


"""
modal deploy eve/agent/session/memory_state.py
"""

app = modal.App(
    name="update_agent_memories",
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)

@app.function(
    image=image, max_containers=1, 
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


######## Small utility to clear the pending session memories state on Modal #########

def clear_pending_session_memories():
    """
    Clear all pending session memories from the modal.Dict.
    Useful for debugging and starting fresh.
    """
    try:
        pending_session_memories.clear()
        print("✓ Successfully cleared pending_session_memories modal.Dict")
    except Exception as e:
        print(f"❌ Error clearing pending_session_memories: {e}")
        traceback.print_exc()

def main():
    """Main function to clear the pending session memories state"""
    print("🧹 Clearing pending session memories state...")
    clear_pending_session_memories()
    print("Done!")

if __name__ == "__main__":
    main()
