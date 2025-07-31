import modal
from bson import ObjectId
from typing import Dict, Any
import logging
import os
from pathlib import Path
import traceback
import json
from datetime import datetime, timezone, timedelta
import sentry_sdk

# Only import ChatMessage when needed (not in standalone mode)
try:
    from eve.agent.session.models import ChatMessage, Session
except ImportError:
    # Running in standalone mode for clearing - ChatMessage not needed
    ChatMessage = None
    Session = None

# Global state dict to track session/memory state per agent_id using modal.Dict
# This acts like redis to store session state and avoid frequent MongoDB queries
pending_session_memories = modal.Dict.from_name("pending-session-memories", create_if_missing=True)

# Default session state structure - defined once to avoid duplication
DEFAULT_SESSION_STATE = {
    "last_activity": None,
    "last_memory_message_id": None,
    "message_count_since_memory": 0,
    "cached_memory_context": None,
    "should_refresh_memory": True,
    "agent_collective_memory_timestamp": None # Timestamp of the last time the agent's collective memory was fetched in this session
}

async def get_session_state(agent_id: ObjectId, session_id: ObjectId) -> Dict[str, Any]:
    """Get session state from modal.Dict, initializing if needed"""
    agent_key = str(agent_id)
    session_key = str(session_id)
    
    # Get agent dict or create if missing
    try:
        agent_dict = pending_session_memories[agent_key]
    except:
        pending_session_memories[agent_key] = {}
        agent_dict = pending_session_memories[agent_key]
    
    # Initialize session state if not present
    if session_key not in agent_dict:
        agent_dict[session_key] = DEFAULT_SESSION_STATE.copy()
        pending_session_memories[agent_key] = agent_dict
    
    return agent_dict[session_key]


async def update_session_state(agent_id: ObjectId, session_id: ObjectId, updates: Dict[str, Any]) -> None:
    """Update session state in modal.Dict"""
    agent_key = str(agent_id)
    session_key = str(session_id)
    
    # Get current agent dict
    agent_dict = pending_session_memories.get(agent_key, {})
    
    # Get current session state
    session_state = agent_dict.get(session_key, DEFAULT_SESSION_STATE.copy())
    
    # Update session state
    session_state.update(updates)
    agent_dict[session_key] = session_state
    
    # Save back to modal.Dict
    pending_session_memories[agent_key] = agent_dict

    print("-----------------------------------")
    print("Updated session_state state:")
    print(json.dumps(session_state, indent=4))
    print("-----------------------------------")

######## Background task to process cold sessions #########

CONSIDER_COLD_AFTER_MINUTES = 5
CLEANUP_COLD_SESSIONS_EVERY_MINUTES = 10

async def process_cold_sessions():
    """
    Process cold sessions (last activity > CONSIDER_COLD_AFTER_MINUTES minutes ago) and trigger memory formation.
    Removes sessions from pending_session_memories after processing to avoid reprocessing.
    """
    print("🧠 Processing cold sessions for memory formation...")
    
    try:
        # Import here to avoid circular imports
        from eve.agent.session.memory import maybe_form_memories
        from eve.agent.session.models import Session
        
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(minutes=CONSIDER_COLD_AFTER_MINUTES)
        
        processed_sessions = 0
        total_sessions = 0

        agent_ids_to_check = list(pending_session_memories.keys())
        print(f"Iterating over {len(agent_ids_to_check)} agents for potential memory formation...")
        
        # Iterate through all agents in pending_session_memories
        for agent_key in agent_ids_to_check:
            agent_id = ObjectId(agent_key)
            agent_dict = pending_session_memories[agent_key]
            
            # Iterate through all sessions for this agent
            sessions_to_remove = []
            for session_key in list(agent_dict.keys()):
                session_id = ObjectId(session_key)
                session_state = agent_dict[session_key]
                
                total_sessions += 1
                
                # Check if session has last_activity
                last_activity_str = session_state.get("last_activity")
                if last_activity_str:
                    try:
                        last_activity = datetime.fromisoformat(last_activity_str.replace('Z', '+00:00'))
                        
                        # Check if session is cold
                        if last_activity < cutoff_time:
                            print(f"Processing cold session {session_id} for agent {agent_id} (last activity: {last_activity})")
                            
                            # Load the session from MongoDB
                            session = Session.from_mongo(session_id)
                            if session:
                                success = await maybe_form_memories(agent_id, session, force_memory_formation=True)
                                if success:
                                    print(f"✓ Memory formation completed for session {session_id}")
                                    processed_sessions += 1
                                else:
                                    print(f"⚠️ Memory formation failed for session {session_id}")
                                
                                # Mark session for removal from pending_session_memories
                                sessions_to_remove.append(session_key)
                            else:
                                print(f"⚠️ Could not load session {session_id} from MongoDB")
                        
                    except Exception as e:
                        print(f"❌ Error processing session {session_id}: {e}")
                        continue
            
            # Remove processed sessions from pending_session_memories
            if sessions_to_remove:
                for session_key in sessions_to_remove:
                    del agent_dict[session_key]
                # Update the agent dict in pending_session_memories
                pending_session_memories[agent_key] = agent_dict
                print(f"Removed {len(sessions_to_remove)} processed sessions from pending_session_memories for agent {agent_id}")
        
        print(f"✓ Cold session processing complete: {processed_sessions}/{total_sessions} sessions had their memories processed")
        
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
