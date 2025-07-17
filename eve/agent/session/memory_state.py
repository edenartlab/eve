import modal
from bson import ObjectId
from typing import Dict, Any, Optional
import logging
import traceback
import json

# Only import ChatMessage when needed (not in standalone mode)
try:
    from eve.agent.session.models import ChatMessage
except ImportError:
    # Running in standalone mode for clearing - ChatMessage not needed
    ChatMessage = None

# Global state dict to track session/memory state per agent_id using modal.Dict
# This acts like redis to store session state and avoid frequent MongoDB queries
pending_session_memories = modal.Dict.from_name("pending-session-memories", create_if_missing=True)

# Default session state structure - defined once to avoid duplication
DEFAULT_SESSION_STATE = {
    "last_activity": None,
    "last_memory_message_id": None,
    "message_count_since_memory": 0,
    "cached_memory_context": None,
    "should_refresh_memory": True
}

def get_session_state(agent_id: ObjectId, session_id: ObjectId) -> Dict[str, Any]:
    """Get session state from modal.Dict, initializing if needed"""
    agent_key = str(agent_id)
    session_key = str(session_id)
    
    # Get agent dict or create if missing
    try:
        agent_dict = pending_session_memories[agent_key]
    except:
        pending_session_memories[agent_key] = {}
        agent_dict = pending_session_memories[agent_key]
    
    if session_key not in agent_dict:
        agent_dict[session_key] = DEFAULT_SESSION_STATE.copy()
        pending_session_memories[agent_key] = agent_dict
    
    return agent_dict[session_key]


def update_session_state(agent_id: ObjectId, session_id: ObjectId, updates: Dict[str, Any]) -> None:
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
    print("Updated agent_dict state:")
    print(json.dumps(agent_dict, indent=4))
    print("-----------------------------------")


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
