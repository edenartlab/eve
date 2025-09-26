import traceback
import math
from typing import List, Optional, Dict
from bson import ObjectId
from eve.mongo import Collection, Document
from datetime import datetime, timezone

from eve.agent.session.memory_constants import MAX_FACTS_PER_SHARD, MAX_AGENT_MEMORIES_BEFORE_CONSOLIDATION, MemoryType
from eve.agent.session.models import ChatMessage, Session
from eve.agent.session.config import DEFAULT_SESSION_SELECTION_LIMIT
from eve.user import User

# Multipliers on the char/token count of different message types for memory importance / formation triggers:
USER_MULTIPLIER  = 1.0
TOOL_MULTIPLIER  = 0.5  # mostly discord_search rn
AGENT_MULTIPLIER = 0.2  # we want memories to come from users, not agents
OTHER_MULTIPLIER = 0.5  # not really used

def calculate_dynamic_limits(base_memory_types: dict, char_counts_by_source: dict, log_power: float = 4.0) -> dict:
    """
    Calculate dynamic memory limits based on conversation text length using logarithmic scaling.

    Args:
        base_memory_types: Dictionary of base memory type configurations
        char_counts_by_source: Character counts by source type
        log_power: Base for logarithmic scaling (default: 2.8 for balanced scaling)
    """
    if not char_counts_by_source:
        return base_memory_types

    # Calculate total weighted character count (same weighting as memory formation)
    total_chars = (
        char_counts_by_source.get("user", 0) * USER_MULTIPLIER +
        char_counts_by_source.get("tool", 0) * TOOL_MULTIPLIER +
        char_counts_by_source.get("other", 0) * OTHER_MULTIPLIER +
        int(char_counts_by_source.get("agent", 0) * AGENT_MULTIPLIER)
    )

    # Use logarithmic scaling to determine multiplier
    base_chars = 3000

    # Logarithmic scaling: log_base(total_chars / base_chars) + 1
    multiplier = math.log(max(total_chars / base_chars, 0.25), log_power) + 1
    multiplier = max(multiplier, 0.5)  # Cap at 0.5x to prevent underflow memory formation
    multiplier = min(multiplier, 2.0)  # Cap at 2.0x to prevent excessive memory formation

    # Create new memory types with scaled limits
    dynamic_memory_types = {}
    for key, memory_type in base_memory_types.items():
        # Scale max_items logarithmically, keep min_items as floor
        original_max = memory_type.max_items
        original_min = memory_type.min_items

        # Apply multiplier to the range (max - min) and add back to min
        range_expansion = (original_max - original_min) * multiplier
        new_max = int(round(original_min + range_expansion))
        new_max = max(original_min, new_max)  # Ensure new_max >= min

        # Create new MemoryType with scaled limits
        dynamic_memory_types[key] = MemoryType(
            name=memory_type.name,
            min_items=original_min,  # Keep original min
            max_items=new_max,       # Scale max
            custom_prompt=memory_type.custom_prompt
        )

    return dynamic_memory_types


def get_sender_id_to_sender_name_map(messages: List[ChatMessage]) -> Dict[ObjectId, str]:
    """Find all unique senders in the messages and return a map of sender id to sender name"""
    unique_sender_ids = {msg.sender for msg in messages if msg.sender}
    
    if not unique_sender_ids:
        return {}
    
    # Perform single MongoDB query with projection to fetch only needed fields
    try:
        from eve.mongo import get_collection
        users_collection = get_collection(User.collection_name)
        
        # Query with projection to only get _id, username, and type fields
        users_cursor = users_collection.find(
            {"_id": {"$in": list(unique_sender_ids)}},
            {"_id": 1, "username": 1, "type": 1}
        )
        
        sender_id_to_sender_name_map = {}
        
        # Process each user from cursor safely
        for user in users_cursor:
            try:
                sender_id_to_sender_name_map[user["_id"]] = f"{user['username']} ({user['type']})"
            except (KeyError, TypeError) as e:
                print(f"Error processing user {user.get('_id', 'unknown')}: {e}")
                traceback.print_exc()
                if "_id" in user:
                    sender_id_to_sender_name_map[user["_id"]] = "unknown"
        
        # Ensure all unique_sender_ids are covered, defaulting to "unknown" for missing ones
        for sender_id in unique_sender_ids:
            if sender_id not in sender_id_to_sender_name_map:
                sender_id_to_sender_name_map[sender_id] = "unknown"
                
        return sender_id_to_sender_name_map
        
    except Exception as e:
        print(f"Error in get_sender_id_to_sender_name_map(): {e}")
        traceback.print_exc()
        return {}

def messages_to_text(messages: List[ChatMessage]) -> tuple[str, dict[str, int]]:
    """Convert messages to readable text for LLM processing

    Returns:
        tuple: (formatted_text, char_counts_by_source)
        - formatted_text: The fully formatted message string
        - char_counts_by_source: Dictionary with character counts per message type
    """
    sender_id_to_sender_name_map = get_sender_id_to_sender_name_map(messages)
    text_parts = []
    char_counts_by_source = {"user": 0, "agent": 0, "tool": 0, "other": 0}

    for msg in messages:
        speaker = sender_id_to_sender_name_map.get(msg.sender) or msg.name or msg.role
        content = msg.content

        # Count original content characters by message type
        if msg.role == "user":
            char_counts_by_source["user"] += len(content)
        elif msg.role in ["agent", "assistant", "eden"]:
            char_counts_by_source["agent"] += len(content)
        else:
            char_counts_by_source["other"] += len(content)

        if msg.tool_calls: # Add tool calls summary if present
            tools_summary = (
                f" [Used tools: {', '.join([tc.tool for tc in msg.tool_calls])}]"
            )
            content += tools_summary
            char_counts_by_source["tool"] += len(tools_summary)

            # Include full tool call results for specific tools
            tools_with_full_results = ["discord_search"] #, "farcaster_search", "twitter_search", "get_tweets"]
            for tc in msg.tool_calls:
                if tc.result and tc.status == "completed" and tc.tool in tools_with_full_results:
                    try:
                        import json
                        result_json = json.dumps(tc.result)
                        tool_result_content = f"\n[{tc.tool} full result: {result_json}]"
                        content += tool_result_content
                        char_counts_by_source["tool"] += len(tool_result_content)
                    except Exception as e:
                        # Fallback: just mention tool was used with results
                        fallback_content = f"\n[{tc.tool} completed with results]"
                        content += fallback_content
                        char_counts_by_source["tool"] += len(fallback_content)
        text_parts.append(f"{speaker}: {content}")
    return "\n".join(text_parts), char_counts_by_source

@Collection("memory_sessions")
class SessionMemory(Document):
    """Individual memory record stored in MongoDB"""

    agent_id: ObjectId
    source_session_id: ObjectId
    memory_type: str
    content: str

    # Context tracking for traceability
    source_message_ids: List[ObjectId] = []
    related_users: List[ObjectId] = []

    # For collective memory tracking - which shard this memory belongs to
    shard_id: Optional[ObjectId] = None
    agent_owner: Optional[ObjectId] = None

    @classmethod
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data for MongoDB storage"""
        # Ensure kwargs is a dict to prevent "argument after ** must be a mapping" error
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data from MongoDB"""
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist for optimal query performance"""
        collection = cls.get_collection()
        
        # Compound index for unabsorbed directives query
        # Query: {"agent_id": agent_id, "memory_type": "directive", "related_users": user_id, "_id": {"$in": ids}}
        collection.create_index([
            ("agent_id", 1),
            ("memory_type", 1), 
            ("related_users", 1)
        ], name="directive_lookup_idx", background=True)
        
        # Compound index for episode memories query
        # Query: {"source_session_id": session_id, "memory_type": MemoryType.EPISODE.value}
        # With sort: sort="createdAt", desc=True
        collection.create_index([
            ("source_session_id", 1),
            ("memory_type", 1),
            ("createdAt", -1)
        ], name="episode_memories_idx", background=True)

@Collection("memory_user")
class UserMemory(Document):
    """Consolidated user memory blob for agent/user pairs"""
    
    agent_id: ObjectId
    user_id: ObjectId
    content: Optional[str] = ""
    agent_owner: Optional[ObjectId] = None
    # Track which directive memories haven't been consolidated yet:
    unabsorbed_memory_ids: List[ObjectId] = []
    # Fully formed user memory containing consolidated content + unabsorbed directives as a single string
    fully_formed_memory: Optional[str] = None
    # Track when the memory blob was last updated:
    last_updated_at: Optional[datetime] = None

    @classmethod
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data for MongoDB storage"""
        # Ensure kwargs is a dict to prevent "argument after ** must be a mapping" error
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data from MongoDB"""
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def find_one_or_create(cls, query, defaults=None):
        """Find a document or create and save a new one if it doesn't exist."""
        if defaults is None:
            defaults = {}

        try:
            doc = cls.find_one(query)
            if doc:
                return doc
        except (TypeError, AttributeError) as e:
            print(f"Error in find_one: {e}")
            traceback.print_exc()
        
        # If we get here, either find_one returned None or crashed
        # Create new instance and save it
        new_doc = {**query, **defaults}
        instance = cls(**new_doc)
        instance.save()
        return instance
        
    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist"""
        collection = cls.get_collection()
        collection.create_index([("agent_id", 1), ("user_id", 1)], unique=True, name="user_memory_lookup_idx", background=True)

@Collection("memory_agent")
class AgentMemory(Document):
    """An agent collective memory blob"""
    agent_id: ObjectId
    shard_name: str # eg "project_name", "event_name", "topic_name", etc
    extraction_prompt: str # A custom prompt indicating what kind of memories to extract for this shard
    
    is_active: Optional[bool] = True # Can be set to False to disable the shard without deleting it
    agent_owner: Optional[ObjectId] = None

    # The consolidated memory blob
    content: Optional[str] = ""
    
    # List of unabsorbed suggestions (raw_memory_ids):
    unabsorbed_memory_ids: List[ObjectId] = []

    # List of relevant facts (raw_memory_ids):
    facts: List[ObjectId] = []

    # Fully formed memory shard containing consolidated content + recent facts + unabsorbed suggestions as a single string
    fully_formed_memory: Optional[str] = None

    # Track when the memory blob was last updated:
    last_updated_at: Optional[datetime] = None

    # Configurable parameters (with defaults from memory_constants.py)
    max_facts_per_shard: Optional[int] = MAX_FACTS_PER_SHARD
    max_agent_memories_before_consolidation: Optional[int] = MAX_AGENT_MEMORIES_BEFORE_CONSOLIDATION

    @classmethod
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data for MongoDB storage"""
        # Ensure kwargs is a dict to prevent "argument after ** must be a mapping" error
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data from MongoDB"""
        if kwargs is None:
            kwargs = {}
        return schema
    
    @classmethod
    def find_one_or_create(cls, query, defaults=None):
        """Find a document or create and save a new one if it doesn't exist."""
        if defaults is None:
            defaults = {}

        try:
            doc = cls.find_one(query)
            if doc:
                return doc
        except (TypeError, AttributeError) as e:
            print(f"Error in find_one: {e}")
            traceback.print_exc()
        
        # If we get here, either find_one returned None or crashed
        # Create new instance and save it
        new_doc = {**query, **defaults}
        instance = cls(**new_doc)
        instance.save()
        return instance
    
    @classmethod
    def ensure_indexes(cls):
        """Create indexes to optimize common queries"""
        collection = cls._get_collection()
        
        # This optimizes: AgentMemory.find({"agent_id": agent_id, "is_active": True})
        collection.create_index([
            ("agent_id", 1),
            ("is_active", 1)
        ], name="agent_id_is_active_idx", background=True)
        
        # This optimizes: AgentMemory.find_one({"agent_id": agent_id, "is_active": True}, sort=[("last_updated_at", -1)])
        collection.create_index([
            ("agent_id", 1),
            ("is_active", 1),
            ("last_updated_at", -1)
        ], name="agent_memory_freshness_idx", background=True)
        
        # Single field index on _id is automatically created by MongoDB
        # This optimizes: AgentMemory.from_mongo(shard_id) which uses _id lookups
        return
    

# Minor utility functions:
def _format_memories_with_age(memories: List[SessionMemory]) -> str:
    """
    Format memories (facts, directives, etc.) with age information.
    """
    if not memories:
        return ""
    
    memory_lines = []
    for memory in memories:
        if memory.createdAt:
            # Ensure both datetimes are timezone-aware
            now_utc = datetime.now(timezone.utc)
            created_at = memory.createdAt.replace(tzinfo=timezone.utc) if memory.createdAt.tzinfo is None else memory.createdAt
            age_days = (now_utc - created_at).days
        else:
            age_days = 0
        memory_lines.append(f"- {memory.content} (age: {age_days} days ago)")
    return "\n".join(memory_lines)

def _get_recent_messages(
    session_messages: List[ChatMessage], 
    last_memory_message_id: ObjectId
) -> List[ChatMessage]:
    """Get messages since the last memory formation."""
    if not last_memory_message_id:
        return session_messages
    
    # Create message ID to index mapping for O(1) lookup
    message_id_to_index = {msg.id: i for i, msg in enumerate(session_messages)}
    last_memory_position = message_id_to_index.get(last_memory_message_id, -1)
    return session_messages[last_memory_position + 1:]

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4.5 characters per token)"""
    return int(len(text) / 4.5)

def get_agent_owner(agent_id: ObjectId) -> Optional[ObjectId]:
    """Get the owner of the agent"""
    try:
        from eve.agent.agent import Agent

        agent = Agent.from_mongo(agent_id)
        return agent.owner
    except Exception as e:
        print(f"Warning: Could not load agent owner for {agent_id}: {e}")
        traceback.print_exc()
        return None


def select_messages(
    session: Session, selection_limit: Optional[int] = DEFAULT_SESSION_SELECTION_LIMIT
):
    messages = ChatMessage.get_collection()
    selected_messages = messages.find({
        "session": session.id, 
        "role": {"$ne": "eden"}
    }).sort(
        "createdAt", -1
    )
    if selection_limit is not None:
        selected_messages = selected_messages.limit(selection_limit)
    selected_messages = list(selected_messages)

    pinned_messages = messages.find({
        "session": session.id, 
        "pinned": True
    })
    pinned_messages = list(pinned_messages)
    pinned_messages = [m for m in pinned_messages if m["_id"] not in [msg["_id"] for msg in selected_messages]]    
    selected_messages.extend(pinned_messages)

    selected_messages.reverse()
    selected_messages = [ChatMessage(**msg) for msg in selected_messages]    
    # Filter out cancelled tool calls from the messages
    selected_messages = [msg.filter_cancelled_tool_calls() for msg in selected_messages]

    return selected_messages
    