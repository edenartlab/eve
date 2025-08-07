from enum import Enum
from typing import List, Optional
from bson import ObjectId
from eve.mongo import Collection, Document
from pydantic import field_serializer
from datetime import datetime, timezone
import traceback

from eve.agent.session.models import ChatMessage

def messages_to_text(messages: List[ChatMessage]) -> str:
    """Convert messages to readable text for LLM processing"""
    text_parts = []
    for msg in messages:
        speaker = msg.name or msg.role
        content = msg.content

        # Add tool calls summary if present
        if msg.tool_calls:
            tools_summary = (
                f" [Used tools: {', '.join([tc.tool for tc in msg.tool_calls])}]"
            )
            content += tools_summary

        text_parts.append(f"{speaker}: {content}")
    return "\n".join(text_parts)

async def _update_agent_memory_timestamp(agent_id: ObjectId):
    """
    Update the agent memory status timestamp to indicate collective memory has changed.
    This allows sessions to detect when they need to refresh their cached agent memory.
    """
    try:
        from eve.agent.session.memory_state import agent_memory_status
        
        agent_key = str(agent_id)
        current_time = datetime.now(timezone.utc).isoformat()

        if agent_key in agent_memory_status:
            agent_memory_status[agent_key]["last_updated_at"] = current_time
        else:
            agent_memory_status[agent_key] = {"last_updated_at": current_time}
        
    except Exception as e:
        print(f"Error updating agent memory status for agent {agent_id}: {e}")
        traceback.print_exc()


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
        ])
        
        # Compound index for episode memories query
        # Query: {"source_session_id": session_id, "memory_type": MemoryType.EPISODE.value}
        # With sort: sort="createdAt", desc=True
        collection.create_index([
            ("source_session_id", 1),
            ("memory_type", 1),
            ("createdAt", -1)
        ])

@Collection("memory_user")
class UserMemory(Document):
    """Consolidated user memory blob for agent/user pairs"""

    agent_id: ObjectId
    user_id: ObjectId
    content: Optional[str] = ""
    agent_owner: Optional[ObjectId] = None
    # Track which directive memories haven't been consolidated yet:
    unabsorbed_memory_ids: List[ObjectId] = []
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
        collection.create_index([("agent_id", 1), ("user_id", 1)], unique=True)

@Collection("memory_agent")
class AgentMemory(Document):
    """An agent collective memory blob"""
    agent_id: ObjectId
    agent_owner: Optional[ObjectId] = None
    shard_name: Optional[str] = None # eg "project_name", "event_name", "topic_name", etc
    is_active: bool = True
    extraction_prompt: Optional[str] = None

    # The consolidated memory blob
    content: Optional[str] = ""
    
    # List of unabsorbed suggestions (raw_memory_ids):
    unabsorbed_memory_ids: List[ObjectId] = []

    # List of relevant facts (raw_memory_ids):
    facts: List[ObjectId] = []

    # Fully formed memory shard containing consolidated content + recent facts + unabsorbed suggestions as a single string
    fully_formed_memory_shard: Optional[str] = ""

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
        
        # If we get here, either find_one returned None or crashed
        # Create new instance and save it
        new_doc = {**query, **defaults}
        instance = cls(**new_doc)
        instance.save()
        return instance
    


# Minor utility functions:
def _format_facts_with_age(facts: List[SessionMemory]) -> str:
    """
    Format facts with age information.
    """
    if not facts:
        return ""
    
    fact_lines = []
    for fact in facts:
        if fact.createdAt:
            # Ensure both datetimes are timezone-aware
            now_utc = datetime.now(timezone.utc)
            created_at = fact.createdAt.replace(tzinfo=timezone.utc) if fact.createdAt.tzinfo is None else fact.createdAt
            age_days = (now_utc - created_at).days
        else:
            age_days = 0
        fact_lines.append(f"- {fact.content} (age: {age_days} days ago)")
    return "\n".join(fact_lines)

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
        return None
    