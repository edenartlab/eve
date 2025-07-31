from bson import ObjectId
from datetime import datetime
from typing import Optional
from eve.agent.session.memory_primitives import MemoryType, SessionMemory, UserMemory, AgentMemory

def create_memory_shard(
    agent_id: ObjectId,
    shard_name: str,
    agent_owner: Optional[ObjectId] = None,
    extraction_prompt: Optional[str] = None,
    content: str = ""
) -> AgentMemory:
    """Create a new memory agent shard in the database"""
    
    # Check if shard already exists for this agent
    query = {"agent_id": agent_id, "shard_name": shard_name}
    defaults = {
        "agent_owner": agent_owner,
        "extraction_prompt": extraction_prompt,
        "content": content,
        "is_active": True,
        "last_updated_at": datetime.utcnow()
    }
    
    # Use find_one_or_create to avoid duplicates
    memory_shard = AgentMemory.find_one_or_create(query, defaults)
    return memory_shard

create_memory_shard(
    agent_id=ObjectId("66a916161616161616161616"),
    shard_name="test_shard",
    agent_owner=ObjectId("66a916161616161616161616"),
    extraction_prompt="Extract the most important facts about the user",
    content="This is a test shard"
)