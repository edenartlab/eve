from bson import ObjectId
from datetime import datetime
from typing import Optional

import sys, os
filepath_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filepath_dir)
sys.path.append(os.path.dirname(filepath_dir))
sys.path.append(os.path.dirname(os.path.dirname(filepath_dir)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(filepath_dir))))

from eve.agent.session.memory_primitives import AgentMemory

def create_memory_shard(
    agent_id: ObjectId,
    shard_name: str,
    agent_owner: Optional[ObjectId] = None,
    extraction_prompt: Optional[str] = None,
    content: str = ""
) -> AgentMemory:
    """Create a new memory agent shard in the database"""
    print("Creating new memory shard...")
    
    # Check if shard already exists for this agent
    query = {"agent_id": agent_id, "shard_name": shard_name}
    defaults = {
        "agent_owner": agent_owner,
        "extraction_prompt": extraction_prompt,
        "content": content,
        "is_active": True,
        "last_updated_at": datetime.utcnow()
    }
    print(f"Defaults: {defaults}")
    
    # Use find_one_or_create to avoid duplicates
    memory_shard = AgentMemory.find_one_or_create(query, defaults)
    print("Done!")
    return

if 0: # create Eve collective shard:
    create_memory_shard(
        agent_id=ObjectId("675fd3af79e00297cdac1324"),
        shard_name="Collective Play",
        extraction_prompt="We are creating the script for a play that integrates random passages of conversation with all the users you interact with into a single, coherent narrative. Focus on specific, unusual events and facts about the users you're talking to and try to create a beautiful, unique and powerful narrative for a play that integrates all of the users into a single, coherent story.",
        content=""
    )

if 1: # VJ expert shard:
    create_memory_shard(
        agent_id=ObjectId("67bcabed8954656486946377"),
        shard_name="Collective Play",
        extraction_prompt="We are creating the script for a play that integrates random passages of conversation with all the users you interact with into a single, coherent narrative. Focus on specific, unusual events and facts about the users you're talking to and try to create a beautiful, unique and powerful narrative for a play that integrates all of the users into a single, coherent story.",
        content="bla bla bla!"
    )
