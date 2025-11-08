#!/usr/bin/env python3
"""
Script to update agent memory shards with fact memories.

This script finds all SessionMemory documents that have a specific shard_id and
memory_type='fact', extracts their ObjectIds, and assigns them to the facts field
of the AgentMemory shard belonging to that agent.
"""

import traceback
from bson import ObjectId
from loguru import logger

from eve.agent.memory.memory_models import SessionMemory, AgentMemory


"""

cd /Users/xandersteenbrugge/Documents/GitHub/Eden/eve
DB=PROD PYTHONPATH=/Users/xandersteenbrugge/Documents/GitHub/Eden python -m eve.agent.memory.memory_update_shard

"""


def update_shard_facts(shard_id: ObjectId) -> None:
    """
    Update agent memory shards with fact memories for the given agent.

    Args:
        shard_id: The ObjectId of the shard whose facts should be updated
    """
    try:
        logger.debug(f"Updating shard facts for agent: {shard_id}")

        # Find all SessionMemory documents with the given shard_id and memory_type='fact'
        fact_memories = SessionMemory.find(
            {"shard_id": shard_id, "memory_type": "fact"}
        )

        if not fact_memories:
            logger.debug(f"No fact memories found for agent {shard_id}")
            return

        # Extract ObjectIds from the found SessionMemory documents
        fact_memory_ids = [ObjectId(memory.id) for memory in fact_memories]
        logger.debug(f"Found {len(fact_memory_ids)} fact memories")

        # Find AgentMemory shards with this ObjectId
        shard = AgentMemory.from_mongo(shard_id)

        logger.debug(f"Updating shard: {shard.shard_name}")

        # Set the facts field to the collected ObjectIds
        shard.facts = fact_memory_ids
        shard.save()

        logger.debug(
            f"Updated shard '{shard.shard_name}' with {len(fact_memory_ids)} fact memories"
        )

    except Exception as e:
        logger.error(f"Error updating shard facts for agent {shard_id}: {e}")
        traceback.print_exc()
        raise


def main():
    shard_id = "689e57eebf23f478f81fefc0"  # Koru_Berlin PROD
    shard_id = ObjectId(shard_id)

    logger.debug(f"Starting shard update process for shard: {shard_id}")

    # Execute the update
    update_shard_facts(shard_id)

    logger.debug("Shard update completed successfully!")


if __name__ == "__main__":
    main()
