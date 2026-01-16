"""
Memory System v2 - Fact Storage

This module handles the storage and retrieval of facts from MongoDB.
It provides utilities for:
- Storing facts with embeddings
- Querying facts by scope
- Managing fact lifecycle (update, delete)
"""

import traceback
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.constants import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, LOCAL_DEV
from eve.agent.memory2.models import Fact


async def get_embedding(text: str) -> List[float]:
    """
    Get embedding for text using OpenAI's embedding model.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding vector
    """
    try:
        import openai

        client = openai.AsyncOpenAI()
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        traceback.print_exc()
        return [0.0] * EMBEDDING_DIMENSIONS


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for multiple texts in a single API call.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    try:
        import openai

        client = openai.AsyncOpenAI()
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        # Sort by index to maintain order
        embeddings = [None] * len(texts)
        for item in response.data:
            embeddings[item.index] = item.embedding

        return embeddings

    except Exception as e:
        logger.error(f"Error getting batch embeddings: {e}")
        traceback.print_exc()
        # Fallback to individual embeddings
        return [await get_embedding(text) for text in texts]


async def store_fact(
    content: str,
    scope: List[Literal["user", "agent"]],
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    message_ids: Optional[List[ObjectId]] = None,
) -> Optional[Fact]:
    """
    Store a single fact with embedding.

    Args:
        content: Fact content
        scope: List of scopes
        agent_id: Agent ID
        user_id: User ID (if user in scope)
        session_id: Source session ID
        message_ids: Source message IDs

    Returns:
        Saved Fact document, or None on error
    """
    try:
        # Get embedding
        embedding = await get_embedding(content)

        # Create fact document
        fact = Fact(
            content=content,
            scope=scope,
            agent_id=agent_id,
            user_id=user_id if "user" in scope else None,
            session_id=session_id,
            source_message_ids=message_ids or [],
            embedding=embedding,
        )

        # Save
        fact.save()

        if LOCAL_DEV:
            logger.debug(f"Stored fact: {content[:50]}...")

        return fact

    except Exception as e:
        logger.error(f"Error storing fact: {e}")
        traceback.print_exc()
        return None


async def store_facts_batch(
    facts_data: List[Dict],
) -> List[Fact]:
    """
    Store multiple facts with embeddings in batch.

    More efficient than storing one at a time due to batch embedding.

    Args:
        facts_data: List of fact dictionaries with keys:
            - content: str
            - scope: List[str]
            - agent_id: ObjectId
            - user_id: Optional[ObjectId]
            - session_id: Optional[ObjectId]
            - source_message_ids: Optional[List[ObjectId]]

    Returns:
        List of saved Fact documents
    """
    if not facts_data:
        return []

    try:
        # Get embeddings for all facts at once
        contents = [f["content"] for f in facts_data]
        embeddings = await get_embeddings_batch(contents)

        # Create fact documents
        facts = []
        for i, fd in enumerate(facts_data):
            fact = Fact(
                content=fd["content"],
                scope=fd["scope"],
                agent_id=fd["agent_id"],
                user_id=fd.get("user_id"),
                session_id=fd.get("session_id"),
                source_message_ids=fd.get("source_message_ids", []),
                embedding=embeddings[i] if i < len(embeddings) else [],
            )
            facts.append(fact)

        # Batch save
        try:
            Fact.save_many(facts)
        except Exception as e:
            logger.error(f"Batch save failed for facts, falling back: {e}")
            for fact in facts:
                fact.save()

        if LOCAL_DEV:
            logger.debug(f"Stored {len(facts)} facts in batch")

        return facts

    except Exception as e:
        logger.error(f"Error in store_facts_batch: {e}")
        traceback.print_exc()
        return []


async def update_fact(
    fact_id: ObjectId,
    new_content: str,
) -> Optional[Fact]:
    """
    Update a fact's content and re-embed.

    Args:
        fact_id: ID of fact to update
        new_content: New content for the fact

    Returns:
        Updated Fact document, or None on error
    """
    try:
        # Load fact
        fact = Fact.from_mongo(fact_id)
        if not fact:
            logger.error(f"Fact not found: {fact_id}")
            return None

        # Get new embedding
        new_embedding = await get_embedding(new_content)

        # Update fact
        fact.update_content(new_content)
        fact.embedding = new_embedding
        fact.save()

        if LOCAL_DEV:
            logger.debug(f"Updated fact {fact_id}: {new_content[:50]}...")

        return fact

    except Exception as e:
        logger.error(f"Error updating fact: {e}")
        traceback.print_exc()
        return None


def delete_fact(fact_id: ObjectId) -> bool:
    """
    Delete a fact from the database.

    Args:
        fact_id: ID of fact to delete

    Returns:
        True if deleted successfully
    """
    try:
        fact = Fact.from_mongo(fact_id)
        if fact:
            fact.delete()
            if LOCAL_DEV:
                logger.debug(f"Deleted fact {fact_id}")
            return True
        return False

    except Exception as e:
        logger.error(f"Error deleting fact: {e}")
        traceback.print_exc()
        return False


def get_facts_by_scope(
    agent_id: ObjectId,
    scope: Literal["user", "agent"],
    user_id: Optional[ObjectId] = None,
    limit: int = 100,
) -> List[Fact]:
    """
    Get facts filtered by scope.

    Args:
        agent_id: Agent ID
        scope: Scope to filter by
        user_id: User ID (required for user scope)
        limit: Maximum number of facts to return

    Returns:
        List of Fact documents
    """
    try:
        query = {
            "agent_id": agent_id,
            "scope": scope,
        }

        if scope == "user" and user_id:
            query["user_id"] = user_id

        return Fact.find(query, limit=limit, sort="createdAt", desc=True)

    except Exception as e:
        logger.error(f"Error getting facts by scope: {e}")
        return []


def get_fact_count(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
) -> Dict[str, int]:
    """
    Get fact counts by scope.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional)

    Returns:
        Dict with counts: {"agent": int, "user": int, "total": int}
    """
    try:
        collection = Fact.get_collection()

        # Count agent-scoped facts
        agent_count = collection.count_documents({
            "agent_id": agent_id,
            "scope": "agent",
        })

        # Count user-scoped facts
        user_count = 0
        if user_id:
            user_count = collection.count_documents({
                "agent_id": agent_id,
                "user_id": user_id,
                "scope": "user",
            })

        return {
            "agent": agent_count,
            "user": user_count,
            "total": agent_count + user_count,
        }

    except Exception as e:
        logger.error(f"Error getting fact count: {e}")
        return {"agent": 0, "user": 0, "total": 0}


async def check_duplicate_by_hash(
    content: str,
    agent_id: ObjectId,
) -> Optional[Fact]:
    """
    Check if a fact with the same hash already exists.

    Args:
        content: Content to check
        agent_id: Agent ID

    Returns:
        Existing Fact if found, None otherwise
    """
    try:
        fact_hash = Fact._compute_hash(content)

        existing = Fact.find_one({
            "hash": fact_hash,
            "agent_id": agent_id,
        })

        return existing

    except Exception as e:
        logger.error(f"Error checking duplicate: {e}")
        return None
