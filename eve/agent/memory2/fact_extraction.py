"""
Memory System v2 - Fact Extraction

This module handles the extraction of facts from conversations.
Facts are atomic, objective statements that are stored in the vector database
for RAG retrieval.

Fact extraction is the FIRST LLM call in the pipeline, using a fast/cheap model.
The extracted facts are then:
1. Processed through the deduplication pipeline (fact_management.py)
2. Passed to reflection extraction to avoid redundancy
"""

import os
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

from bson import ObjectId
from loguru import logger

from eve.agent.llm.llm import async_prompt
from eve.agent.memory2.constants import (
    FACT_EXTRACTION_PROMPT,
    FACT_MAX_WORDS,
    LOCAL_DEV,
    MEMORY_LLM_MODEL_FAST,
)
from eve.utils.system_utils import async_exponential_backoff
from eve.agent.memory2.models import (
    ExtractedFact,
    Fact,
    FactExtractionResponse,
)
from eve.agent.session.models import (
    ChatMessage,
    LLMConfig,
    LLMContext,
    LLMContextMetadata,
    LLMTraceMetadata,
)


async def extract_facts(
    conversation_text: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    model: str = MEMORY_LLM_MODEL_FAST,
) -> List[ExtractedFact]:
    """
    Extract facts from conversation text using LLM.

    This is the FIRST LLM call in the extraction pipeline.
    It uses a fast/cheap model since fact extraction doesn't need
    memory context awareness.

    Facts have NO session scope - only user and/or agent scope.
    Session-level context is handled entirely by session reflections.

    Args:
        conversation_text: The conversation to extract facts from
        agent_id: Agent ID
        user_id: User ID (optional, for user-scoped facts)
        session_id: Session ID (for tracing only)
        model: LLM model to use

    Returns:
        List of ExtractedFact objects with content and scope
    """
    try:
        # Build prompt
        prompt = FACT_EXTRACTION_PROMPT.format(
            conversation_text=conversation_text,
            max_words=FACT_MAX_WORDS,
        )

        # LLM call with structured output
        context = LLMContext(
            messages=[ChatMessage(role="user", content=prompt)],
            config=LLMConfig(
                model=model,
                response_format=FactExtractionResponse,
            ),
            metadata=LLMContextMetadata(
                session_id=f"{os.getenv('DB')}-{str(session_id)}"
                if session_id
                else f"{os.getenv('DB')}-memory2-fact-extraction",
                trace_name="FN_memory2_extract_facts",
                trace_id=str(uuid.uuid4()),
                generation_name="memory2_fact_extraction",
                trace_metadata=LLMTraceMetadata(
                    session_id=str(session_id) if session_id else None,
                    user_id=str(user_id) if user_id else None,
                    agent_id=str(agent_id),
                ),
            ),
            enable_tracing=True,
        )

        if LOCAL_DEV:
            logger.debug("Running fact extraction LLM call...")

        # LLM call with automatic retry (3 attempts with exponential backoff)
        response = await async_exponential_backoff(
            lambda: async_prompt(context),
            max_attempts=3,
            initial_delay=2,
            max_jitter=0.5,
        )

        # Parse response
        if hasattr(response, "parsed"):
            extracted = response.parsed
        else:
            extracted = FactExtractionResponse.model_validate_json(response.content)

        if LOCAL_DEV:
            logger.debug(f"Extracted {len(extracted.facts)} facts")
            for fact in extracted.facts:
                logger.debug(f"  - [{', '.join(fact.scope)}] {fact.content[:60]}...")

        return extracted.facts

    except Exception as e:
        logger.error(f"Error extracting facts: {e}")
        traceback.print_exc()
        return []


async def extract_and_prepare_facts(
    conversation_text: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    message_ids: Optional[List[ObjectId]] = None,
) -> Tuple[List[Dict], List[str]]:
    """
    Extract facts and prepare them for storage.

    This function:
    1. Calls LLM to extract facts
    2. Prepares fact documents (without embedding - done in fact_management)
    3. Returns both prepared facts and simple content list for reflection extraction

    Args:
        conversation_text: The conversation to extract facts from
        agent_id: Agent ID
        user_id: User ID for user-scoped facts
        session_id: Session ID
        message_ids: IDs of messages that were processed

    Returns:
        Tuple of (prepared_fact_dicts, fact_content_strings)
    """
    try:
        # Extract facts
        extracted_facts = await extract_facts(
            conversation_text=conversation_text,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
        )

        if not extracted_facts:
            return [], []

        # Prepare fact documents
        prepared_facts = []
        fact_contents = []

        for ef in extracted_facts:
            # Validate scope
            if not ef.scope or not any(s in ["user", "agent"] for s in ef.scope):
                logger.warning(f"Invalid scope for fact: {ef.scope}")
                continue

            # Skip empty content
            if not ef.content or not ef.content.strip():
                continue

            fact_dict = {
                "content": ef.content.strip(),
                "scope": ef.scope,
                "agent_id": agent_id,
                "user_id": user_id if "user" in ef.scope else None,
                "source_session_id": session_id,
                "source_message_ids": message_ids or [],
            }
            prepared_facts.append(fact_dict)
            fact_contents.append(ef.content.strip())

        if LOCAL_DEV:
            logger.debug(f"Prepared {len(prepared_facts)} facts for processing")

        return prepared_facts, fact_contents

    except Exception as e:
        logger.error(f"Error in extract_and_prepare_facts: {e}")
        traceback.print_exc()
        return [], []


def create_fact_document(
    content: str,
    scope: List[str],
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    message_ids: Optional[List[ObjectId]] = None,
    embedding: Optional[List[float]] = None,
) -> Fact:
    """
    Create a Fact document ready for storage.

    Args:
        content: Fact content
        scope: List of scopes ["user", "agent"]
        agent_id: Agent ID
        user_id: User ID (if user in scope)
        session_id: Source session ID
        message_ids: Source message IDs
        embedding: Pre-computed embedding (optional)

    Returns:
        Fact document (not yet saved)
    """
    return Fact(
        content=content,
        scope=scope,
        agent_id=agent_id,
        user_id=user_id if "user" in scope else None,
        source_session_id=session_id,
        source_message_ids=message_ids or [],
        embedding=embedding or [],
    )


async def save_facts(
    facts: List[Fact],
) -> List[Fact]:
    """
    Save fact documents to the database.

    Args:
        facts: List of Fact documents to save

    Returns:
        List of saved facts with IDs
    """
    if not facts:
        return []

    try:
        # Batch save
        try:
            Fact.save_many(facts)
        except Exception as e:
            logger.error(f"Batch save failed for facts, falling back: {e}")
            for fact in facts:
                fact.save()

        if LOCAL_DEV:
            logger.debug(f"Saved {len(facts)} facts to database")

        return facts

    except Exception as e:
        logger.error(f"Error saving facts: {e}")
        traceback.print_exc()
        return []
