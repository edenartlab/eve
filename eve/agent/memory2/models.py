"""
Memory System v2 - Data Models

This module defines the core data models for the memory system:
- Fact: Atomic factual memory for RAG retrieval
- Reflection: Interpreted memory for always-in-context
- ConsolidatedMemory: Merged reflection blobs

All models follow the existing Document pattern from eve.mongo.
"""

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional

from bson import ObjectId
from pydantic import BaseModel, Field

from eve.mongo import Collection, Document


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class MemoryScope(str, Enum):
    """Valid memory scopes."""
    SESSION = "session"
    USER = "user"
    AGENT = "agent"


class FactDecisionEvent(str, Enum):
    """Fact management decision events."""
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"


# -----------------------------------------------------------------------------
# Pydantic Models for LLM Structured Output
# -----------------------------------------------------------------------------

class ExtractedFact(BaseModel):
    """Schema for facts extracted by LLM."""
    content: str
    scope: List[Literal["user", "agent"]]


class ExtractedReflection(BaseModel):
    """Schema for reflections extracted by LLM."""
    content: str


class FactExtractionResponse(BaseModel):
    """Response schema for fact extraction LLM call."""
    facts: List[ExtractedFact] = Field(default_factory=list)


class ReflectionExtractionResponse(BaseModel):
    """Response schema for reflection extraction LLM call."""
    agent_reflections: List[ExtractedReflection] = Field(default_factory=list)
    user_reflections: List[ExtractedReflection] = Field(default_factory=list)
    session_reflections: List[ExtractedReflection] = Field(default_factory=list)


class FactDecision(BaseModel):
    """Schema for fact management decisions."""
    new_fact: str
    event: FactDecisionEvent
    existing_id: Optional[str] = None
    existing_text: Optional[str] = None
    final_text: Optional[str] = None
    reasoning: str


class FactDecisionResponse(BaseModel):
    """Response schema for fact management LLM call."""
    decisions: List[FactDecision] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# MongoDB Document Models
# -----------------------------------------------------------------------------

@Collection("memory2_facts")
class Fact(Document):
    """
    Atomic factual memory stored in vector DB for RAG retrieval.

    Facts are objective statements that stand alone without context.
    They have NO session scope - only user and/or agent scope.
    Session-level context is handled entirely by session reflections.

    Example facts:
    - "Xander's birthday is March 15th"
    - "Gene loves hockey"
    - "Project deadline is January 30th"
    """

    # Content
    content: str
    hash: str = ""  # MD5 hash for deduplication

    # Embedding (for vector search)
    embedding: List[float] = Field(default_factory=list)
    embedding_model: str = "text-embedding-3-small"

    # Scope (NO session scope - only user/agent)
    scope: List[Literal["user", "agent"]]
    agent_id: ObjectId
    user_id: Optional[ObjectId] = None  # Required if "user" in scope

    # Temporal
    formed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Source (provenance - which session created this fact)
    source_session_id: Optional[ObjectId] = None
    source_message_ids: List[ObjectId] = Field(default_factory=list)

    # Access tracking (for future importance scoring)
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None

    # Update tracking
    version: int = 1
    previous_content: Optional[str] = None
    updated_at: Optional[datetime] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-generate hash if not provided
        if not self.hash and self.content:
            self.hash = self._compute_hash(self.content)

    @staticmethod
    def _compute_hash(content: str) -> str:
        """Compute MD5 hash for content deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    def update_content(self, new_content: str) -> None:
        """Update fact content, preserving version history."""
        self.previous_content = self.content
        self.content = new_content
        self.hash = self._compute_hash(new_content)
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)

    def record_access(self) -> None:
        """Record that this fact was accessed via RAG."""
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)

    @classmethod
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data for MongoDB storage."""
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data from MongoDB."""
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist for optimal query performance."""
        collection = cls.get_collection()

        # Deduplication index
        collection.create_index(
            [("hash", 1), ("agent_id", 1)],
            name="fact_dedup_idx",
            background=True,
        )

        # Scope filtering index
        collection.create_index(
            [("agent_id", 1), ("scope", 1)],
            name="fact_scope_idx",
            background=True,
        )

        # User-specific facts index
        collection.create_index(
            [("agent_id", 1), ("user_id", 1)],
            name="fact_user_idx",
            background=True,
        )


@Collection("memory2_reflections")
class Reflection(Document):
    """
    Interpreted memory that evolves agent persona.

    Reflections are buffered until consolidated into a blob.
    They can have session, user, or agent scope.

    Example reflections:
    - "User seems frustrated with slow responses - prioritize conciseness"
    - "We are currently debugging the login bug, tried 3 approaches"
    - "Memory system v2 is the current priority project"
    """

    # Content
    content: str

    # Scope
    scope: Literal["session", "user", "agent"]
    agent_id: ObjectId
    user_id: Optional[ObjectId] = None  # Set if scope is "user"
    session_id: Optional[ObjectId] = None  # Set if scope is "session"

    # Temporal
    formed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Source
    source_session_id: Optional[ObjectId] = None
    source_message_ids: List[ObjectId] = Field(default_factory=list)

    # Consolidation state
    absorbed: bool = False  # True after consolidated
    absorbed_at: Optional[datetime] = None
    consolidated_into: Optional[ObjectId] = None  # FK to ConsolidatedMemory

    @classmethod
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data for MongoDB storage."""
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data from MongoDB."""
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist for optimal query performance."""
        collection = cls.get_collection()

        # Agent-level unabsorbed reflections
        collection.create_index(
            [("agent_id", 1), ("scope", 1), ("absorbed", 1)],
            name="reflection_unabsorbed_idx",
            background=True,
        )

        # User-specific reflections
        collection.create_index(
            [("agent_id", 1), ("user_id", 1), ("scope", 1), ("absorbed", 1)],
            name="reflection_user_idx",
            background=True,
        )

        # Session-specific reflections
        collection.create_index(
            [("session_id", 1), ("scope", 1), ("absorbed", 1)],
            name="reflection_session_idx",
            background=True,
        )


@Collection("memory2_consolidated")
class ConsolidatedMemory(Document):
    """
    Merged reflection blob for a specific scope.

    These are always injected into agent context.
    Each scope_key combination has exactly one consolidated blob.

    Examples:
    - Agent-level: Overall persona, ongoing projects, domain knowledge
    - User-level: User preferences, interaction style for specific user
    - Session-level: Rolling summary of current session events
    """

    # Scope key (unique per combination)
    scope_type: Literal["agent", "user", "session"]
    agent_id: ObjectId
    user_id: Optional[ObjectId] = None  # For user-level consolidation
    session_id: Optional[ObjectId] = None  # For session-level consolidation

    # Content
    consolidated_content: str = ""
    word_limit: int = 400  # Max words for this blob

    # Buffer tracking
    unabsorbed_ids: List[ObjectId] = Field(default_factory=list)

    # Temporal
    last_consolidated_at: Optional[datetime] = None

    @classmethod
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data for MongoDB storage."""
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        """Convert data from MongoDB."""
        if kwargs is None:
            kwargs = {}
        return schema

    @classmethod
    def get_or_create(
        cls,
        scope_type: Literal["agent", "user", "session"],
        agent_id: ObjectId,
        user_id: Optional[ObjectId] = None,
        session_id: Optional[ObjectId] = None,
        word_limit: int = 400,
    ) -> "ConsolidatedMemory":
        """
        Find an existing consolidated memory or create a new one.

        Args:
            scope_type: Type of scope (agent, user, session)
            agent_id: Agent ID
            user_id: User ID (required for user scope)
            session_id: Session ID (required for session scope)
            word_limit: Maximum words for consolidated blob

        Returns:
            ConsolidatedMemory instance
        """
        query = {
            "scope_type": scope_type,
            "agent_id": agent_id,
        }

        if scope_type == "user":
            if user_id is None:
                raise ValueError("user_id required for user scope")
            query["user_id"] = user_id

        if scope_type == "session":
            if session_id is None:
                raise ValueError("session_id required for session scope")
            query["session_id"] = session_id

        existing = cls.find_one(query)
        if existing:
            return existing

        # Create new
        doc = cls(
            scope_type=scope_type,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            word_limit=word_limit,
        )
        doc.save()
        return doc

    @classmethod
    def ensure_indexes(cls):
        """Ensure indexes exist for optimal query performance."""
        collection = cls.get_collection()

        # Unique index on scope combination
        collection.create_index(
            [("agent_id", 1), ("user_id", 1), ("session_id", 1), ("scope_type", 1)],
            name="consolidated_scope_idx",
            unique=True,
            background=True,
        )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_unabsorbed_reflections(
    scope: Literal["session", "user", "agent"],
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    session_id: Optional[ObjectId] = None,
    limit: int = 100,
) -> List[Reflection]:
    """
    Get unabsorbed reflections for a specific scope.

    Args:
        scope: Scope type
        agent_id: Agent ID
        user_id: User ID (for user scope)
        session_id: Session ID (for session scope)
        limit: Maximum number of reflections to return

    Returns:
        List of unabsorbed Reflection documents
    """
    query = {
        "agent_id": agent_id,
        "scope": scope,
        "absorbed": False,
    }

    if scope == "user" and user_id:
        query["user_id"] = user_id
    elif scope == "session" and session_id:
        query["session_id"] = session_id

    return Reflection.find(query, sort="createdAt", limit=limit)


def mark_reflections_absorbed(
    reflection_ids: List[ObjectId],
    consolidated_id: ObjectId,
) -> None:
    """
    Mark reflections as absorbed after consolidation.

    Args:
        reflection_ids: List of reflection IDs to mark
        consolidated_id: ID of the consolidated memory they were merged into
    """
    if not reflection_ids:
        return

    collection = Reflection.get_collection()
    collection.update_many(
        {"_id": {"$in": reflection_ids}},
        {
            "$set": {
                "absorbed": True,
                "absorbed_at": datetime.now(timezone.utc),
                "consolidated_into": consolidated_id,
            }
        },
    )
