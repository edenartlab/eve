"""
Artifacts: Persistent, structured world-state objects for Eve agents.

Artifacts are canonical state objects that agents and sessions can read, mutate,
and rely on over long time scales without that state aging out of the chat history.

Examples of artifacts:
- Screenplay storyboards (scenes, images, narration)
- Character bibles
- Game level definitions
- Design documents
- Project plans
"""

import copy
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Type, Union

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eve.mongo import Collection, Document

logger = logging.getLogger(__name__)


# =============================================================================
# Type Schemas for Artifact Validation
# =============================================================================


class SceneSchema(BaseModel):
    """Schema for a scene in a screenplay storyboard."""

    image_prompt: str
    description: str
    order: int
    duration: Optional[float] = None
    audio_prompt: Optional[str] = None
    narration: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ScreenplayStoryboardSchema(BaseModel):
    """Schema for screenplay_storyboard artifact type."""

    title: str
    audio_prompt: Optional[str] = None
    narration: Optional[str] = None
    scenes: List[SceneSchema] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class RelationshipSchema(BaseModel):
    """Schema for character relationships."""

    character: str
    relationship: str

    model_config = ConfigDict(extra="allow")


class CharacterBibleSchema(BaseModel):
    """Schema for character_bible artifact type."""

    name: str
    traits: List[str] = Field(default_factory=list)
    backstory: Optional[str] = None
    appearance: Optional[str] = None
    relationships: List[RelationshipSchema] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class MilestoneSchema(BaseModel):
    """Schema for project milestones."""

    name: str
    status: str = "pending"
    due_date: Optional[str] = None
    tasks: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class ProjectPlanSchema(BaseModel):
    """Schema for project_plan artifact type."""

    title: str
    description: Optional[str] = None
    goals: List[str] = Field(default_factory=list)
    milestones: List[MilestoneSchema] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class LocationSchema(BaseModel):
    """Schema for world locations."""

    name: str
    description: str

    model_config = ConfigDict(extra="allow")


class WorldBibleSchema(BaseModel):
    """Schema for world_bible artifact type."""

    name: str
    description: Optional[str] = None
    locations: List[LocationSchema] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    history: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class EnemySchema(BaseModel):
    """Schema for game enemies."""

    type: str
    count: int = 1
    position: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class GameLevelSchema(BaseModel):
    """Schema for game_level artifact type."""

    name: str
    description: Optional[str] = None
    difficulty: str = "normal"
    objectives: List[str] = Field(default_factory=list)
    enemies: List[EnemySchema] = Field(default_factory=list)
    rewards: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


# Registry of artifact type schemas
ARTIFACT_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "screenplay_storyboard": ScreenplayStoryboardSchema,
    "character_bible": CharacterBibleSchema,
    "project_plan": ProjectPlanSchema,
    "world_bible": WorldBibleSchema,
    "game_level": GameLevelSchema,
}


class ArtifactValidationError(Exception):
    """Raised when artifact data fails validation."""

    def __init__(self, artifact_type: str, errors: List[Dict[str, Any]]):
        self.artifact_type = artifact_type
        self.errors = errors
        super().__init__(f"Validation failed for artifact type '{artifact_type}': {errors}")


def validate_artifact_data(
    artifact_type: str, data: Dict[str, Any], raise_on_error: bool = False
) -> tuple[bool, Optional[List[Dict[str, Any]]]]:
    """
    Validate artifact data against its type schema.

    Args:
        artifact_type: The type of artifact
        data: The data to validate
        raise_on_error: If True, raise ArtifactValidationError on failure

    Returns:
        Tuple of (is_valid, errors). Errors is None if valid.
    """
    schema = ARTIFACT_SCHEMAS.get(artifact_type)
    if not schema:
        # Unknown type - allow anything
        return True, None

    try:
        schema.model_validate(data)
        return True, None
    except ValidationError as e:
        errors = e.errors()
        if raise_on_error:
            raise ArtifactValidationError(artifact_type, errors)
        return False, errors


# =============================================================================
# Artifact Operation and Version Models
# =============================================================================


class ArtifactOperation(BaseModel):
    """A single operation to apply to an artifact's data."""

    op: Literal["set", "append", "insert", "remove", "update", "replace"]
    path: Optional[str] = None  # Dot-notation path, e.g., "scenes.0.title"
    index: Optional[int] = None  # For insert/remove/update operations on arrays
    value: Optional[Any] = None  # The value to set/append/insert/update
    data: Optional[Dict[str, Any]] = None  # For replace operation (full data)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ArtifactVersion(BaseModel):
    """A snapshot of an artifact at a point in time."""

    version: int  # Version number (1, 2, 3, ...)
    timestamp: datetime  # When this version was created
    data_snapshot: Dict[str, Any]  # Full data at this version
    operations: List[Dict[str, Any]] = []  # Operations that created this version
    actor_type: Literal["user", "agent", "system"] = "user"
    actor_id: Optional[ObjectId] = None  # Who made this change
    message: Optional[str] = None  # Optional commit message

    model_config = ConfigDict(arbitrary_types_allowed=True)


# =============================================================================
# Main Artifact Document
# =============================================================================


# Size thresholds for context injection (in characters of JSON)
SMALL_ARTIFACT_THRESHOLD = 2000  # Include full data in context
MEDIUM_ARTIFACT_THRESHOLD = 8000  # Include partial data in context


@Collection("artifacts")
class Artifact(Document):
    """
    A persistent, structured world-state object.

    Artifacts store canonical state that agents can read and mutate through
    structured operations. They support versioning for history tracking.
    """

    # Identity & Type
    type: str  # Artifact type: "screenplay_storyboard", "character_bible", etc.
    name: str  # Human-readable name
    description: Optional[str] = None  # Optional description
    schema_version: str = "1.0"  # For future migrations

    # Content - the actual structured data
    data: Dict[str, Any] = Field(default_factory=dict)

    # Ownership & Access Control
    owner: ObjectId  # User who owns this artifact
    session: Optional[ObjectId] = None  # Primary session link (optional)
    sessions: List[ObjectId] = Field(default_factory=list)  # All sessions with access
    agents: List[ObjectId] = Field(default_factory=list)  # Agents with special access

    # Versioning
    version: int = 1  # Current version number
    versions: List[ArtifactVersion] = Field(
        default_factory=list
    )  # Version history (recent versions)
    max_versions: int = 50  # Max versions to keep in history

    # Lifecycle
    archived: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        # Convert string IDs to ObjectId
        if isinstance(data.get("owner"), str):
            data["owner"] = ObjectId(data["owner"])
        if isinstance(data.get("session"), str):
            data["session"] = ObjectId(data["session"])
        if "sessions" in data:
            data["sessions"] = [
                ObjectId(s) if isinstance(s, str) else s for s in data["sessions"]
            ]
        if "agents" in data:
            data["agents"] = [
                ObjectId(a) if isinstance(a, str) else a for a in data["agents"]
            ]
        super().__init__(**data)

    def validate_data(self, raise_on_error: bool = False) -> tuple[bool, Optional[List[Dict[str, Any]]]]:
        """
        Validate this artifact's data against its type schema.

        Args:
            raise_on_error: If True, raise ArtifactValidationError on failure

        Returns:
            Tuple of (is_valid, errors)
        """
        return validate_artifact_data(self.type, self.data, raise_on_error)

    def get_data_size(self) -> int:
        """Get the approximate size of the data in characters (JSON serialized)."""
        try:
            return len(json.dumps(self.data, default=str))
        except Exception:
            return 0

    def is_small(self) -> bool:
        """Check if this artifact is small enough to include full data in context."""
        return self.get_data_size() <= SMALL_ARTIFACT_THRESHOLD

    def is_medium(self) -> bool:
        """Check if this artifact is medium-sized (partial data in context)."""
        size = self.get_data_size()
        return SMALL_ARTIFACT_THRESHOLD < size <= MEDIUM_ARTIFACT_THRESHOLD

    def _get_nested_value(self, path: str) -> Any:
        """Get a value from nested data using dot notation."""
        keys = path.split(".")
        value = self.data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                idx = int(key)
                value = value[idx] if 0 <= idx < len(value) else None
            else:
                return None
        return value

    def _set_nested_value(self, path: str, value: Any) -> None:
        """Set a value in nested data using dot notation."""
        keys = path.split(".")
        target = self.data
        for key in keys[:-1]:
            if isinstance(target, dict):
                if key not in target:
                    # Create intermediate dict or list based on next key
                    next_key = keys[keys.index(key) + 1]
                    target[key] = [] if next_key.isdigit() else {}
                target = target[key]
            elif isinstance(target, list) and key.isdigit():
                target = target[int(key)]

        final_key = keys[-1]
        if isinstance(target, dict):
            target[final_key] = value
        elif isinstance(target, list) and final_key.isdigit():
            idx = int(final_key)
            if 0 <= idx < len(target):
                target[idx] = value

    def _delete_nested_value(self, path: str, index: Optional[int] = None) -> None:
        """Delete a value from nested data using dot notation."""
        keys = path.split(".")
        target = self.data
        for key in keys[:-1]:
            if isinstance(target, dict):
                target = target.get(key)
            elif isinstance(target, list) and key.isdigit():
                target = target[int(key)]
            if target is None:
                return

        final_key = keys[-1]
        if isinstance(target, dict) and final_key in target:
            if isinstance(target[final_key], list) and index is not None:
                if 0 <= index < len(target[final_key]):
                    target[final_key].pop(index)
            else:
                del target[final_key]
        elif isinstance(target, list) and final_key.isdigit():
            idx = int(final_key)
            if 0 <= idx < len(target):
                target.pop(idx)

    def apply_operation(self, operation: ArtifactOperation) -> bool:
        """
        Apply a single operation to the artifact's data.

        Returns True if the operation was applied successfully.
        """
        op = operation.op

        if op == "replace":
            # Full replacement of data
            if operation.data is not None:
                self.data = copy.deepcopy(operation.data)
                return True
            return False

        if op == "set":
            # Set a value at a path
            if operation.path and operation.value is not None:
                self._set_nested_value(operation.path, copy.deepcopy(operation.value))
                return True
            return False

        if op == "append":
            # Append to an array
            if operation.path:
                target = self._get_nested_value(operation.path)
                if isinstance(target, list):
                    target.append(copy.deepcopy(operation.value))
                    return True
                elif target is None:
                    # Create the array if it doesn't exist
                    self._set_nested_value(
                        operation.path, [copy.deepcopy(operation.value)]
                    )
                    return True
            return False

        if op == "insert":
            # Insert at a specific index in an array
            if operation.path and operation.index is not None:
                target = self._get_nested_value(operation.path)
                if isinstance(target, list):
                    target.insert(operation.index, copy.deepcopy(operation.value))
                    return True
            return False

        if op == "remove":
            # Remove from an array by index
            if operation.path and operation.index is not None:
                target = self._get_nested_value(operation.path)
                if isinstance(target, list) and 0 <= operation.index < len(target):
                    target.pop(operation.index)
                    return True
            return False

        if op == "update":
            # Update an item in an array by index
            if operation.path and operation.index is not None:
                target = self._get_nested_value(operation.path)
                if isinstance(target, list) and 0 <= operation.index < len(target):
                    if isinstance(target[operation.index], dict) and isinstance(
                        operation.value, dict
                    ):
                        # Merge update for dicts
                        target[operation.index].update(copy.deepcopy(operation.value))
                    else:
                        # Replace for non-dicts
                        target[operation.index] = copy.deepcopy(operation.value)
                    return True
            return False

        return False

    def apply_operations(
        self,
        operations: List[Union[ArtifactOperation, Dict[str, Any]]],
        actor_type: Literal["user", "agent", "system"] = "user",
        actor_id: Optional[ObjectId] = None,
        message: Optional[str] = None,
        save: bool = True,
        validate: bool = True,
    ) -> "Artifact":
        """
        Apply multiple operations to the artifact and optionally save.

        Creates a new version entry in the history.

        Args:
            operations: List of operations to apply
            actor_type: Who is making this change
            actor_id: ObjectId of the actor (user or agent)
            message: Optional commit message
            save: Whether to save to database after applying
            validate: Whether to validate data after applying operations

        Returns:
            self for chaining
        """
        # Normalize operations to ArtifactOperation objects
        normalized_ops = []
        for op in operations:
            if isinstance(op, dict):
                normalized_ops.append(ArtifactOperation(**op))
            else:
                normalized_ops.append(op)

        # Apply each operation
        applied_ops = []
        for op in normalized_ops:
            if self.apply_operation(op):
                applied_ops.append(op.model_dump())

        if applied_ops:
            # Validate data after operations if requested
            if validate:
                is_valid, errors = self.validate_data()
                if not is_valid:
                    logger.warning(
                        f"Artifact {self.id} data validation failed after operations: {errors}"
                    )

            # Create version entry
            new_version = ArtifactVersion(
                version=self.version + 1,
                timestamp=datetime.now(timezone.utc),
                data_snapshot=copy.deepcopy(self.data),
                operations=applied_ops,
                actor_type=actor_type,
                actor_id=actor_id,
                message=message,
            )

            # Add to versions, keeping only recent ones
            self.versions.append(new_version)
            if len(self.versions) > self.max_versions:
                self.versions = self.versions[-self.max_versions :]

            self.version = new_version.version

            if save:
                self.save()

        return self

    def get_summary(self, max_length: int = 200) -> str:
        """
        Generate a brief summary of the artifact for context injection.

        Returns a human-readable summary suitable for LLM context.
        """
        summary_parts = []

        # Add basic info
        summary_parts.append(f"**{self.name}** (type: {self.type})")

        if self.description:
            desc = (
                self.description[:100] + "..."
                if len(self.description) > 100
                else self.description
            )
            summary_parts.append(f"Description: {desc}")

        # Add data summary based on type
        if self.data:
            # Count top-level keys
            keys = list(self.data.keys())
            if keys:
                # Show array lengths for common patterns
                data_info = []
                for key in keys[:5]:  # Limit to first 5 keys
                    val = self.data[key]
                    if isinstance(val, list):
                        data_info.append(f"{key}: {len(val)} items")
                    elif isinstance(val, str) and len(val) > 50:
                        data_info.append(f"{key}: '{val[:30]}...'")
                    elif isinstance(val, (str, int, float, bool)):
                        data_info.append(f"{key}: {val}")
                if data_info:
                    summary_parts.append("Data: " + ", ".join(data_info))

        summary_parts.append(f"Version: {self.version}")

        summary = "\n".join(summary_parts)
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    def to_context_dict(self, include_data: bool = False) -> Dict[str, Any]:
        """
        Convert artifact to a dict suitable for LLM context.

        Args:
            include_data: Whether to include the full data object

        Returns:
            Dict with artifact info for context injection
        """
        result = {
            "artifact_id": str(self.id),
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "updated_at": self.updatedAt.isoformat() if self.updatedAt else None,
        }

        if include_data:
            result["data"] = self.data

        return result

    def get_context_data(self) -> Optional[Dict[str, Any]]:
        """
        Get data appropriate for context injection based on size.

        Returns:
            Full data for small artifacts, truncated data for medium, None for large.
        """
        if self.is_small():
            return self.data
        elif self.is_medium():
            # Return truncated version - top-level fields only, truncate arrays
            truncated = {}
            for key, val in self.data.items():
                if isinstance(val, list):
                    truncated[key] = val[:3] if len(val) > 3 else val
                    if len(val) > 3:
                        truncated[f"_{key}_total"] = len(val)
                elif isinstance(val, str) and len(val) > 200:
                    truncated[key] = val[:200] + "..."
                else:
                    truncated[key] = val
            return truncated
        return None

    def link_to_session(self, session_id: ObjectId) -> None:
        """Link this artifact to a session."""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)

        if session_id not in self.sessions:
            self.push(pushes={"sessions": session_id})

    def unlink_from_session(self, session_id: ObjectId) -> None:
        """Unlink this artifact from a session."""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)

        if session_id in self.sessions:
            self.push(pulls={"sessions": session_id})

    @classmethod
    def find_for_session(
        cls, session_id: ObjectId, include_archived: bool = False
    ) -> List["Artifact"]:
        """Find all artifacts linked to a session."""
        if isinstance(session_id, str):
            session_id = ObjectId(session_id)

        query = {"$or": [{"session": session_id}, {"sessions": session_id}]}
        if not include_archived:
            query["archived"] = {"$ne": True}

        return cls.find(query, sort="updatedAt", desc=True)

    @classmethod
    def find_for_user(
        cls,
        user_id: ObjectId,
        artifact_type: Optional[str] = None,
        include_archived: bool = False,
        limit: int = 50,
    ) -> List["Artifact"]:
        """Find all artifacts owned by a user."""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)

        query = {"owner": user_id}
        if artifact_type:
            query["type"] = artifact_type
        if not include_archived:
            query["archived"] = {"$ne": True}

        return cls.find(query, sort="updatedAt", desc=True, limit=limit)

    def archive(self) -> None:
        """Archive this artifact (soft delete)."""
        self.update(archived=True)

    def restore(self) -> None:
        """Restore an archived artifact."""
        self.update(archived=False)

    def rollback_to_version(self, target_version: int) -> bool:
        """
        Rollback the artifact to a previous version.

        Args:
            target_version: The version number to rollback to

        Returns:
            True if rollback was successful
        """
        # Find the version in history
        for v in self.versions:
            if v.version == target_version:
                # Create a new version that is a rollback
                self.apply_operations(
                    [{"op": "replace", "data": v.data_snapshot}],
                    actor_type="system",
                    message=f"Rollback to version {target_version}",
                )
                return True
        return False
