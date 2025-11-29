"""
Artifact - A generic container for generative assets on Eden.

Artifacts represent a hierarchical tree structure of creative works, from simple
seeds (a logline and thumbnail) to complex screenplays with multiple captioned media files.

The parent-child relationship allows seeds to mature through stages and be linked
ancestrally, forming a tree of creative evolution.
"""

from typing import List, Literal, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

from .mongo import Collection, Document


class ArtifactCreation(BaseModel):
    """A creation within an artifact, with optional caption."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    creation: ObjectId
    caption: Optional[str] = None

    def __init__(self, **data):
        if isinstance(data.get("creation"), str):
            data["creation"] = ObjectId(data["creation"])
        super().__init__(**data)


@Collection("artifacts")
class Artifact(Document):
    """
    A generic container for a set of generative assets on Eden.

    Artifacts can range from simple seeds (a logline and single thumbnail image)
    to complex screenplays with multiple captioned creations (images, videos, audio, etc.).

    Through a system of parent-to-child mappings, seeds can mature in stages and be
    linked to each other ancestrally, forming a tree structure of creative evolution.

    Attributes:
        title: The title of the artifact
        type: The type/stage of the artifact ("seed", "plant", "screenplay", etc.)
        parent: Reference to parent artifact for tree structure
        agents: List of agents that contributed to this artifact
        public: Whether this artifact is publicly visible
        description: Detailed description of the artifact
        session_id: The Eden session ID that created this artifact
        creations: Ordered list of creations with optional captions
    """

    title: str
    type: Literal["seed", "plant", "screenplay"] = "seed"
    parent: Optional[ObjectId] = None
    agents: List[ObjectId] = Field(default_factory=list)
    public: bool = True
    description: str
    session_id: ObjectId
    creations: List[ArtifactCreation] = Field(default_factory=list)

    def __init__(self, **data):
        # Convert string IDs to ObjectIds
        if isinstance(data.get("parent"), str):
            data["parent"] = ObjectId(data["parent"])
        if isinstance(data.get("session_id"), str):
            data["session_id"] = ObjectId(data["session_id"])

        # Convert agents list
        if "agents" in data:
            data["agents"] = [
                ObjectId(agent) if isinstance(agent, str) else agent
                for agent in data.get("agents", [])
            ]

        # Convert creations list
        if "creations" in data:
            converted_creations = []
            for c in data.get("creations", []):
                if isinstance(c, dict):
                    converted_creations.append(ArtifactCreation(**c))
                elif isinstance(c, ArtifactCreation):
                    converted_creations.append(c)
            data["creations"] = converted_creations

        super().__init__(**data)

    @classmethod
    def get_children(cls, parent_id: ObjectId) -> List["Artifact"]:
        """Get all direct children of an artifact."""
        parent_id = ObjectId(parent_id) if isinstance(parent_id, str) else parent_id
        return list(cls.find({"parent": parent_id}))

    @classmethod
    def get_tree(cls, root_id: ObjectId) -> List["Artifact"]:
        """Get the full tree of descendants from a root artifact."""
        root_id = ObjectId(root_id) if isinstance(root_id, str) else root_id
        descendants = []
        queue = [root_id]

        while queue:
            current_id = queue.pop(0)
            children = cls.get_children(current_id)
            descendants.extend(children)
            queue.extend([child.id for child in children])

        return descendants

    def add_creation(self, creation_id: ObjectId, caption: Optional[str] = None):
        """Add a creation to this artifact."""
        creation_id = (
            ObjectId(creation_id) if isinstance(creation_id, str) else creation_id
        )
        artifact_creation = ArtifactCreation(creation=creation_id, caption=caption)
        self.creations.append(artifact_creation)
        self.save()

    def get_ancestors(self) -> List["Artifact"]:
        """Get all ancestors of this artifact up to the root."""
        ancestors = []
        current = self

        while current.parent:
            parent = Artifact.find_one({"_id": current.parent})
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break

        return ancestors
