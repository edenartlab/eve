from enum import Enum
from bson import ObjectId
from typing import Dict, Any, Optional, List, Literal

# from .mongo import MongoModel, get_collection
from .app.database.mongo import Document, Collection, get_collection


@Collection("models3")
class Model(Document):
    name: str
    user: ObjectId
    requester: ObjectId
    task: ObjectId
    # slug: str = None
    thumbnail: str
    public: bool = False
    deleted: bool = False
    args: Dict[str, Any]
    checkpoint: str
    base_model: str
    # users: SkipJsonSchema[Optional[Collection]] = Field(None, exclude=True)

    # def __init__(self, env, **data):
    #     super().__init__(collection_name="models", env=env, **data)
    #     # self.users = get_collection("users", env=env)
    #     self._make_slug()

    def __init__(self, **data):
        if isinstance(data.get("user"), str):
            data["user"] = ObjectId(data["user"])
        if isinstance(data.get("requester"), str):
            data["requester"] = ObjectId(data["requester"])
        if isinstance(data.get("task"), str):
            data["task"] = ObjectId(data["task"])
        super().__init__(**data)

    # @classmethod
    # def from_id(self, document_id: str, env: str):
    #     # self.users = get_collection("users", env=env)
    #     return super().from_id(self, document_id, "models", env)

    # @classmethod
    # def get_collection_name(cls) -> str:
    #     return "models"

    def _make_slug(self):
        # if self.collection is None:
        #     return
        if self.slug:
            # slug already assigned
            return
        name = self.name.lower().replace(" ", "-")
        # collection = get_collection(self.get_collection_name(), env=self.env)
        existing_docs = list(
            self.get_collection().find({"name": self.name, "user": self.user})
        )
        versions = [
            int(doc.get("slug", "").split("/")[-1][1:])
            for doc in existing_docs
            if doc.get("slug")
        ]
        new_version = max(versions or [0]) + 1
        users = get_collection("users3", db=self.db)
        username = users.find_one({"_id": self.user})["username"]
        # username = self.users.find_one({"_id": self.user})["username"]
        self.slug = f"{username}/{name}/v{new_version}"

    # def save(self, upsert_filter=None):
    #     # self._make_slug()
    #     super().save(upsert_filter)

    # def update(self, **kwargs):
    #     # self._make_slug()
    #     super().update(**kwargs)


class ClientType(Enum):
    LOCAL = "local"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    FARCASTER = "farcaster"
