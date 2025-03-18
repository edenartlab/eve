from bson import ObjectId
from typing import Dict, Any, Optional

from .mongo import Document, Collection


@Collection("models3")
class Model(Document):
    name: str
    user: ObjectId
    agent: Optional[ObjectId] = None
    task: ObjectId
    thumbnail: str
    public: bool = False
    deleted: bool = False
    args: Dict[str, Any]
    checkpoint: str
    base_model: str
    lora_trigger_text: Optional[str] = None
    lora_model: Optional[str] = None

    def __init__(self, **data):
        if isinstance(data.get("user"), str):
            data["user"] = ObjectId(data["user"])
        if isinstance(data.get("agent"), str):
            data["agent"] = ObjectId(data["agent"])
        if isinstance(data.get("task"), str):
            data["task"] = ObjectId(data["task"])
        super().__init__(**data)

