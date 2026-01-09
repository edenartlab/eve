from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId
from pydantic import Field

from .mongo import Collection, Document


@Collection("data_exports")
class DataExport(Document):
    user: ObjectId
    status: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    counts: Optional[Dict[str, Any]] = None
    archive: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    download_count: int = 0
    last_downloaded_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    def __init__(self, **data):
        if isinstance(data.get("user"), str):
            data["user"] = ObjectId(data["user"])
        super().__init__(**data)
