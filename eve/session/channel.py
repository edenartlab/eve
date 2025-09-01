import json
import os
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timezone

import magic
from bson import ObjectId
from pydantic import ConfigDict, Field, BaseModel, field_serializer

from eve.utils import download_file, image_to_base64, prepare_result, dumps_json
from eve.mongo import Collection, Document
from eve.tool import Tool



class ChannelType(str, Enum):
    EDEN = "eden"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    TWITTER = "twitter"
    FARCASTER = "farcaster"


@Collection("channels")
class Channel(Document):
    type: Literal["eden", "discord", "telegram", "twitter"]
    key: Optional[str] = None

