from enum import Enum
import os
from typing import List, Optional
import aiohttp
from ably import AblyRest
from abc import ABC, abstractmethod

from bson import ObjectId
from pydantic import BaseModel

from eve.agent.agent import Agent
from eve.mongo import Collection, Document
from eve.api.errors import APIError

db = os.getenv("DB", "STAGE").upper()

















