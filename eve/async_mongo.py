import os
from typing import Dict, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.server_api import ServerApi

_async_client: Optional[AsyncIOMotorClient] = None
_async_collections: Dict[str, AsyncIOMotorCollection] = {}


def get_async_mongo_client() -> AsyncIOMotorClient:
    """Get a Motor client with connection pooling."""
    global _async_client
    mongo_uri = os.getenv("MONGO_URI")
    if _async_client is None:
        _async_client = AsyncIOMotorClient(
            mongo_uri,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=60000,
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=3000,
            retryWrites=True,
            server_api=ServerApi("1"),
        )
    return _async_client


def get_async_collection(collection_name: str) -> AsyncIOMotorCollection:
    """Get a Motor collection with connection pooling."""
    db = os.getenv("DB", "STAGE")
    cache_key = f"{db}:{collection_name}"
    if cache_key in _async_collections:
        return _async_collections[cache_key]

    mongo_db_name = os.getenv("MONGO_DB_NAME")
    client = get_async_mongo_client()
    _async_collections[cache_key] = client[mongo_db_name][collection_name]
    return _async_collections[cache_key]
