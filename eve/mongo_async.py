import copy
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pymongo import ReturnDocument
from pymongo.server_api import ServerApi

from eve.mongo import MongoDocumentNotFound

# Global async connection pool
_async_mongo_client = None
_async_collections: Dict[str, Any] = {}


def get_async_mongo_client():
    """Get a MongoDB async client with connection pooling."""
    global _async_mongo_client
    MONGO_URI = os.getenv("MONGO_URI")
    if _async_mongo_client is None:
        _async_mongo_client = AsyncIOMotorClient(
            MONGO_URI,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=60000,
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=3000,
            retryWrites=True,
            server_api=ServerApi("1"),
        )
    return _async_mongo_client


def get_async_collection(collection_name: str):
    """Get a MongoDB async collection with connection pooling."""
    db = os.getenv("DB", "STAGE")
    cache_key = f"{db}:{collection_name}"
    if cache_key in _async_collections:
        return _async_collections[cache_key]

    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    mongo_client = get_async_mongo_client()
    _async_collections[cache_key] = mongo_client[MONGO_DB_NAME][collection_name]
    return _async_collections[cache_key]


def _resolve_collection_name(obj_or_cls) -> str:
    return getattr(obj_or_cls, "collection_name", obj_or_cls.__name__.lower())


def AsyncCollection(name):
    def wrapper(cls):
        cls.collection_name = name
        cls.find_async = classmethod(async_find)
        cls.find_one_async = classmethod(async_find_one)
        return cls

    return wrapper


async def async_find(
    cls, query: Optional[Dict[str, Any]] = None, sort=None, desc=False, limit=None
):
    """Find all documents matching the query asynchronously."""
    collection = get_async_collection(_resolve_collection_name(cls))
    docs = collection.find(query or {})
    if sort:
        docs = docs.sort(sort, -1 if desc else 1)
    if limit:
        docs = docs.limit(limit)
    results = []
    async for doc in docs:
        sub_cls = cls.get_sub_class(doc, from_yaml=False)
        converted = sub_cls.convert_from_mongo(doc)
        results.append(sub_cls.from_schema(converted, from_yaml=False))
        if limit and len(results) >= limit:
            break
    return results


async def async_find_one(cls, query: Dict[str, Any]):
    """Find one document matching the query asynchronously."""
    collection = get_async_collection(_resolve_collection_name(cls))
    doc = await collection.find_one(query)
    if not doc:
        return None
    sub_cls = cls.get_sub_class(doc, from_yaml=False)
    converted = sub_cls.convert_from_mongo(doc)
    return sub_cls.from_schema(converted, from_yaml=False)


async def async_from_mongo(cls, document_id: ObjectId):
    """Load the document from the database and return an instance of the model."""
    document_id = (
        document_id if isinstance(document_id, ObjectId) else ObjectId(document_id)
    )
    schema = await get_async_collection(_resolve_collection_name(cls)).find_one(
        {"_id": document_id}
    )
    if not schema:
        db = os.getenv("DB", "STAGE")
        raise ValueError(
            f"Document {document_id} not found in {cls.collection_name}:{db}"
        )
    sub_cls = cls.get_sub_class(schema, from_yaml=False)
    schema = sub_cls.convert_from_mongo(schema)
    return cls.from_schema(schema, from_yaml=False)


async def async_load(cls, **kwargs):
    """Load the document from the database and return an instance of the model."""
    schema = await get_async_collection(_resolve_collection_name(cls)).find_one(kwargs)
    if not schema:
        raise MongoDocumentNotFound(cls.collection_name, **kwargs)
    sub_cls = cls.get_sub_class(schema, from_yaml=False)
    schema = sub_cls.convert_from_mongo(schema)
    return cls.from_schema(schema, from_yaml=False)


async def async_save(document, upsert_filter=None, **kwargs):
    """Save the current state of the model to the database asynchronously."""
    document.updatedAt = datetime.now(timezone.utc)
    schema = document.model_dump(by_alias=True)
    document.model_validate(schema)
    schema = document.convert_to_mongo(schema)
    schema.update(kwargs)

    filter = upsert_filter or {"_id": document.id or ObjectId()}
    collection = get_async_collection(_resolve_collection_name(document.__class__))

    if document.id or filter:
        if filter:
            schema.pop("_id", None)
        created_at = schema.pop("createdAt", None)
        result = await collection.find_one_and_update(
            filter,
            {
                "$set": schema,
                "$setOnInsert": {"createdAt": created_at},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if result and "_id" in result:
            document.id = result["_id"]
        elif "_id" in filter:
            document.id = filter["_id"]
    else:
        schema["_id"] = ObjectId()
        document.createdAt = datetime.now(timezone.utc)
        await collection.insert_one(schema)
        document.id = schema["_id"]


async def async_insert(document, **kwargs):
    """Insert a new document without an upsert round-trip."""
    now = datetime.now(timezone.utc)
    if not getattr(document, "id", None):
        document.id = ObjectId()
    if not getattr(document, "createdAt", None):
        document.createdAt = now
    document.updatedAt = now

    schema = document.model_dump(by_alias=True)
    document.model_validate(schema)
    schema = document.convert_to_mongo(schema)
    schema.update(kwargs)

    collection = get_async_collection(_resolve_collection_name(document.__class__))
    await collection.insert_one(schema)
    return document


async def async_save_many(cls, documents: List[BaseModel]):
    collection = get_async_collection(_resolve_collection_name(cls))
    payloads = []
    for d in range(len(documents)):
        documents[d].id = documents[d].id or ObjectId()
        payload = documents[d].model_dump(by_alias=True)
        cls.model_validate(payload)
        payload = cls.convert_to_mongo(payload)
        payload["createdAt"] = payload.get("createdAt", datetime.now(timezone.utc))
        payload["updatedAt"] = payload.get("updatedAt", datetime.now(timezone.utc))
        payloads.append(payload)
    if payloads:
        await collection.insert_many(payloads)


async def async_bulk_write(collection_name: str, operations: List[Any]):
    if not operations:
        return None
    collection = get_async_collection(collection_name)
    return await collection.bulk_write(operations, ordered=False)


async def async_update(document, **kwargs):
    """Perform granular updates on specific fields."""
    collection = get_async_collection(_resolve_collection_name(document.__class__))
    update_result = await collection.update_one(
        {"_id": document.id}, {"$set": kwargs, "$currentDate": {"updatedAt": True}}
    )
    if update_result.modified_count > 0:
        for key, value in kwargs.items():
            setattr(document, key, value)


async def async_set_against_filter(
    document, updates: Optional[Dict] = None, filter: Optional[Dict] = None
):
    """Perform granular updates on specific fields, given an optional filter."""
    collection = get_async_collection(_resolve_collection_name(document.__class__))
    update_result = await collection.update_one(
        {"_id": document.id, **(filter or {})},
        {"$set": updates or {}, "$currentDate": {"updatedAt": True}},
    )
    if update_result.modified_count > 0:
        document.updatedAt = datetime.now(timezone.utc)


async def async_push(
    document,
    pushes: Dict[str, Union[Any, List[Any]]] = {},
    pulls: Dict[str, Any] = {},
):
    """Push or pull values granularly to array fields in document."""
    push_ops, pull_ops = {}, {}
    for field_name, value in pushes.items():
        values_to_push = value if isinstance(value, list) else [value]

        values_original = [copy.deepcopy(v) for v in values_to_push]
        values_to_push = [
            v.model_dump() if isinstance(v, BaseModel) else v for v in values_to_push
        ]

        updated_data = copy.deepcopy(document)
        if hasattr(updated_data, field_name) and isinstance(
            getattr(updated_data, field_name), list
        ):
            getattr(updated_data, field_name).extend(values_original)

        push_ops[field_name] = {"$each": values_to_push}

        if hasattr(document, field_name) and isinstance(
            getattr(document, field_name), list
        ):
            setattr(
                document, field_name, getattr(document, field_name) + values_original
            )

    for field_name, value in pulls.items():
        pull_ops[field_name] = value

        if hasattr(document, field_name) and isinstance(
            getattr(document, field_name), list
        ):
            current_list = getattr(document, field_name)
            setattr(document, field_name, [x for x in current_list if x != value])

    collection = get_async_collection(_resolve_collection_name(document.__class__))
    update_ops = {"$currentDate": {"updatedAt": True}}
    if push_ops:
        update_ops["$push"] = push_ops
    if pull_ops:
        update_ops["$pull"] = pull_ops

    update_result = await collection.update_one({"_id": document.id}, update_ops)
    if update_result.modified_count > 0:
        document.updatedAt = datetime.now(timezone.utc)


async def async_update_nested_field(
    document, field_name: str, index: int, sub_field: str, value
):
    """Update a specific field within an array of dictionaries."""
    updated_data = document.model_copy()
    if hasattr(updated_data, field_name) and isinstance(
        getattr(updated_data, field_name), list
    ):
        field_list = getattr(updated_data, field_name)
        if len(field_list) > index and isinstance(field_list[index], dict):
            field_list[index][sub_field] = value
        else:
            raise ValidationError(
                f"Field '{field_name}[{index}]' is not a valid dictionary field."
            )
    else:
        raise ValidationError(f"Field '{field_name}' is not a valid list field.")

    collection = get_async_collection(_resolve_collection_name(document.__class__))
    update_result = await collection.update_one(
        {"_id": document.id},
        {
            "$set": {f"{field_name}.{index}.{sub_field}": value},
            "$currentDate": {"updatedAt": True},
        },
    )
    if update_result.modified_count > 0:
        if hasattr(document, field_name) and isinstance(
            getattr(document, field_name), list
        ):
            field_list = getattr(document, field_name)
            if len(field_list) > index and isinstance(field_list[index], dict):
                field_list[index][sub_field] = value
                document.updatedAt = datetime.now(timezone.utc)


async def async_reload(document):
    """Reload the current document from the database."""
    updated_instance = await async_from_mongo(document.__class__, document.id)
    if updated_instance:
        for key, value in updated_instance.model_dump().items():
            setattr(document, key, value)


async def async_delete(document):
    """Delete the document from the database."""
    collection = get_async_collection(_resolve_collection_name(document.__class__))
    await collection.delete_one({"_id": document.id})


class AsyncDocument(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    createdAt: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updatedAt: Optional[datetime] = None

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
