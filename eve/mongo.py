import os
import copy
import yaml
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timezone
from bson import ObjectId
from typing import Optional, List, Dict, Any, Union
from sentry_sdk import trace


# Global connection pool
_mongo_client = None
_collections = {}


@trace
def get_mongo_client():
    """Get a MongoDB client with connection pooling"""
    global _mongo_client
    MONGO_URI = os.getenv("MONGO_URI")
    if _mongo_client is None:
        _mongo_client = MongoClient(
            MONGO_URI,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=60000,
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=3000,
            retryWrites=True,
            server_api=ServerApi("1"),
        )
    return _mongo_client


@trace
def get_collection(collection_name: str):
    """Get a MongoDB collection with connection pooling"""
    db = os.getenv("DB")
    cache_key = f"{db}:{collection_name}"
    if cache_key in _collections:
        return _collections[cache_key]

    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    mongo_client = get_mongo_client()
    _collections[cache_key] = mongo_client[MONGO_DB_NAME][collection_name]
    return _collections[cache_key]


@trace
def Collection(name):
    @trace
    def wrapper(cls):
        cls.collection_name = name
        return cls

    return wrapper


class Document(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    createdAt: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updatedAt: Optional[datetime] = None

    model_config = ConfigDict(
        json_encoders={
            ObjectId: str,
            datetime: lambda v: v.isoformat(),
        },
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    @trace
    def get_collection(cls):
        """Override this method to provide the correct collection for the model."""
        collection_name = getattr(cls, "collection_name", cls.__name__.lower())
        return get_collection(collection_name)

    @classmethod
    @trace
    def from_schema(cls, schema: dict, from_yaml=True):
        """Load a document from a schema."""
        sub_cls = cls.get_sub_class(schema, from_yaml=from_yaml)
        result = sub_cls.model_validate(schema)
        return result

    @classmethod
    @trace
    def from_yaml(cls, file_path: str):
        """Load a document from a YAML file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        with open(file_path, "r") as file:
            schema = yaml.safe_load(file)
        sub_cls = cls.get_sub_class(schema, from_yaml=True)
        schema = sub_cls.convert_from_yaml(schema, file_path=file_path)
        return cls.from_schema(schema, from_yaml=True)

    @classmethod
    @trace
    def from_mongo(cls, document_id: ObjectId):
        """Load the document from the database and return an instance of the model."""
        document_id = (
            document_id if isinstance(document_id, ObjectId) else ObjectId(document_id)
        )
        schema = cls.get_collection().find_one({"_id": document_id})
        if not schema:
            db = os.getenv("DB")
            raise ValueError(
                f"Document {document_id} not found in {cls.collection_name}:{db}"
            )
        sub_cls = cls.get_sub_class(schema, from_yaml=False)
        schema = sub_cls.convert_from_mongo(schema)
        return cls.from_schema(schema, from_yaml=False)

    @classmethod
    @trace
    def load(cls, **kwargs):
        """Load the document from the database and return an instance of the model."""
        schema = cls.get_collection().find_one(kwargs)
        if not schema:
            raise MongoDocumentNotFound(cls.collection_name, **kwargs)
        sub_cls = cls.get_sub_class(schema, from_yaml=False)
        schema = sub_cls.convert_from_mongo(schema)
        return cls.from_schema(schema, from_yaml=False)

    @classmethod
    @trace
    def get_sub_class(cls, schema: dict = None, from_yaml=True) -> type:
        return cls

    @classmethod
    @trace
    def convert_from_mongo(cls, schema: dict, **kwargs) -> dict:
        return schema

    @classmethod
    @trace
    def convert_from_yaml(cls, schema: dict, **kwargs) -> dict:
        return schema

    @classmethod
    @trace
    def convert_to_mongo(cls, schema: dict, **kwargs) -> dict:
        return schema

    @classmethod
    @trace
    def convert_to_yaml(cls, schema: dict, **kwargs) -> dict:
        return schema

    @trace
    def save(self, upsert_filter=None, **kwargs):
        """Save the current state of the model to the database."""
        self.updatedAt = datetime.now(timezone.utc)
        schema = self.model_dump(by_alias=True)
        self.model_validate(schema)
        schema = self.convert_to_mongo(schema)
        schema.update(kwargs)

        filter = upsert_filter or {"_id": self.id or ObjectId()}
        collection = self.get_collection()

        if self.id or filter:
            if filter:
                schema.pop("_id", None)
            created_at = schema.pop("createdAt", None)
            result = collection.find_one_and_update(
                filter,
                {
                    "$set": schema,
                    "$setOnInsert": {"createdAt": created_at},
                },
                upsert=True,
                return_document=True,
            )
            self.id = result["_id"]
        else:
            schema["_id"] = ObjectId()
            self.createdAt = datetime.now(timezone.utc)
            result = collection.insert_one(schema)
            self.id = schema["_id"]

    @classmethod
    @trace
    def save_many(cls, documents: List[BaseModel]):
        collection = cls.get_collection()
        for d in range(len(documents)):
            documents[d].id = documents[d].id or ObjectId()
            documents[d] = documents[d].model_dump(by_alias=True)
            cls.model_validate(documents[d])
            documents[d] = cls.convert_to_mongo(documents[d])
            documents[d]["createdAt"] = documents[d].get(
                "createdAt", datetime.now(timezone.utc)
            )
            documents[d]["updatedAt"] = documents[d].get(
                "updatedAt", datetime.now(timezone.utc)
            )
        collection.insert_many(documents)

    @trace
    def update(self, **kwargs):
        """Perform granular updates on specific fields."""
        collection = self.get_collection()
        update_result = collection.update_one(
            {"_id": self.id}, {"$set": kwargs, "$currentDate": {"updatedAt": True}}
        )
        if update_result.modified_count > 0:
            for key, value in kwargs.items():
                setattr(self, key, value)

    @trace
    def set_against_filter(self, updates: Dict = None, filter: Optional[Dict] = None):
        """Perform granular updates on specific fields, given an optional filter."""
        collection = self.get_collection()
        update_result = collection.update_one(
            {"_id": self.id, **filter},
            {"$set": updates, "$currentDate": {"updatedAt": True}},
        )
        if update_result.modified_count > 0:
            self.updatedAt = datetime.now(timezone.utc)

    @trace
    def push(
        self, pushes: Dict[str, Union[Any, List[Any]]] = {}, pulls: Dict[str, Any] = {}
    ):
        """Push or pull values granularly to array fields in document."""
        push_ops, pull_ops = {}, {}
        for field_name, value in pushes.items():
            values_to_push = value if isinstance(value, list) else [value]

            # Convert Pydantic models to dictionaries if needed
            values_original = [copy.deepcopy(v) for v in values_to_push]
            values_to_push = [
                v.model_dump() if isinstance(v, BaseModel) else v
                for v in values_to_push
            ]

            # Create a copy of the current instance and update the array field with the new values for validation
            updated_data = copy.deepcopy(self)
            if hasattr(updated_data, field_name) and isinstance(
                getattr(updated_data, field_name), list
            ):
                getattr(updated_data, field_name).extend(values_original)

            push_ops[field_name] = {"$each": values_to_push}

            # Set field values in local instance
            if hasattr(self, field_name) and isinstance(
                getattr(self, field_name), list
            ):
                setattr(self, field_name, getattr(self, field_name) + values_original)

        # Do same thing for pulls
        for field_name, value in pulls.items():
            pull_ops[field_name] = value

            if hasattr(self, field_name) and isinstance(
                getattr(self, field_name), list
            ):
                # Remove all instances of value from the local list
                current_list = getattr(self, field_name)
                setattr(self, field_name, [x for x in current_list if x != value])

        # Update MongoDB operation to use $pull instead of $pop
        collection = self.get_collection()
        update_ops = {"$currentDate": {"updatedAt": True}}
        if push_ops:
            update_ops["$push"] = push_ops
        if pull_ops:
            update_ops["$pull"] = pull_ops

        update_result = collection.update_one({"_id": self.id}, update_ops)
        if update_result.modified_count > 0:
            self.updatedAt = datetime.now(timezone.utc)

    @trace
    def update_nested_field(self, field_name: str, index: int, sub_field: str, value):
        """Update a specific field within an array of dictionaries."""
        # Create a copy of the current instance and update the nested field for validation
        updated_data = self.model_copy()
        if hasattr(updated_data, field_name) and isinstance(
            getattr(updated_data, field_name), list
        ):
            field_list = getattr(updated_data, field_name)
            if len(field_list) > index and isinstance(field_list[index], dict):
                field_list[index][sub_field] = value
                # updated_data.validate_fields()
            else:
                raise ValidationError(
                    f"Field '{field_name}[{index}]' is not a valid dictionary field."
                )
        else:
            raise ValidationError(f"Field '{field_name}' is not a valid list field.")

        # Perform the update operation in MongoDB
        collection = self.get_collection()
        update_result = collection.update_one(
            {"_id": self.id},
            {
                "$set": {f"{field_name}.{index}.{sub_field}": value},
                "$currentDate": {"updatedAt": True},
            },
        )
        if update_result.modified_count > 0:
            # Update the value in the local instance if the update was successful
            if hasattr(self, field_name) and isinstance(
                getattr(self, field_name), list
            ):
                field_list = getattr(self, field_name)
                if len(field_list) > index and isinstance(field_list[index], dict):
                    field_list[index][sub_field] = value
                    self.updatedAt = datetime.now(timezone.utc)

    @trace
    def reload(self):
        """Reload the current document from the database."""
        updated_instance = self.from_mongo(self.id)
        if updated_instance:
            # Use model_dump to get the data while maintaining type information
            for key, value in updated_instance.model_dump().items():
                setattr(self, key, value)

    @trace
    def delete(self):
        """Delete the document from the database."""
        collection = self.get_collection()
        collection.delete_one({"_id": self.id})


@trace
def serialize_document(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: serialize_document(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_document(item) for item in obj]
    return obj


class MongoDocumentNotFound(Exception):
    """Exception raised when a document is not found in MongoDB."""

    def __init__(self, collection_name: str, document_id: str = None, **kwargs):
        db = os.getenv("DB")
        if document_id:
            self.message = f"Document with id {document_id} not found in collection {collection_name}, db: {db}"
        else:
            self.message = f"Document {kwargs} not found in {collection_name}:{db}"
        super().__init__(self.message)
