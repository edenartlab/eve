import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import pytest
from bson import ObjectId

import eve.mongo as mongo_module


class FakeCursor:
    def __init__(self, documents: Iterable[Dict[str, Any]]):
        self._docs = [deepcopy(doc) for doc in documents]

    def sort(self, field: str, direction: int):
        reverse = direction == -1
        self._docs.sort(key=lambda doc: doc.get(field), reverse=reverse)
        return self

    def limit(self, count: int):
        self._docs = self._docs[:count]
        return self

    def __iter__(self):
        for doc in self._docs:
            yield deepcopy(doc)


class FakeCollection:
    def __init__(self):
        self._docs: List[Dict[str, Any]] = []

    def _matches(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        for key, value in query.items():
            if isinstance(value, dict) and "$in" in value:
                if doc.get(key) not in value["$in"]:
                    return False
            else:
                if doc.get(key) != value:
                    return False
        return True

    def _project(self, doc: Dict[str, Any], projection: Dict[str, int] | None):
        if not projection:
            return deepcopy(doc)
        projected = {}
        include_id = projection.get("_id", 1)
        for key, include in projection.items():
            if include and key in doc and key != "_id":
                projected[key] = deepcopy(doc[key])
        if include_id and "_id" in doc:
            projected["_id"] = deepcopy(doc["_id"])
        return projected

    def find_one(self, query: Dict[str, Any], projection: Dict[str, int] | None = None):
        for doc in reversed(self._docs):
            if self._matches(doc, query):
                return self._project(doc, projection)
        return None

    def find(
        self,
        query: Dict[str, Any] | None = None,
        projection: Dict[str, int] | None = None,
    ):
        query = query or {}
        matched = [
            self._project(doc, projection)
            for doc in self._docs
            if self._matches(doc, query)
        ]
        return FakeCursor(matched)

    def insert_one(self, document: Dict[str, Any]):
        doc = deepcopy(document)
        doc.setdefault("_id", ObjectId())
        self._docs.append(doc)
        return SimpleNamespace(inserted_id=doc["_id"])

    def insert_many(self, documents: List[Dict[str, Any]]):
        ids = []
        for document in documents:
            ids.append(self.insert_one(document).inserted_id)
        return SimpleNamespace(inserted_ids=ids)

    def find_one_and_update(
        self,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
        return_document: bool = False,
    ):
        doc = self.find_one(query)
        if not doc:
            if not upsert:
                return None
            new_doc = {"_id": query.get("_id", ObjectId())}
            if "$setOnInsert" in update:
                new_doc.update(deepcopy(update["$setOnInsert"]))
            if "$set" in update:
                new_doc.update(deepcopy(update["$set"]))
            self._docs.append(new_doc)
            return deepcopy(new_doc)

        stored_doc = next(d for d in self._docs if d["_id"] == doc["_id"])
        if "$set" in update:
            for key, value in update["$set"].items():
                stored_doc[key] = value
        return deepcopy(stored_doc)

    def update_one(self, query: Dict[str, Any], update: Dict[str, Any]):
        doc = self.find_one(query)
        if not doc:
            return SimpleNamespace(modified_count=0)
        stored_doc = next(d for d in self._docs if d["_id"] == doc["_id"])
        if "$set" in update:
            for key, value in update["$set"].items():
                stored_doc[key] = value
        if "$currentDate" in update:
            for key in update["$currentDate"].keys():
                stored_doc[key] = datetime.now(timezone.utc)
        return SimpleNamespace(modified_count=1)

    def create_index(self, *args, **kwargs):
        return None


class FakeDatabase:
    def __init__(self):
        self.collections = defaultdict(FakeCollection)

    def __getitem__(self, name: str):
        return self.collections[name]


class FakeMongoClient:
    def __init__(self):
        self.databases = defaultdict(FakeDatabase)

    def __getitem__(self, name: str):
        return self.databases[name]


@pytest.fixture(autouse=True)
def fake_mongo(monkeypatch):
    os.environ.setdefault("MONGO_URI", "mongodb://localhost")
    os.environ.setdefault("MONGO_DB_NAME", "eve_new_tests")
    os.environ.setdefault("DB", "TEST")

    client = FakeMongoClient()

    def fake_get_mongo_client():
        return client

    mongo_module._mongo_client = client
    mongo_module._collections.clear()
    monkeypatch.setattr(mongo_module, "get_mongo_client", fake_get_mongo_client)

    yield client

    mongo_module._mongo_client = None
    mongo_module._collections.clear()
