"""API keys are looked up by SHA-256 hash only; plaintext rows never match."""

import hashlib

import pytest
from bson import ObjectId
from fastapi import HTTPException

from eve import auth


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def find_one(self, query):
        self.queries.append(query)
        for doc in self.docs:
            if (
                all(
                    doc.get(k) == v for k, v in query.items() if not isinstance(v, dict)
                )
                and doc.get("deleted") is not True
            ):
                return doc
        return None


KEY = "a" * 48
USER_ID = ObjectId()


@pytest.fixture
def collections(monkeypatch):
    keys = FakeCollection([])
    users = FakeCollection([{"_id": USER_ID, "userId": "user_1", "isAdmin": True}])
    monkeypatch.setattr(auth, "get_api_keys", lambda: keys)
    monkeypatch.setattr(auth, "get_users", lambda: users)
    return keys, users


def test_hash_is_full_sha256_hex():
    assert auth.hash_api_key(KEY) == hashlib.sha256(KEY.encode()).hexdigest()
    assert len(auth.hash_api_key(KEY)) == 64


def test_hashed_row_authenticates(collections):
    keys, _ = collections
    keys.docs.append({"apiKeyHash": auth.hash_api_key(KEY), "user": USER_ID})
    user = auth.verify_api_key(KEY)
    assert user.userId == str(USER_ID)
    assert all("apiKey" not in q for q in keys.queries)


def test_plaintext_row_does_not_authenticate(collections):
    keys, _ = collections
    keys.docs.append({"apiKey": KEY, "user": USER_ID})
    with pytest.raises(HTTPException) as exc:
        auth.verify_api_key(KEY)
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        auth.authenticate_admin_api_key(KEY)
    assert exc.value.status_code == 401


def test_deleted_row_does_not_authenticate(collections):
    keys, _ = collections
    keys.docs.append(
        {"apiKeyHash": auth.hash_api_key(KEY), "user": USER_ID, "deleted": True}
    )
    with pytest.raises(HTTPException) as exc:
        auth.verify_api_key(KEY)
    assert exc.value.status_code == 401


def test_unknown_or_empty_key_is_401(collections):
    for value in ("nope", "", None):
        with pytest.raises(HTTPException) as exc:
            auth.verify_api_key(value)
        assert exc.value.status_code == 401
