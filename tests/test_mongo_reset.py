from unittest.mock import MagicMock

from eve import mongo


def test_reset_drops_client_and_collections(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(mongo, "_mongo_client", fake)
    mongo._collections["PROD:tasks3"] = object()

    mongo.reset_mongo_client()

    assert mongo._mongo_client is None
    assert mongo._collections == {}
    fake.close.assert_called_once()


def test_reset_is_safe_without_a_client(monkeypatch):
    monkeypatch.setattr(mongo, "_mongo_client", None)
    mongo.reset_mongo_client()
    assert mongo._mongo_client is None
