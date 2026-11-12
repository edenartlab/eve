"""Admin bearer: constant-time match, never matches when unset, never sent off-site."""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from eve import auth


def _bearer(value):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=value)


def test_admin_key_matches(monkeypatch):
    monkeypatch.setattr(auth, "EDEN_ADMIN_KEY", "k" * 40)
    assert auth.authenticate_admin(_bearer("k" * 40)) is None


@pytest.mark.parametrize("configured", [None, ""])
def test_unset_admin_key_never_matches(monkeypatch, configured):
    monkeypatch.setattr(auth, "EDEN_ADMIN_KEY", configured)
    for presented in ("", "None", "anything"):
        with pytest.raises(HTTPException) as exc:
            auth.authenticate_admin(_bearer(presented))
        assert exc.value.status_code == 401


def test_wrong_or_missing_admin_key_is_401(monkeypatch):
    monkeypatch.setattr(auth, "EDEN_ADMIN_KEY", "k" * 40)
    for token in (None, _bearer("k" * 39), _bearer("x" * 40)):
        with pytest.raises(HTTPException) as exc:
            auth.authenticate_admin(token)
        assert exc.value.status_code == 401


def test_updates_only_go_to_own_api(monkeypatch):
    from eve.api import helpers

    monkeypatch.delenv("LOCAL_API_URL", raising=False)
    monkeypatch.setenv("EDEN_API_URL", "https://api.example.test")
    assert helpers.is_own_api_url("https://api.example.test/v2/deployments/emission")
    assert not helpers.is_own_api_url("https://evil.example.test/collect")
    assert not helpers.is_own_api_url("https://api.example.test.evil.test/x")
    assert not helpers.is_own_api_url("https://api.example.test@evil.test/x")
    assert not helpers.is_own_api_url("http://api.example.test/x")
    assert not helpers.is_own_api_url(None)
    monkeypatch.delenv("EDEN_API_URL")
    assert not helpers.is_own_api_url("https://api.example.test/x")
