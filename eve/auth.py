import os
from typing import Optional

import httpx
from bson import ObjectId
from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from . import EDEN_API_KEY
from .mongo import get_collection
from .user import User

# Lazy load Clerk SDK to avoid circular import issues
_clerk = None


def get_clerk():
    global _clerk
    if _clerk is None:
        try:
            from clerk_backend_api import Clerk

            _clerk = Clerk(bearer_auth=os.getenv("CLERK_SECRET_KEY"))
        except ImportError as e:
            raise ImportError(f"Failed to import Clerk SDK: {e}") from e
    return _clerk


api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

EDEN_ADMIN_KEY = os.getenv("EDEN_ADMIN_KEY")
ABRAHAM_ADMIN_KEY = os.getenv("ABRAHAM_ADMIN_KEY")
ISSUER_URL = os.getenv("CLERK_ISSUER_URL")

# Lazy load collections
_api_keys = None
_users = None


def get_api_keys():
    global _api_keys
    if _api_keys is None:
        _api_keys = get_collection("apikeys")
    return _api_keys


def get_users():
    global _users
    if _users is None:
        _users = get_collection("users3")
    return _users


class UserData(BaseModel):
    userId: str
    subscriptionTier: int = 0
    featureFlags: list = []
    isAdmin: bool = False


def get_my_eden_user() -> str:
    """Get the user id for the api key in your env file"""
    api_key = EDEN_API_KEY
    api_key = get_api_keys().find_one({"apiKey": api_key.get_secret_value()})
    if not api_key:
        raise HTTPException(status_code=401, detail="API key not found")
    user = User.from_mongo(api_key["user"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_user_data(user_id: str) -> UserData:
    """Get user data from DB and return structured format"""
    user = get_users().find_one({"userId": user_id})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return UserData(
        userId=str(user["_id"]),
        subscriptionTier=user.get("subscriptionTier", 0),
        featureFlags=user.get("featureFlags", []),
        isAdmin=user.get("isAdmin", False),
    )


def verify_api_key(api_key: str) -> dict:
    api_key = get_api_keys().find_one({"apiKey": api_key})
    user_obj = get_users().find_one({"_id": ObjectId(api_key["user"])})
    if user_obj is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )
    user_data = get_user_data(user_obj["userId"])
    return user_data


async def get_clerk_session(
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """Verify Clerk session and return user data"""
    try:
        from clerk_backend_api.jwks_helpers import AuthenticateRequestOptions

        clerk = get_clerk()
        # Create a mock httpx.Request with the token
        mock_request = httpx.Request(
            method="GET",
            url="https://eden.art",
            headers={"Authorization": f"Bearer {token.credentials}"},
        )

        request_state = clerk.authenticate_request(
            mock_request,
            AuthenticateRequestOptions(authorized_parties=[ISSUER_URL]),
        )

        if not request_state.is_signed_in:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Not signed in"
            )

        user_id = request_state.payload["sub"]
        user = get_user_data(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )
        return user
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


async def authenticate(
    api_key: str = Depends(api_key_header),
    bearer_token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """Authenticate using either API key or Clerk session"""
    if api_key:
        return verify_api_key(api_key)

    if bearer_token:
        return await get_clerk_session(bearer_token)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Either API key or valid auth token required",
    )


async def authenticate_ws(websocket: WebSocket):
    api_key = websocket.headers.get("X-Api-Key")
    token = websocket.headers.get("Authorization")
    if token:
        token = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=token.replace("Bearer ", "")
        )
    try:
        user = await authenticate(api_key=api_key, bearer_token=token)
        return user
    except HTTPException as e:
        await websocket.accept()
        await websocket.send_json({"error": e.detail})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise e


def authenticate_admin(
    token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
):
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )
    if token.credentials != EDEN_ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )


def authenticate_admin_api_key(
    api_key: Optional[str] = Depends(api_key_header),
):
    """Authenticate admin users by checking their API key's admin status"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required"
        )

    api_key_doc = get_api_keys().find_one({"apiKey": api_key})
    if not api_key_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    user_obj = get_users().find_one({"_id": ObjectId(api_key_doc["user"])})
    if not user_obj or not user_obj.get("isAdmin", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin access required"
        )
