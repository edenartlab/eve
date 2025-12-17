import os
import time
from typing import Optional

import requests
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
from eve.tool import ToolContext


def _normalize_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    normalized = token.strip().strip('"').strip("'")
    if normalized.lower().startswith("bearer "):
        normalized = normalized.split(" ", 1)[1].strip()
    # Tokens should never contain whitespace; remove any accidental newlines/spaces from copy/paste.
    normalized = "".join(normalized.split())
    return normalized or None


def _wait_for_media_ready(
    *, api_base: str, creation_id: str, auth_params: dict, headers: dict, timeout_s: int
) -> None:
    """
    Instagram can return error 9007 ("media not ready") if we publish too quickly.
    Poll container status until FINISHED (or ERROR/timeout).
    """
    deadline = time.time() + timeout_s
    last_payload = None

    while time.time() < deadline:
        resp = requests.get(
            f"{api_base}/{creation_id}",
            params={"fields": "status_code,status", **auth_params},
            headers=headers,
            timeout=15,
        )
        if resp.ok:
            payload = resp.json() or {}
            last_payload = payload
            status = payload.get("status_code") or payload.get("status")
            if status == "FINISHED":
                return
            if status == "ERROR":
                raise Exception(f"IG media container failed processing: {payload!r}")

        time.sleep(2)

    raise Exception(
        f"Timed out waiting for IG media to be ready (creation_id={creation_id}). Last status: {last_payload!r}"
    )


def _get_env_token() -> Optional[str]:
    token = (
        os.getenv("INSTAGRAM_ACCESS_TOKEN")
        or os.getenv("INSTAGRAM_USER_TOKEN")
        or os.getenv("IG_ACCESS_TOKEN")
        # Back-compat with older env names used in staging.
        or os.getenv("INSTAGRAM_PAGE_TOKEN")
        or os.getenv("IG_PAGE_TOKEN")
    )
    return _normalize_token(token)


def _get_env_ig_user_id() -> Optional[str]:
    return os.getenv("INSTAGRAM_IG_USER_ID")


async def handler(context: ToolContext):
    """
    Minimal Instagram publisher for staging: uses Instagram Login user access token.
    Args: image_url (required), caption (optional)
    """
    if not context.agent:
        raise Exception("Agent is required")

    image_url = context.args.get("image_url")
    caption = context.args.get("caption") or ""
    if not image_url:
        raise Exception("image_url is required")

    agent = Agent.from_mongo(context.agent)
    deployment = Deployment.load(agent=agent.id, platform="instagram")

    token = _get_env_token()
    if not token and deployment and deployment.secrets and deployment.secrets.instagram:
        token = _normalize_token(deployment.secrets.instagram.access_token)

    ig_user_id = _get_env_ig_user_id()
    if (
        not ig_user_id
        and deployment
        and deployment.config
        and deployment.config.instagram
    ):
        ig_user_id = deployment.config.instagram.ig_user_id

    api_host = os.getenv("INSTAGRAM_GRAPH_HOST", "https://graph.instagram.com").rstrip(
        "/"
    )
    api_version = os.getenv("INSTAGRAM_API_VERSION", "v24.0").strip().lstrip("/")
    api_base = f"{api_host}/{api_version}" if api_version else api_host

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    auth_params = {"access_token": token} if token else {}

    # If still missing, try to resolve via Graph using the user token
    if token and not ig_user_id:
        resp = requests.get(
            f"{api_base}/me",
            params={"fields": "user_id,username,id", **auth_params},
            headers=headers,
            timeout=15,
        )
        if resp.ok:
            data = resp.json() or {}
            ig_user_id = data.get("user_id") or data.get("id") or ig_user_id

    if not token or not ig_user_id:
        raise Exception(
            "Missing Instagram token or ig_user_id. Ensure INSTAGRAM_ACCESS_TOKEN is set or deployment has Instagram secrets."
        )

    # Step 1: create media container
    media_resp = requests.post(
        f"{api_base}/{ig_user_id}/media",
        data={"image_url": image_url, "caption": caption},
        params=auth_params,
        headers=headers,
        timeout=30,
    )
    if not media_resp.ok:
        raise Exception(f"Failed to create IG media: {media_resp.text}")
    creation_id = media_resp.json().get("id")
    if not creation_id:
        raise Exception("No creation_id returned from IG media creation")

    _wait_for_media_ready(
        api_base=api_base,
        creation_id=creation_id,
        auth_params=auth_params,
        headers=headers,
        timeout_s=int(os.getenv("INSTAGRAM_PUBLISH_WAIT_TIMEOUT_S", "90")),
    )

    # Step 2: publish
    publish_resp = requests.post(
        f"{api_base}/{ig_user_id}/media_publish",
        data={"creation_id": creation_id},
        params=auth_params,
        headers=headers,
        timeout=30,
    )
    if not publish_resp.ok:
        raise Exception(f"Failed to publish IG media: {publish_resp.text}")

    publish_id = publish_resp.json().get("id")

    # Step 3: fetch permalink
    permalink = None
    media_url = None
    info_resp = requests.get(
        f"{api_base}/{publish_id}",
        params={"fields": "permalink,media_url,caption", **auth_params},
        headers=headers,
        timeout=30,
    )
    if info_resp.ok:
        data = info_resp.json()
        permalink = data.get("permalink")
        media_url = data.get("media_url")

    logger.info(f"Instagram post published: {publish_id}")

    post_url = permalink or media_url

    return {
        "output": [
            {
                "type": "text",
                "value": "Instagram post published",
                "media_id": publish_id,
                "url": post_url,
                "media_url": media_url,
                "caption": caption,
            }
        ]
    }
