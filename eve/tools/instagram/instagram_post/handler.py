import os
from typing import Optional

import requests
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
from eve.tool import ToolContext


def _get_env_token() -> Optional[str]:
    return (
        os.getenv("INSTAGRAM_PAGE_TOKEN")
        or os.getenv("IG_ACCESS_TOKEN")
        or os.getenv("IG_PAGE_TOKEN")
    )


def _get_env_ig_user_id() -> Optional[str]:
    return os.getenv("INSTAGRAM_IG_USER_ID")


async def handler(context: ToolContext):
    """
    Minimal Instagram publisher for staging: uses shared page access token.
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
        token = deployment.secrets.instagram.access_token

    ig_user_id = _get_env_ig_user_id()
    if (
        not ig_user_id
        and deployment
        and deployment.config
        and deployment.config.instagram
    ):
        ig_user_id = deployment.config.instagram.ig_user_id

    # If still missing, try to resolve via Graph using the page token
    if token and not ig_user_id:
        resp = requests.get(
            "https://graph.facebook.com/v24.0/me",
            params={
                "fields": "instagram_business_account",
                "access_token": token,
            },
            timeout=15,
        )
        if resp.ok:
            ig_user_id = (
                resp.json().get("instagram_business_account", {}).get("id", ig_user_id)
            )

    if not token or not ig_user_id:
        raise Exception(
            "Missing Instagram token or ig_user_id. Ensure INSTAGRAM_PAGE_TOKEN is set or deployment has Instagram secrets."
        )

    # Step 1: create media container
    media_resp = requests.post(
        f"https://graph.facebook.com/v24.0/{ig_user_id}/media",
        data={"image_url": image_url, "caption": caption},
        params={"access_token": token},
        timeout=30,
    )
    if not media_resp.ok:
        raise Exception(f"Failed to create IG media: {media_resp.text}")
    creation_id = media_resp.json().get("id")
    if not creation_id:
        raise Exception("No creation_id returned from IG media creation")

    # Step 2: publish
    publish_resp = requests.post(
        f"https://graph.facebook.com/v24.0/{ig_user_id}/media_publish",
        data={"creation_id": creation_id},
        params={"access_token": token},
        timeout=30,
    )
    if not publish_resp.ok:
        raise Exception(f"Failed to publish IG media: {publish_resp.text}")

    publish_id = publish_resp.json().get("id")

    # Step 3: fetch permalink
    permalink = None
    media_url = None
    info_resp = requests.get(
        f"https://graph.facebook.com/v24.0/{publish_id}",
        params={"fields": "permalink,media_url,caption", "access_token": token},
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
