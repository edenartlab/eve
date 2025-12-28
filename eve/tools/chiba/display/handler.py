import os

import requests

from eve.tool import ToolContext


async def handler(context: ToolContext):
    """
    Control the Chiba kiosk display via HTTP API.

    Supports actions: status, files, play, off, url, cache, sync, sync_and_play
    """
    base_url = os.getenv("CHIBA_DISPLAY_URL")
    api_key = os.getenv("CHIBA_API_KEY")

    if not base_url:
        raise ValueError("CHIBA_DISPLAY_URL environment variable not set")
    if not api_key:
        raise ValueError("CHIBA_API_KEY environment variable not set")

    base_url = base_url.rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    action = context.args.get("action")
    if not action:
        raise ValueError("action parameter is required")

    if action == "status":
        response = requests.get(f"{base_url}/status")
        response.raise_for_status()
        return {"output": response.json()}

    elif action == "files":
        response = requests.get(f"{base_url}/files")
        response.raise_for_status()
        return {"output": response.json()}

    elif action == "play":
        file = context.args.get("file")
        if not file:
            raise ValueError("file parameter is required for 'play' action")
        response = requests.post(
            f"{base_url}/file",
            headers=headers,
            json={"file": file},
        )
        response.raise_for_status()
        return {"output": response.json()}

    elif action == "off":
        response = requests.post(f"{base_url}/off", headers=headers)
        response.raise_for_status()
        return {"output": response.json()}

    elif action == "url":
        url = context.args.get("url")
        if not url:
            raise ValueError("url parameter is required for 'url' action")
        response = requests.post(
            f"{base_url}/url",
            headers=headers,
            json={"url": url},
        )
        response.raise_for_status()
        return {"output": response.json()}

    elif action == "cache":
        url = context.args.get("url")
        if not url:
            raise ValueError("url parameter is required for 'cache' action")
        response = requests.post(
            f"{base_url}/cache",
            headers=headers,
            json={"url": url},
        )
        response.raise_for_status()
        result = response.json()

        play_after_cache = context.args.get("play_after_cache", False)
        if play_after_cache and "filename" in result:
            play_response = requests.post(
                f"{base_url}/file",
                headers=headers,
                json={"file": result["filename"]},
            )
            play_response.raise_for_status()
            result["play_result"] = play_response.json()

        return {"output": result}

    elif action == "sync":
        collection_id = context.args.get("collection_id")
        if not collection_id:
            raise ValueError("collection_id parameter is required for 'sync' action")
        response = requests.post(
            f"{base_url}/sync",
            headers=headers,
            json={"collectionId": collection_id},
        )
        response.raise_for_status()
        return {"output": response.json()}

    elif action == "sync_and_play":
        collection_id = context.args.get("collection_id")
        if not collection_id:
            raise ValueError(
                "collection_id parameter is required for 'sync_and_play' action"
            )
        loop = context.args.get("loop", True)
        response = requests.post(
            f"{base_url}/sync_and_play",
            headers=headers,
            json={"collectionId": collection_id, "loop": loop},
        )
        response.raise_for_status()
        return {"output": response.json()}

    else:
        raise ValueError(f"Unknown action: {action}")
