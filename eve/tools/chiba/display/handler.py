import json
import os

import requests
from loguru import logger

from eve.tool import ToolContext


async def handler(context: ToolContext):
    """
    Control the Chiba kiosk display via HTTP API.

    Supports actions: status, files, play, off, url, cache, sync, sync_and_play
    """
    # Log incoming args for debugging
    logger.info(f"[DISPLAY] Received args: {json.dumps(context.args, default=str)}")

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
        logger.error(
            f"[DISPLAY] Missing 'action' parameter. Args received: {context.args}"
        )
        raise ValueError(f"action parameter is required. Received args: {context.args}")

    def handle_response(response, action_name, request_data=None):
        """Helper to handle response and provide detailed error info."""
        if not response.ok:
            error_detail = {
                "action": action_name,
                "status_code": response.status_code,
                "response_text": response.text,
                "request_data": request_data,
                "args": context.args,
            }
            logger.error(
                f"[DISPLAY] API error: {json.dumps(error_detail, default=str)}"
            )
            raise ValueError(
                f"Display API error for '{action_name}': {response.status_code} - {response.text}. "
                f"Request data: {request_data}"
            )
        return response.json()

    if action == "status":
        response = requests.get(f"{base_url}/status")
        result = handle_response(response, "status")
        logger.info(f"[DISPLAY] status result: {result}")
        return {"output": result}

    elif action == "files":
        response = requests.get(f"{base_url}/files")
        result = handle_response(response, "files")
        logger.info(f"[DISPLAY] files result: {result}")
        return {"output": result}

    elif action == "play":
        file = context.args.get("file")
        if not file:
            logger.error(
                f"[DISPLAY] Missing 'file' for play action. Args: {context.args}"
            )
            raise ValueError(
                f"file parameter is required for 'play' action. Received args: {context.args}"
            )
        request_data = {"file": file}
        response = requests.post(
            f"{base_url}/file",
            headers=headers,
            json=request_data,
        )
        result = handle_response(response, "play", request_data)
        logger.info(f"[DISPLAY] play result: {result}")
        return {"output": result}

    elif action == "off":
        response = requests.post(f"{base_url}/off", headers=headers)
        result = handle_response(response, "off")
        logger.info(f"[DISPLAY] off result: {result}")
        return {"output": result}

    elif action == "url":
        url = context.args.get("url")
        if not url:
            logger.error(
                f"[DISPLAY] Missing 'url' for url action. Args: {context.args}"
            )
            raise ValueError(
                f"url parameter is required for 'url' action. Received args: {context.args}"
            )
        request_data = {"url": url}
        response = requests.post(
            f"{base_url}/url",
            headers=headers,
            json=request_data,
        )
        result = handle_response(response, "url", request_data)
        logger.info(f"[DISPLAY] url result: {result}")
        return {"output": result}

    elif action == "cache":
        url = context.args.get("url")
        if not url:
            logger.error(
                f"[DISPLAY] Missing 'url' for cache action. Args: {context.args}"
            )
            raise ValueError(
                f"url parameter is required for 'cache' action. Received args: {context.args}"
            )
        request_data = {"url": url}
        response = requests.post(
            f"{base_url}/cache",
            headers=headers,
            json=request_data,
        )
        result = handle_response(response, "cache", request_data)

        play_after_cache = context.args.get("play_after_cache", False)
        if play_after_cache and "filename" in result:
            play_request = {"file": result["filename"]}
            play_response = requests.post(
                f"{base_url}/file",
                headers=headers,
                json=play_request,
            )
            result["play_result"] = handle_response(
                play_response, "cache+play", play_request
            )

        logger.info(f"[DISPLAY] cache result: {result}")
        return {"output": result}

    elif action == "sync":
        collection_id = context.args.get("collection_id")
        if not collection_id:
            logger.error(
                f"[DISPLAY] Missing 'collection_id' for sync action. Args: {context.args}"
            )
            raise ValueError(
                f"collection_id parameter is required for 'sync' action. Received args: {context.args}"
            )
        request_data = {"collectionId": collection_id, "db": os.getenv("DB", "STAGE")}
        response = requests.post(
            f"{base_url}/sync",
            headers=headers,
            json=request_data,
        )
        result = handle_response(response, "sync", request_data)
        logger.info(f"[DISPLAY] sync result: {result}")
        return {"output": result}

    elif action == "sync_and_play":
        collection_id = context.args.get("collection_id")
        if not collection_id:
            logger.error(
                f"[DISPLAY] Missing 'collection_id' for sync_and_play action. Args: {context.args}"
            )
            raise ValueError(
                f"collection_id parameter is required for 'sync_and_play' action. Received args: {context.args}"
            )
        loop = context.args.get("loop", True)
        request_data = {
            "collectionId": collection_id,
            "loop": loop,
            "db": os.getenv("DB", "STAGE"),
        }
        response = requests.post(
            f"{base_url}/sync_and_play",
            headers=headers,
            json=request_data,
        )
        result = handle_response(response, "sync_and_play", request_data)
        logger.info(f"[DISPLAY] sync_and_play result: {result}")
        return {"output": result}

    else:
        logger.error(f"[DISPLAY] Unknown action '{action}'. Args: {context.args}")
        raise ValueError(
            f"Unknown action: {action}. Valid actions: status, files, play, off, url, cache, sync, sync_and_play. Received args: {context.args}"
        )
