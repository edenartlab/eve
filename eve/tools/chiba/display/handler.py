import json
import os

import httpx
from loguru import logger

from eve.tool import ToolContext


async def handler(context: ToolContext):
    """
    Control the Chiba kiosk display via HTTP API.

    Supports actions: status, files, play, off, url, cache, sync, sync_and_play,
    playlist, next, previous, restart, pause, resume, volume, get_volume
    """
    # Log incoming args for debugging
    logger.info(f"[DISPLAY] Received args: {json.dumps(context.args, default=str)}")

    base_url = os.getenv("CHIBA_DISPLAY_URL")
    api_key = os.getenv("CHIBA_API_KEY")

    logger.info(
        f"[DISPLAY] base_url={base_url}, api_key={api_key[:10] if api_key else None}..."
    )

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

    kiosk = context.args.get("kiosk")

    def handle_response(response, action_name, request_data=None):
        """Helper to handle response and provide detailed error info."""
        if not response.is_success:
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

    async with httpx.AsyncClient(timeout=30.0) as client:
        if action == "status":
            params = {"kiosk": kiosk} if kiosk else {}
            response = await client.get(f"{base_url}/status", params=params)
            result = handle_response(response, "status")
            logger.info(f"[DISPLAY] status result: {result}")
            return {"output": result}

        elif action == "files":
            params = {"kiosk": kiosk} if kiosk else {}
            response = await client.get(f"{base_url}/files", params=params)
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
            if kiosk:
                request_data["kiosk"] = kiosk
            response = await client.post(
                f"{base_url}/file",
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "play", request_data)
            logger.info(f"[DISPLAY] play result: {result}")
            return {"output": result}

        elif action == "off":
            request_data = {"kiosk": kiosk} if kiosk else {}
            response = await client.post(
                f"{base_url}/off", headers=headers, json=request_data
            )
            result = handle_response(response, "off", request_data)
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
            if kiosk:
                request_data["kiosk"] = kiosk
            response = await client.post(
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
            if kiosk:
                request_data["kiosk"] = kiosk
            response = await client.post(
                f"{base_url}/cache",
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "cache", request_data)

            play_after_cache = context.args.get("play_after_cache", False)
            if play_after_cache and "filename" in result:
                play_request = {"file": result["filename"]}
                if kiosk:
                    play_request["kiosk"] = kiosk
                play_response = await client.post(
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
            request_data = {
                "collectionId": collection_id,
                "db": os.getenv("DB", "STAGE"),
            }
            if kiosk:
                request_data["kiosk"] = kiosk
            response = await client.post(
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
            if kiosk:
                request_data["kiosk"] = kiosk
            response = await client.post(
                f"{base_url}/sync_and_play",
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "sync_and_play", request_data)
            logger.info(f"[DISPLAY] sync_and_play result: {result}")
            return {"output": result}

        elif action == "playlist":
            playlist = context.args.get("playlist")
            if not playlist:
                logger.error(
                    f"[DISPLAY] Missing 'playlist' for playlist action. Args: {context.args}"
                )
                raise ValueError(
                    f"playlist parameter is required for 'playlist' action. Received args: {context.args}"
                )
            loop = context.args.get("loop", True)
            request_data = {"items": playlist, "loop": loop}
            if kiosk:
                request_data["kiosk"] = kiosk
            response = await client.post(
                f"{base_url}/playlist",
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "playlist", request_data)
            logger.info(f"[DISPLAY] playlist result: {result}")
            return {"output": result}

        elif action == "next":
            request_data = {"kiosk": kiosk} if kiosk else {}
            response = await client.post(
                f"{base_url}/next", headers=headers, json=request_data
            )
            result = handle_response(response, "next", request_data)
            logger.info(f"[DISPLAY] next result: {result}")
            return {"output": result}

        elif action == "previous":
            request_data = {"kiosk": kiosk} if kiosk else {}
            response = await client.post(
                f"{base_url}/previous", headers=headers, json=request_data
            )
            result = handle_response(response, "previous", request_data)
            logger.info(f"[DISPLAY] previous result: {result}")
            return {"output": result}

        elif action == "restart":
            request_data = {"kiosk": kiosk} if kiosk else {}
            response = await client.post(
                f"{base_url}/restart", headers=headers, json=request_data
            )
            result = handle_response(response, "restart", request_data)
            logger.info(f"[DISPLAY] restart result: {result}")
            return {"output": result}

        elif action == "pause":
            request_data = {"kiosk": kiosk} if kiosk else {}
            response = await client.post(
                f"{base_url}/pause", headers=headers, json=request_data
            )
            result = handle_response(response, "pause", request_data)
            logger.info(f"[DISPLAY] pause result: {result}")
            return {"output": result}

        elif action == "resume":
            request_data = {"kiosk": kiosk} if kiosk else {}
            response = await client.post(
                f"{base_url}/resume", headers=headers, json=request_data
            )
            result = handle_response(response, "resume", request_data)
            logger.info(f"[DISPLAY] resume result: {result}")
            return {"output": result}

        elif action == "get_volume":
            params = {"kiosk": kiosk} if kiosk else {}
            response = await client.get(f"{base_url}/volume", params=params)
            result = handle_response(response, "get_volume")
            logger.info(f"[DISPLAY] get_volume result: {result}")
            return {"output": result}

        elif action == "volume":
            level = context.args.get("level")
            if level is None:
                logger.error(
                    f"[DISPLAY] Missing 'level' for volume action. Args: {context.args}"
                )
                raise ValueError(
                    f"level parameter is required for 'volume' action. Received args: {context.args}"
                )
            request_data = {"level": level}
            if kiosk:
                request_data["kiosk"] = kiosk
            response = await client.post(
                f"{base_url}/volume",
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "volume", request_data)
            logger.info(f"[DISPLAY] volume result: {result}")
            return {"output": result}

        else:
            logger.error(f"[DISPLAY] Unknown action '{action}'. Args: {context.args}")
            raise ValueError(
                f"Unknown action: {action}. Valid actions: status, files, play, off, url, cache, sync, sync_and_play, playlist, next, previous, restart, pause, resume, get_volume, volume. Received args: {context.args}"
            )
