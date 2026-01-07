import json
import os

import httpx
from loguru import logger

from eve.tool import ToolContext


async def handler(context: ToolContext):
    """
    Control the Chiba kiosk display via HTTP API.

    Supports actions: status, health, files, play, stop, pause, resume,
    next, previous, volume, loop, image_duration, cache, clear_cache
    """
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

    def build_url(endpoint: str) -> str:
        """Build URL, optionally routing through specific kiosk."""
        if kiosk:
            # Route to specific node via controller proxy
            return f"{base_url}/api/nodes/{kiosk}/{endpoint}"
        return f"{base_url}/{endpoint}"

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

    async with httpx.AsyncClient(timeout=60.0) as client:
        # === Public Endpoints (GET) ===

        if action == "status":
            response = await client.get(build_url("status"))
            result = handle_response(response, "status")
            logger.info(f"[DISPLAY] status result: {result}")
            return {"output": result}

        elif action == "health":
            response = await client.get(build_url("health"))
            result = handle_response(response, "health")
            logger.info(f"[DISPLAY] health result: {result}")
            return {"output": result}

        elif action == "files":
            response = await client.get(build_url("files"))
            result = handle_response(response, "files")
            logger.info(f"[DISPLAY] files result: {result}")
            return {"output": result}

        # === Protected Endpoints (POST) ===

        elif action == "play":
            # Unified play endpoint - auto-detects content type from parameters
            request_data = {}

            # Content source (one of these)
            if context.args.get("filename"):
                request_data["filename"] = context.args["filename"]
            elif context.args.get("url"):
                request_data["url"] = context.args["url"]
            elif context.args.get("collection_id"):
                request_data["collectionId"] = context.args["collection_id"]
                request_data["db"] = context.args.get("db", os.getenv("DB", "PROD"))
            elif context.args.get("creation_id"):
                request_data["creationId"] = context.args["creation_id"]
                request_data["db"] = context.args.get("db", os.getenv("DB", "PROD"))
            else:
                logger.error(
                    f"[DISPLAY] No content source for play action. Args: {context.args}"
                )
                raise ValueError(
                    "play action requires one of: filename, url, collection_id, or creation_id. "
                    f"Received args: {context.args}"
                )

            # Optional parameters
            if context.args.get("loop") is not None:
                request_data["loop"] = context.args["loop"]
            if context.args.get("name"):
                request_data["name"] = context.args["name"]

            response = await client.post(
                build_url("play"),
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "play", request_data)
            logger.info(f"[DISPLAY] play result: {result}")
            return {"output": result}

        elif action == "stop":
            response = await client.post(
                build_url("stop"),
                headers=headers,
            )
            result = handle_response(response, "stop")
            logger.info(f"[DISPLAY] stop result: {result}")
            return {"output": result}

        elif action == "pause":
            response = await client.post(
                build_url("pause"),
                headers=headers,
            )
            result = handle_response(response, "pause")
            logger.info(f"[DISPLAY] pause result: {result}")
            return {"output": result}

        elif action == "resume":
            response = await client.post(
                build_url("resume"),
                headers=headers,
            )
            result = handle_response(response, "resume")
            logger.info(f"[DISPLAY] resume result: {result}")
            return {"output": result}

        elif action == "next":
            response = await client.post(
                build_url("next"),
                headers=headers,
            )
            result = handle_response(response, "next")
            logger.info(f"[DISPLAY] next result: {result}")
            return {"output": result}

        elif action == "previous":
            response = await client.post(
                build_url("previous"),
                headers=headers,
            )
            result = handle_response(response, "previous")
            logger.info(f"[DISPLAY] previous result: {result}")
            return {"output": result}

        elif action == "volume":
            level = context.args.get("level")
            if level is None:
                logger.error(
                    f"[DISPLAY] Missing 'level' for volume action. Args: {context.args}"
                )
                raise ValueError(
                    f"level parameter (0-100) is required for 'volume' action. "
                    f"Received args: {context.args}"
                )
            request_data = {"level": level}
            response = await client.post(
                build_url("volume"),
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "volume", request_data)
            logger.info(f"[DISPLAY] volume result: {result}")
            return {"output": result}

        elif action == "loop":
            request_data = {}
            if context.args.get("loop") is not None:
                request_data["enabled"] = context.args["loop"]
            # If no loop param provided, API will toggle
            response = await client.post(
                build_url("loop"),
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "loop", request_data)
            logger.info(f"[DISPLAY] loop result: {result}")
            return {"output": result}

        elif action == "image_duration":
            duration = context.args.get("duration")
            if duration is None:
                logger.error(
                    f"[DISPLAY] Missing 'duration' for image_duration action. Args: {context.args}"
                )
                raise ValueError(
                    f"duration parameter (milliseconds, min 1000) is required for 'image_duration' action. "
                    f"Received args: {context.args}"
                )
            request_data = {"duration": duration}
            response = await client.post(
                build_url("image-duration"),
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "image_duration", request_data)
            logger.info(f"[DISPLAY] image_duration result: {result}")
            return {"output": result}

        elif action == "cache":
            # Cache content without playing
            request_data = {}

            if context.args.get("url"):
                request_data["url"] = context.args["url"]
            elif context.args.get("collection_id"):
                request_data["collectionId"] = context.args["collection_id"]
                request_data["db"] = context.args.get("db", os.getenv("DB", "PROD"))
            elif context.args.get("creation_id"):
                request_data["creationId"] = context.args["creation_id"]
                request_data["db"] = context.args.get("db", os.getenv("DB", "PROD"))
            else:
                logger.error(
                    f"[DISPLAY] No content source for cache action. Args: {context.args}"
                )
                raise ValueError(
                    "cache action requires one of: url, collection_id, or creation_id. "
                    f"Received args: {context.args}"
                )

            response = await client.post(
                build_url("cache"),
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "cache", request_data)
            logger.info(f"[DISPLAY] cache result: {result}")
            return {"output": result}

        elif action == "clear_cache":
            response = await client.post(
                build_url("clear-cache"),
                headers=headers,
            )
            result = handle_response(response, "clear_cache")
            logger.info(f"[DISPLAY] clear_cache result: {result}")
            return {"output": result}

        else:
            valid_actions = [
                "status",
                "health",
                "files",
                "play",
                "stop",
                "pause",
                "resume",
                "next",
                "previous",
                "volume",
                "loop",
                "image_duration",
                "cache",
                "clear_cache",
            ]
            logger.error(f"[DISPLAY] Unknown action '{action}'. Args: {context.args}")
            raise ValueError(
                f"Unknown action: {action}. Valid actions: {', '.join(valid_actions)}. "
                f"Received args: {context.args}"
            )
