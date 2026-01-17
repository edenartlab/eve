import json
import os

import httpx
from loguru import logger

from eve.tool import ToolContext


async def handler(context: ToolContext):
    """
    Control the Chiba lighting system via HTTP API.

    Supports actions: status, control, presets, apply_preset
    """
    logger.info(f"[LIGHTS] Received args: {json.dumps(context.args, default=str)}")

    base_url = os.getenv("CHIBA_DISPLAY_URL")
    api_key = os.getenv("CHIBA_API_KEY")

    logger.info(f"[LIGHTS] base_url={base_url}, api_key={'***' if api_key else None}")

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
            f"[LIGHTS] Missing 'action' parameter. Args received: {context.args}"
        )
        raise ValueError(f"action parameter is required. Received args: {context.args}")

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
            logger.error(f"[LIGHTS] API error: {json.dumps(error_detail, default=str)}")
            raise ValueError(
                f"Lights API error for '{action_name}': {response.status_code} - {response.text}. "
                f"Request data: {request_data}"
            )
        return response.json()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # === GET Endpoints ===

        if action == "status":
            # Get all lights with their current state
            response = await client.get(f"{base_url}/api/lights")
            result = handle_response(response, "status")
            logger.info(f"[LIGHTS] status result: {result}")
            return {"output": result}

        elif action == "presets":
            # Get all available presets
            response = await client.get(f"{base_url}/api/presets")
            result = handle_response(response, "presets")
            logger.info(f"[LIGHTS] presets result: {result}")
            return {"output": result}

        # === POST Endpoints (require authentication) ===

        elif action == "control":
            target = context.args.get("target")
            if not target:
                logger.error(
                    f"[LIGHTS] Missing 'target' for control action. Args: {context.args}"
                )
                raise ValueError(
                    "target parameter is required for 'control' action. "
                    "Use 'all' for all lights or a specific zone code (gw1, gw2, ge1, ge2, a). "
                    f"Received args: {context.args}"
                )

            # Build request data from optional parameters
            request_data = {}
            if context.args.get("power") is not None:
                power = context.args["power"]
                # Coerce string values to boolean
                if isinstance(power, str):
                    power = power.lower() in ("true", "on", "1", "yes")
                request_data["power"] = power
            if context.args.get("hue") is not None:
                request_data["hue"] = context.args["hue"]
            if context.args.get("saturation") is not None:
                request_data["saturation"] = context.args["saturation"]
            if context.args.get("brightness") is not None:
                request_data["brightness"] = context.args["brightness"]

            if not request_data:
                logger.error(
                    f"[LIGHTS] No control parameters provided. Args: {context.args}"
                )
                raise ValueError(
                    "At least one control parameter (power, hue, saturation, brightness) "
                    f"is required for 'control' action. Received args: {context.args}"
                )

            endpoint = f"{base_url}/api/lights/{target}/control"
            response = await client.post(
                endpoint,
                headers=headers,
                json=request_data,
            )
            result = handle_response(response, "control", request_data)
            logger.info(f"[LIGHTS] control result for {target}: {result}")
            return {"output": result}

        elif action == "apply_preset":
            preset_id = context.args.get("preset_id")
            if not preset_id:
                logger.error(
                    f"[LIGHTS] Missing 'preset_id' for apply_preset action. Args: {context.args}"
                )
                raise ValueError(
                    "preset_id parameter is required for 'apply_preset' action. "
                    f"Received args: {context.args}"
                )

            response = await client.post(
                f"{base_url}/api/presets/{preset_id}/apply",
                headers=headers,
            )
            result = handle_response(response, "apply_preset")
            logger.info(f"[LIGHTS] apply_preset result: {result}")
            return {"output": result}

        else:
            valid_actions = [
                "status",
                "control",
                "presets",
                "apply_preset",
            ]
            logger.error(f"[LIGHTS] Unknown action '{action}'. Args: {context.args}")
            raise ValueError(
                f"Unknown action: {action}. Valid actions: {', '.join(valid_actions)}. "
                f"Received args: {context.args}"
            )
