import asyncio
import random
import requests
from typing import Dict, Any

from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
from eve.tool import ToolContext


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    agent_obj = Agent.from_mongo(context.agent)
    deployment = Deployment.load(agent=agent_obj.id, platform="captions")
    if not deployment:
        raise Exception("No valid Captions deployments found")

    api_key = deployment.secrets.captions.api_key

    if not api_key:
        raise ValueError("Missing required Captions API key")

    BASE_URL = "https://api.captions.ai"
    HEADERS = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "EdenCaptions/1.0.0",
    }

    def _api_request(method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make a request to the Captions API"""
        url = f"{BASE_URL}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, headers=HEADERS, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=HEADERS, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise RuntimeError("Invalid API key. Please check your credentials.")
            elif e.response.status_code == 429:
                raise RuntimeError(
                    "Rate limit exceeded. Please wait before making another request."
                )
            elif e.response.status_code == 402:
                raise RuntimeError(
                    "Insufficient credits. Please purchase more credits."
                )
            elif e.response.status_code == 400:
                try:
                    error_detail = e.response.json()
                    raise RuntimeError(f"Bad request: {error_detail}")
                except Exception:
                    raise RuntimeError(f"Bad request: {e.response.text}")
            else:
                raise RuntimeError(f"API request failed: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    # Validate inputs
    script = context.args.get("script")
    if not script or len(script) < 10 or len(script) > 800:
        raise ValueError("Script must be between 10 and 800 characters long")

    media_urls = context.args.get("media_urls")
    if not media_urls or not isinstance(media_urls, list) or len(media_urls) == 0:
        raise ValueError("At least one media URL is required")

    # Select creator
    creator_name = context.args.get("creator_name")
    if not creator_name:
        # Fetch available creators from API and select random one
        try:
            creators_response = _api_request("POST", "/api/ads/list-creators")
            available_creators = creators_response.get("supportedCreators", [])
            if available_creators:
                creator_name = random.choice(available_creators)
            else:
                raise RuntimeError("No creators available from API")
        except Exception:
            # Fallback to a few known creators if API fails
            fallback_creators = [
                "Jason",
                "Grace-1",
                "Isabella-1",
                "James",
                "Ava-1",
                "Michael",
                "Olivia",
            ]
            creator_name = random.choice(fallback_creators)

    resolution = context.args.get("resolution", "fhd")
    webhook_id = context.args.get("webhook_id")
    wait_for_completion = context.args.get("wait_for_completion", True)

    # Prepare payload
    payload = {
        "script": script,
        "creatorName": creator_name,
        "mediaUrls": media_urls,
        "resolution": resolution,
    }

    if webhook_id:
        payload["webhookId"] = webhook_id

    # Submit video generation request
    submit_response = _api_request("POST", "/api/ads/submit", payload)
    operation_id = submit_response.get("operationId")

    if not operation_id:
        raise RuntimeError("No operation ID received from API")

    # If not waiting for completion, return operation ID
    if not wait_for_completion:
        return {
            "output": {
                "operation_id": operation_id,
                "status": "processing",
                "message": "Video generation started. Use operation ID to check status.",
            }
        }

    # Wait for video generation to complete
    max_attempts = 360  # 1 hour with 10-second intervals
    poll_interval = 10

    for attempt in range(max_attempts):
        # Check job status
        status_payload = {"operationId": operation_id}
        status_response = _api_request("POST", "/api/ads/poll", status_payload)

        state = status_response.get("state", "unknown")

        if state == "COMPLETE":
            video_url = status_response.get("url")
            if video_url:
                # Clean URL - remove query parameters for display
                video_url.split("?")[0] if "?" in video_url else video_url
                return {"output": video_url}
            else:
                raise RuntimeError("Job completed but no video URL found")

        elif state == "FAILED":
            error = status_response.get("error", "Unknown error")
            raise RuntimeError(f"Video generation failed: {error}")

        elif state in ["PENDING", "PROCESSING", "QUEUED"]:
            await asyncio.sleep(poll_interval)
            continue

        else:
            raise RuntimeError(f"Unknown job state: {state}")

    # Timeout
    raise RuntimeError(
        f"Video generation timed out after {max_attempts * poll_interval} seconds"
    )
