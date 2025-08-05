import os
import time
import random
import requests
from typing import Dict, Any, Optional
from datetime import datetime

from eve.agent.agent import Agent
from eve.agent.session.models import Deployment


# Available AI creators
AVAILABLE_CREATORS = ["Jason", "Sarah", "Emma", "Michael", "Olivia", "Daniel"]


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")
    
    agent_obj = Agent.from_mongo(agent)
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
        "User-Agent": "EdenCaptions/1.0.0"
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
                raise RuntimeError("Rate limit exceeded. Please wait before making another request.")
            elif e.response.status_code == 402:
                raise RuntimeError("Insufficient credits. Please purchase more credits.")
            elif e.response.status_code == 400:
                try:
                    error_detail = e.response.json()
                    raise RuntimeError(f"Bad request: {error_detail}")
                except:
                    raise RuntimeError(f"Bad request: {e.response.text}")
            else:
                raise RuntimeError(f"API request failed: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")
    
    # Validate inputs
    script = args.get("script")
    if not script or len(script) < 10:
        raise ValueError("Script must be at least 10 characters long")
    
    media_urls = args.get("media_urls")
    if not media_urls or not isinstance(media_urls, list) or len(media_urls) == 0:
        raise ValueError("At least one media URL is required")
    
    # Select creator
    creator_name = args.get("creator_name")
    if not creator_name:
        # Select random creator if not specified
        creator_name = random.choice(AVAILABLE_CREATORS)
        print(f"Selected random creator: {creator_name}")
    
    resolution = args.get("resolution", "fhd")
    webhook_id = args.get("webhook_id")
    wait_for_completion = args.get("wait_for_completion", True)
    
    # Prepare payload
    payload = {
        "script": script,
        "creatorName": creator_name,
        "mediaUrls": media_urls,
        "resolution": resolution
    }
    
    if webhook_id:
        payload["webhookId"] = webhook_id
    
    # Submit video generation request
    submit_response = _api_request("POST", "/api/ads/submit", payload)
    operation_id = submit_response.get("operationId")
    
    if not operation_id:
        raise RuntimeError("No operation ID received from API")
    
    print(f"Video generation started with operation ID: {operation_id}")
    
    # If not waiting for completion, return operation ID
    if not wait_for_completion:
        return {
            "output": [{
                "operation_id": operation_id,
                "status": "processing",
                "message": "Video generation started. Use operation ID to check status."
            }]
        }
    
    # Wait for video generation to complete
    max_attempts = 360  # 1 hour with 10-second intervals
    poll_interval = 10
    
    for attempt in range(max_attempts):
        # Check job status
        status_payload = {"operationId": operation_id}
        status_response = _api_request("POST", "/api/ads/poll", status_payload)
        
        state = status_response.get("state", "unknown")
        print(f"Job status: {state} (attempt {attempt + 1}/{max_attempts})")
        
        if state == "COMPLETE":
            video_url = status_response.get("url")
            if video_url:
                # Generate descriptive info
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ai_ad_{creator_name}_{resolution}_{timestamp}.mp4"
                
                return {
                    "output": [{
                        "url": video_url,
                        "operation_id": operation_id,
                        "creator": creator_name,
                        "resolution": resolution,
                        "filename": filename,
                        "status": "completed"
                    }]
                }
            else:
                raise RuntimeError("Job completed but no video URL found")
        
        elif state == "FAILED":
            error = status_response.get("error", "Unknown error")
            raise RuntimeError(f"Video generation failed: {error}")
        
        elif state in ["PENDING", "PROCESSING", "QUEUED"]:
            time.sleep(poll_interval)
            continue
        
        else:
            raise RuntimeError(f"Unknown job state: {state}")
    
    # Timeout
    raise RuntimeError(f"Video generation timed out after {max_attempts * poll_interval} seconds")