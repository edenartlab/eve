from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
import requests
from datetime import datetime, timedelta
from eve.utils import file_utils as utils
from typing import Dict, Any
import tempfile
import asyncio
import json
import os


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")

    agent = Agent.from_mongo(agent)

    deployment = Deployment.load(agent=agent.id, platform="instagram")
    if not deployment:
        raise Exception("No valid Instagram deployments found")

    # Get parameters from args
    media_url = args["media_url"]
    caption = args.get("caption", "")
    media_type = args.get("media_type", "IMAGE")

    # Get Instagram secrets from deployment
    if not deployment.secrets or not deployment.secrets.instagram:
        raise Exception("Instagram credentials not found in deployment")

    instagram_secrets = deployment.secrets.instagram
    access_token = instagram_secrets.access_token
    user_id = instagram_secrets.user_id

    # Check if token needs refresh (if expires_at is set)
    if instagram_secrets.expires_at and instagram_secrets.expires_at < datetime.now():
        raise Exception("Instagram access token has expired. Please re-authenticate.")

    try:
        # Download media file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".jpg" if media_type == "IMAGE" else ".mp4", delete=False
        )
        
        media_file_path = utils.download_file(media_url, temp_file.name, overwrite=True)

        # Step 1: Create media container
        container_params = {
            "media_type": media_type,
            "access_token": access_token,
        }

        if media_type == "IMAGE":
            container_params["image_url"] = media_url
        else:  # VIDEO
            container_params["video_url"] = media_url

        if caption:
            container_params["caption"] = caption

        container_response = requests.post(
            f"https://graph.instagram.com/v21.0/{user_id}/media",
            params=container_params,
        )

        if not container_response.ok:
            error_data = container_response.json()
            raise Exception(f"Failed to create media container: {error_data.get('error', {}).get('message', container_response.text)}")

        container_data = container_response.json()
        creation_id = container_data["id"]

        # Step 2: Check container status for videos (they need processing time)
        if media_type == "VIDEO":
            await _wait_for_video_processing(creation_id, access_token, max_retries=10, retry_delay=10)

        # Step 3: Publish the media
        publish_response = requests.post(
            f"https://graph.instagram.com/v21.0/{user_id}/media_publish",
            params={
                "creation_id": creation_id,
                "access_token": access_token,
            },
        )

        if not publish_response.ok:
            error_data = publish_response.json()
            raise Exception(f"Failed to publish media: {error_data.get('error', {}).get('message', publish_response.text)}")

        publish_data = publish_response.json()
        media_id = publish_data["id"]

        # Get the published post info to construct URL
        post_info_response = requests.get(
            f"https://graph.instagram.com/v21.0/{media_id}",
            params={
                "fields": "id,media_type,media_url,permalink,username,timestamp",
                "access_token": access_token,
            },
        )

        instagram_url = f"https://www.instagram.com/p/{media_id}/"
        username = instagram_secrets.username or "unknown"

        if post_info_response.ok:
            post_data = post_info_response.json()
            permalink = post_data.get("permalink")
            if permalink:
                instagram_url = permalink

        response_data = {
            "output": [
                {
                    "url": instagram_url,
                    "media_id": media_id,
                    "media_type": media_type,
                    "title": caption or f"Instagram {media_type.lower()} post",
                    "status": "posted",
                    "username": username,
                }
            ]
        }

        return response_data

    except Exception as e:
        raise Exception(f"Failed to post to Instagram: {str(e)}")

    finally:
        # Clean up temp file
        try:
            if 'media_file_path' in locals():
                os.unlink(media_file_path)
        except:
            pass


async def _wait_for_video_processing(creation_id: str, access_token: str, max_retries: int = 10, retry_delay: int = 10):
    """Wait for video to finish processing before publishing"""
    for attempt in range(max_retries):
        try:
            status_response = requests.get(
                f"https://graph.instagram.com/v21.0/{creation_id}",
                params={
                    "fields": "status_code",
                    "access_token": access_token,
                },
            )

            if status_response.ok:
                status_data = status_response.json()
                status_code = status_data.get("status_code")
                
                print(f"[DEBUG] Video processing status (attempt {attempt + 1}): {status_code}")
                
                if status_code == "FINISHED":
                    print("[DEBUG] Video processing completed")
                    return
                elif status_code == "ERROR":
                    raise Exception("Video processing failed")
                elif status_code == "IN_PROGRESS":
                    if attempt < max_retries - 1:
                        print(f"[DEBUG] Video still processing, waiting {retry_delay}s")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print("[DEBUG] Max retries reached, proceeding anyway")
                        return
                else:
                    print(f"[DEBUG] Unknown status code: {status_code}, proceeding")
                    return
            else:
                print(f"[DEBUG] Failed to check video status: {status_response.text}")
                # If we can't check status, just proceed
                return

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[DEBUG] Video status check failed, proceeding anyway: {e}")
                return
            print(f"[DEBUG] Status check failed (attempt {attempt + 1}): {e}")
            await asyncio.sleep(retry_delay)

    print("[DEBUG] Max retries reached for video processing, proceeding")