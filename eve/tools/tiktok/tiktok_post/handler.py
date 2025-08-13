from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
import requests
from datetime import datetime, timedelta
from eve.utils import file_utils as utils
from typing import Dict, Any
import tempfile


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")

    agent = Agent.from_mongo(agent)

    deployment = Deployment.load(agent=agent.id, platform="tiktok")
    if not deployment:
        raise Exception("No valid TikTok deployments found")

    # Get parameters from args
    video_url = args["video_url"]
    caption = args["caption"]
    privacy_level = args.get("privacy_level", "PUBLIC_TO_EVERYONE")

    # Get TikTok secrets from deployment
    if not deployment.secrets or not deployment.secrets.tiktok:
        raise Exception("TikTok credentials not found in deployment")

    tiktok_secrets = deployment.secrets.tiktok
    access_token = tiktok_secrets.access_token

    # Check if token needs refresh
    if tiktok_secrets.expires_at and tiktok_secrets.expires_at < datetime.now():
        # Refresh token
        if tiktok_secrets.refresh_token:
            new_tokens = await _refresh_token(tiktok_secrets.refresh_token)
            if "error" in new_tokens:
                raise Exception(new_tokens["error"])

            # Update deployment with new tokens
            tiktok_secrets.access_token = new_tokens["access_token"]
            tiktok_secrets.refresh_token = new_tokens["refresh_token"]
            tiktok_secrets.expires_at = datetime.now() + timedelta(
                seconds=new_tokens["expires_in"]
            )
            deployment.save()
            access_token = new_tokens["access_token"]

    # Download video file
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    video_file_path = utils.download_file(video_url, temp_file.name, overwrite=True)

    # Read video file as bytes
    with open(video_file_path, "rb") as f:
        video_file = f.read()

    video_size = len(video_file)

    try:
        # Step 1: Initialize video upload

        # Calculate chunk info for TikTok API
        video_size = len(video_file)
        chunk_size = min(video_size, 10 * 1024 * 1024)  # 10MB max chunk size
        total_chunk_count = 1  # Single chunk upload for simplicity

        init_payload = {
            "post_info": {
                "title": caption,
                "privacy_level": privacy_level,
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
                "video_cover_timestamp_ms": 1000,  # Use 1 second as thumbnail timestamp
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": chunk_size,
                "total_chunk_count": total_chunk_count,
            },
        }

        init_response = requests.post(
            "https://open.tiktokapis.com/v2/post/publish/video/init/",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=init_payload,
        )

        if not init_response.ok:
            raise Exception(f"Failed to initialize upload: {init_response.text}")

        init_data = init_response.json()

        publish_id = init_data["data"]["publish_id"]
        upload_url = init_data["data"]["upload_url"]

        # Step 2: Upload video

        upload_headers = {
            "Content-Type": "video/mp4",
            "Content-Range": f"bytes 0-{len(video_file)-1}/{len(video_file)}",
        }

        upload_response = requests.put(
            upload_url,
            headers=upload_headers,
            data=video_file,
        )

        if not upload_response.ok:
            raise Exception(f"Failed to upload video: {upload_response.text}")

        # Step 3: Check publish status

        status_payload = {"publish_id": publish_id}

        status_response = requests.post(
            "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=status_payload,
        )

        if not status_response.ok:
            raise Exception(f"Failed to check status: {status_response.text}")

        status_data = status_response.json()

        # Check if the publish was actually successful
        publish_status = status_data["data"]["status"]
        fail_reason = status_data["data"].get("fail_reason")

        if fail_reason:
            print(f"[DEBUG] Failure reason: {fail_reason}")
            raise Exception(fail_reason)

        if publish_status == "FAILED":
            error_msg = (
                f"TikTok video upload failed: {fail_reason}"
                if fail_reason
                else "TikTok video upload failed for unknown reason"
            )
            raise Exception(error_msg)

        if publish_status not in ["PUBLISHED", "PROCESSING_UPLOAD"]:
            raise Exception(f"Unexpected publish status: {publish_status}")

        # For TikTok, we don't get a direct URL from the API response
        # The URL would typically be: https://www.tiktok.com/@username/video/{video_id}
        # But we don't get the video_id from the publish status response
        # So we'll use a general profile URL with the post info

        tiktok_url = (
            f"https://www.tiktok.com/@{tiktok_secrets.username}"
            if tiktok_secrets.username
            else "https://www.tiktok.com"
        )

        # Determine success message based on status
        if publish_status == "PUBLISHED":
            message = "TikTok post published successfully"
        elif publish_status == "PROCESSING_UPLOAD":
            message = "TikTok post is being processed"
        else:
            message = f"TikTok post status: {publish_status}"

        response_data = {
            "output": [
                {
                    "url": tiktok_url,
                    "publish_id": publish_id,
                    "status": publish_status,
                    "title": caption,
                    "message": message,
                }
            ]
        }

        return response_data

    except Exception as e:
        raise Exception(f"Failed to post to TikTok: {str(e)}")


async def _refresh_token(refresh_token: str) -> Dict[str, Any]:
    import os

    try:
        client_key = os.getenv("TIKTOK_CLIENT_KEY")
        client_secret = os.getenv("TIKTOK_CLIENT_SECRET")

        refresh_payload = {
            "client_key": client_key,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }

        response = requests.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            headers={"Content-Type": "application/json"},
            json=refresh_payload,
        )

        if not response.ok:
            return {"error": f"Failed to refresh token: {response.text}"}

        refresh_data = response.json()
        return refresh_data

    except Exception as e:
        return {"error": f"Token refresh failed: {str(e)}"}
