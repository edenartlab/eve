from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
import requests
from datetime import datetime, timedelta
from eve.utils import file_utils as utils
from typing import Dict, Any
import tempfile


async def handler(args: dict, user: str = None, agent: str = None):
    print(f"[DEBUG] TikTok handler called with args: {args}")
    print(f"[DEBUG] User: {user}, Agent: {agent}")

    if not agent:
        raise Exception("Agent is required")

    print(f"[DEBUG] Loading agent from mongo: {agent}")
    agent = Agent.from_mongo(agent)
    print(f"[DEBUG] Agent loaded successfully: {agent.id}")

    print(f"[DEBUG] Loading TikTok deployment for agent: {agent.id}")
    deployment = Deployment.load(agent=agent.id, platform="tiktok")
    if not deployment:
        raise Exception("No valid TikTok deployments found")
    print(f"[DEBUG] Deployment loaded: {deployment.id}")

    # Get parameters from args
    video_url = args["video_url"]
    caption = args["caption"]
    privacy_level = args.get("privacy_level", "PUBLIC_TO_EVERYONE")

    print(f"[DEBUG] Parameters extracted:")
    print(f"[DEBUG]   Video URL: {video_url}")
    print(f"[DEBUG]   Caption: {caption}")
    print(f"[DEBUG]   Privacy Level: {privacy_level}")

    # Get TikTok secrets from deployment
    print("[DEBUG] Checking deployment secrets...")
    if not deployment.secrets or not deployment.secrets.tiktok:
        raise Exception("TikTok credentials not found in deployment")

    tiktok_secrets = deployment.secrets.tiktok
    access_token = tiktok_secrets.access_token
    print(
        f"[DEBUG] Access token retrieved: {access_token[:20]}..."
        if access_token
        else "[DEBUG] No access token found"
    )
    print(f"[DEBUG] Token expires at: {tiktok_secrets.expires_at}")

    # Check if token needs refresh
    if tiktok_secrets.expires_at and tiktok_secrets.expires_at < datetime.now():
        print("[DEBUG] Token expired, refreshing...")
        # Refresh token
        if tiktok_secrets.refresh_token:
            print("[DEBUG] Using refresh token to get new access token")
            new_tokens = await _refresh_token(tiktok_secrets.refresh_token)
            if "error" in new_tokens:
                print(f"[DEBUG] Token refresh failed: {new_tokens['error']}")
                raise Exception(new_tokens["error"])

            print("[DEBUG] Token refresh successful, updating deployment")
            # Update deployment with new tokens
            tiktok_secrets.access_token = new_tokens["access_token"]
            tiktok_secrets.refresh_token = new_tokens["refresh_token"]
            tiktok_secrets.expires_at = datetime.now() + timedelta(
                seconds=new_tokens["expires_in"]
            )
            deployment.save()
            access_token = new_tokens["access_token"]
            print(f"[DEBUG] New access token: {access_token[:20]}...")
    else:
        print("[DEBUG] Token is still valid, proceeding with existing token")

    # Download video file
    print(f"[DEBUG] Downloading video from: {video_url}")
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    print(f"[DEBUG] Created temp file: {temp_file.name}")

    video_file_path = utils.download_file(video_url, temp_file.name, overwrite=True)
    print(f"[DEBUG] Video downloaded to: {video_file_path}")

    # Read video file as bytes
    with open(video_file_path, "rb") as f:
        video_file = f.read()

    video_size = len(video_file)
    print(
        f"[DEBUG] Video file size: {video_size} bytes ({video_size / (1024*1024):.2f} MB)"
    )

    try:
        # Step 1: Initialize video upload
        print("[DEBUG] Step 1: Initializing video upload with TikTok API")
        
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
        print(f"[DEBUG] Init payload: {init_payload}")
        
        init_response = requests.post(
            "https://open.tiktokapis.com/v2/post/publish/video/init/",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=init_payload,
        )
        
        print(f"[DEBUG] Init response status: {init_response.status_code}")
        print(f"[DEBUG] Init response headers: {dict(init_response.headers)}")

        if not init_response.ok:
            print(f"[DEBUG] Init response failed: {init_response.text}")
            raise Exception(f"Failed to initialize upload: {init_response.text}")

        init_data = init_response.json()
        print(f"[DEBUG] Init response data: {init_data}")
        
        publish_id = init_data["data"]["publish_id"]
        upload_url = init_data["data"]["upload_url"]
        print(f"[DEBUG] Publish ID: {publish_id}")
        print(f"[DEBUG] Upload URL: {upload_url}")

        # Step 2: Upload video
        print("[DEBUG] Step 2: Uploading video to TikTok")
        
        upload_headers = {
            "Content-Type": "video/mp4",
            "Content-Range": f"bytes 0-{len(video_file)-1}/{len(video_file)}",
        }
        print(f"[DEBUG] Upload headers: {upload_headers}")
        print(f"[DEBUG] Uploading {len(video_file)} bytes to: {upload_url}")
        
        upload_response = requests.put(
            upload_url,
            headers=upload_headers,
            data=video_file,
        )

        print(f"[DEBUG] Upload response status: {upload_response.status_code}")
        print(f"[DEBUG] Upload response headers: {dict(upload_response.headers)}")
        
        if not upload_response.ok:
            print(f"[DEBUG] Upload response failed: {upload_response.text}")
            raise Exception(f"Failed to upload video: {upload_response.text}")
        
        print("[DEBUG] Video upload successful")

        # Step 3: Check publish status
        print("[DEBUG] Step 3: Checking publish status")
        
        status_payload = {"publish_id": publish_id}
        print(f"[DEBUG] Status payload: {status_payload}")
        
        status_response = requests.post(
            "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=status_payload,
        )

        print(f"[DEBUG] Status response status: {status_response.status_code}")
        print(f"[DEBUG] Status response headers: {dict(status_response.headers)}")

        if not status_response.ok:
            print(f"[DEBUG] Status response failed: {status_response.text}")
            raise Exception(f"Failed to check status: {status_response.text}")

        status_data = status_response.json()
        print(f"[DEBUG] Status response data: {status_data}")

        # Check if the publish was actually successful
        publish_status = status_data["data"]["status"]
        fail_reason = status_data["data"].get("fail_reason")
        
        print(f"[DEBUG] Publish status: {publish_status}")
        if fail_reason:
            print(f"[DEBUG] Failure reason: {fail_reason}")

        if publish_status == "FAILED":
            error_msg = f"TikTok video upload failed: {fail_reason}" if fail_reason else "TikTok video upload failed for unknown reason"
            print(f"[DEBUG] Upload failed, raising exception: {error_msg}")
            raise Exception(error_msg)
        
        if publish_status not in ["PUBLISHED", "PROCESSING_UPLOAD"]:
            print(f"[DEBUG] Unexpected status: {publish_status}, treating as potential failure")
            # Still proceed but with a warning message

        # For TikTok, we don't get a direct URL from the API response
        # The URL would typically be: https://www.tiktok.com/@username/video/{video_id}
        # But we don't get the video_id from the publish status response
        # So we'll use a general profile URL with the post info
        print("[DEBUG] Building response data")
        
        tiktok_url = (
            f"https://www.tiktok.com/@{tiktok_secrets.username}"
            if tiktok_secrets.username
            else "https://www.tiktok.com"
        )
        print(f"[DEBUG] Using TikTok URL: {tiktok_url}")

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
        print(f"[DEBUG] Final response data: {response_data}")
        
        return response_data

    except Exception as e:
        print(f"[DEBUG] Exception occurred: {type(e).__name__}: {str(e)}")
        print(f"[DEBUG] Full exception details: {e}")
        raise Exception(f"Failed to post to TikTok: {str(e)}")


async def _refresh_token(refresh_token: str) -> Dict[str, Any]:
    import os

    print(f"[DEBUG] _refresh_token called with refresh_token: {refresh_token[:20]}...")
    
    try:
        client_key = os.getenv("TIKTOK_CLIENT_KEY")
        client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
        
        print(f"[DEBUG] Client key available: {bool(client_key)}")
        print(f"[DEBUG] Client secret available: {bool(client_secret)}")
        
        refresh_payload = {
            "client_key": client_key,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        print(f"[DEBUG] Refresh payload (secrets masked): {{'client_key': bool(client_key), 'client_secret': bool(client_secret), 'grant_type': 'refresh_token', 'refresh_token': '{refresh_token[:10]}...'}}")
        
        response = requests.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            headers={"Content-Type": "application/json"},
            json=refresh_payload,
        )

        print(f"[DEBUG] Refresh response status: {response.status_code}")
        print(f"[DEBUG] Refresh response headers: {dict(response.headers)}")
        
        if not response.ok:
            print(f"[DEBUG] Refresh response failed: {response.text}")
            return {"error": f"Failed to refresh token: {response.text}"}

        refresh_data = response.json()
        print("[DEBUG] Refresh successful, got new tokens")
        return refresh_data
        
    except Exception as e:
        print(f"[DEBUG] Exception in _refresh_token: {type(e).__name__}: {str(e)}")
        return {"error": f"Token refresh failed: {str(e)}"}
