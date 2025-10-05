from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
import requests
from datetime import datetime, timedelta
from eve.utils import file_utils as utils
from typing import Dict, Any
import tempfile
import asyncio
import json
import re


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
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
    
    # Note: In test/private setups, posts may be forced to SELF_ONLY
    # In production with audited API clients, PUBLIC_TO_EVERYONE should work
    print(f"[DEBUG] Using privacy level: {privacy_level}")

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
                raise Exception(f"Token refresh failed: {new_tokens['error']}")

            # Update deployment with new tokens
            tiktok_secrets.access_token = new_tokens["access_token"]
            tiktok_secrets.refresh_token = new_tokens["refresh_token"]
            tiktok_secrets.expires_at = datetime.now() + timedelta(
                seconds=new_tokens["expires_in"]
            )
            deployment.save()
            access_token = new_tokens["access_token"]
        else:
            raise Exception("Access token expired and no refresh token available")

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

        # Step 3: Check publish status with retry logic

        status_data = await _check_publish_status_with_retry(
            publish_id, access_token, max_retries=6, retry_delay=20
        )

        publish_status = status_data["data"]["status"]
        fail_reason = status_data["data"].get("fail_reason")

        if fail_reason:
            raise Exception(fail_reason)

        if publish_status == "FAILED":
            error_msg = (
                f"TikTok video upload failed: {fail_reason}"
                if fail_reason
                else "TikTok video upload failed for unknown reason"
            )
            raise Exception(error_msg)

        if publish_status not in ["PUBLISHED", "PROCESSING_UPLOAD", "PUBLISH_COMPLETE"]:
            raise Exception(f"Unexpected publish status: {publish_status}")

        # Check for the actual video ID from TikTok
        publicaly_available_post_id = status_data["data"].get(
            "publicaly_available_post_id", []
        )

        print(
            f"[DEBUG] Final URL construction - publicaly_available_post_id: {publicaly_available_post_id}"
        )
        print(
            f"[DEBUG] Username available: {tiktok_secrets.username if hasattr(tiktok_secrets, 'username') else 'None'}"
        )

        if publicaly_available_post_id and len(publicaly_available_post_id) > 0:
            # We have the actual video ID - construct the proper TikTok URL
            video_id = publicaly_available_post_id[0]
            print(f"[DEBUG] Got video ID: {video_id}")
            if tiktok_secrets.username:
                tiktok_url = f"https://www.tiktok.com/@{tiktok_secrets.username}/video/{video_id}"
                print(f"[DEBUG] Constructed URL with username: {tiktok_url}")
            else:
                # Fallback URL format without username
                tiktok_url = f"https://www.tiktok.com/video/{video_id}"
                print(f"[DEBUG] Constructed URL without username: {tiktok_url}")
        else:
            # Video is still processing or we don't have the ID yet
            # Get username from TikTok API first
            print(f"[DEBUG] No video ID available, getting username from TikTok API")
            username = await _get_tiktok_username(access_token)
            
            if not username:
                # Try to get from stored secrets as fallback
                username = getattr(tiktok_secrets, 'username', None)
                print(f"[DEBUG] Using stored username: {username}")
            
            # Try Display API as fallback
            print(f"[DEBUG] Trying Display API fallback with username: {username}")
            display_api_url = await _try_get_video_url_from_display_api(access_token, username)
            
            if display_api_url:
                tiktok_url = display_api_url
                print(f"[DEBUG] Using Display API URL: {tiktok_url}")
            else:
                # Final fallback - in prod with public posts, this should rarely be needed
                tiktok_url = (
                    f"https://www.tiktok.com/@{username}"
                    if username
                    else "https://www.tiktok.com"
                )
                print(f"[DEBUG] Using fallback URL (post likely private): {tiktok_url}")

        # Determine success message based on status
        if publish_status in ["PUBLISHED", "PUBLISH_COMPLETE"]:
            print("[DEBUG] TikTok post published successfully")
            print(f"[DEBUG] TikTok URL: {tiktok_url}")
            message = "TikTok post published successfully"
        elif publish_status == "PROCESSING_UPLOAD":
            print("[DEBUG] TikTok post is being processed")
            message = "TikTok post is being processed"
        else:
            print(f"[DEBUG] TikTok post status: {publish_status}")
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

        if not client_key or not client_secret:
            missing = []
            if not client_key:
                missing.append("TIKTOK_CLIENT_KEY")
            if not client_secret:
                missing.append("TIKTOK_CLIENT_SECRET")
            return {
                "error": f"Missing required environment variables: {', '.join(missing)}"
            }

        refresh_payload = {
            "client_key": client_key,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }

        response = requests.post(
            "https://open.tiktokapis.com/v2/oauth/token/",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=refresh_payload,
        )

        if not response.ok:
            try:
                error_data = response.json()
                error_msg = f"HTTP {response.status_code}: {error_data.get('error', 'Unknown error')}"
                if "error_description" in error_data:
                    error_msg += f" - {error_data['error_description']}"
                return {"error": error_msg}
            except:
                return {"error": f"HTTP {response.status_code}: {response.text}"}

        refresh_data = response.json()

        # Check if the response contains an error even with 200 status
        if "error" in refresh_data:
            error_msg = refresh_data["error"]
            if "error_description" in refresh_data:
                error_msg += f" - {refresh_data['error_description']}"
            return {"error": error_msg}

        return refresh_data

    except Exception as e:
        return {"error": f"Token refresh failed: {str(e)}"}


def _parse_bigint_json(response_text: str) -> dict:
    """
    Parse JSON response that may contain BigInt values.
    TikTok API returns post IDs as large integers that need special handling.
    """
    try:
        print(f"[DEBUG] Attempting to parse response with BigInt handling")
        
        # First, try to parse normally
        data = json.loads(response_text)
        
        # Check if publicaly_available_post_id exists and is empty/problematic
        if (isinstance(data, dict) and 
            "data" in data and 
            isinstance(data["data"], dict)):
            
            print(f"[DEBUG] Response has data section: {data['data']}")
            
            if "publicaly_available_post_id" in data["data"]:
                post_ids = data["data"]["publicaly_available_post_id"]
                print(f"[DEBUG] Current post IDs from JSON: {post_ids}")
                
                # If it's empty but we expect an ID, try to extract from raw text
                if not post_ids or (isinstance(post_ids, list) and len(post_ids) == 0):
                    print(f"[DEBUG] Post IDs empty, searching raw text for BigInt")
                    
                    # Look for various patterns of large numbers in the response
                    patterns = [
                        r'"publicaly_available_post_id":\s*\[([0-9]+)\]',
                        r'"publicaly_available_post_id":\s*\[\s*([0-9]+)\s*\]',
                        r'"publicaly_available_post_id":\[([0-9]+)\]'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, response_text)
                        if match:
                            post_id = int(match.group(1))
                            data["data"]["publicaly_available_post_id"] = [post_id]
                            print(f"[DEBUG] Extracted BigInt post ID using pattern '{pattern}': {post_id}")
                            break
                    else:
                        print(f"[DEBUG] No BigInt patterns found in response")
            else:
                print(f"[DEBUG] No publicaly_available_post_id field in response")
        
        return data
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON decode error: {e}")
        # Fallback to regular parsing
        return json.loads(response_text)


async def _get_tiktok_username(access_token: str) -> str:
    """
    Get the TikTok username from the user info API.
    """
    try:
        print(f"[DEBUG] Getting username from TikTok user info API")
        
        # Try the basic user info endpoint first
        user_response = requests.get(
            "https://open.tiktokapis.com/v2/user/info/",
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            params={
                "fields": "display_name,username"
            }
        )
        
        print(f"[DEBUG] User info response status: {user_response.status_code}")
        print(f"[DEBUG] User info response: {user_response.text}")
        
        if user_response.ok:
            user_data = user_response.json()
            if "data" in user_data and "user" in user_data["data"]:
                username = user_data["data"]["user"].get("username")
                if username:
                    print(f"[DEBUG] Got username from API: {username}")
                    return username
        elif user_response.status_code == 401:
            # User info scope not authorized - common in test/private setups
            print(f"[DEBUG] User info API unauthorized - likely private/test setup")
        
        print(f"[DEBUG] Failed to get username from any API")
        return None
        
    except Exception as e:
        print(f"[DEBUG] Error getting username: {str(e)}")
        return None


async def _try_get_video_url_from_display_api(access_token: str, username: str = None) -> str:
    """
    Fallback: Try to get the latest video URL from TikTok's Display API.
    This is used when the Content Posting API doesn't return the video ID.
    """
    try:
        if not username:
            print("[DEBUG] No username available for Display API fallback")
            return None
            
        print(f"[DEBUG] Trying Display API fallback for user: {username}")
        
        # Get user videos from Display API
        display_response = requests.get(
            "https://open.tiktokapis.com/v2/video/list/",
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            params={
                "fields": "id,share_url,create_time",
                "max_count": 5  # Get last 5 videos
            }
        )
        
        if display_response.ok:
            display_data = display_response.json()
            print(f"[DEBUG] Display API response: {display_data}")
            
            if "data" in display_data and "videos" in display_data["data"]:
                videos = display_data["data"]["videos"]
                if videos and len(videos) > 0:
                    # Get the most recent video (first in list)
                    latest_video = videos[0]
                    share_url = latest_video.get("share_url")
                    if share_url:
                        print(f"[DEBUG] Found latest video URL from Display API: {share_url}")
                        return share_url
        else:
            print(f"[DEBUG] Display API failed: {display_response.status_code} - {display_response.text}")
            
    except Exception as e:
        print(f"[DEBUG] Display API fallback failed: {str(e)}")
    
    return None


async def _check_publish_status_with_retry(
    publish_id: str, access_token: str, max_retries: int = 6, retry_delay: int = 20
) -> Dict[str, Any]:
    """
    Check publish status with retry logic to wait for video processing.
    TikTok videos can take 30 seconds to 2 minutes to process.
    """
    status_payload = {"publish_id": publish_id}

    for attempt in range(max_retries):
        try:
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

            print(f"[DEBUG] Raw status response text: {status_response.text}")
            
            # Use BigInt-aware parsing
            status_data = _parse_bigint_json(status_response.text)
            publish_status = status_data["data"]["status"]

            print(f"[DEBUG] Status check attempt {attempt + 1}: {publish_status}")
            print(f"[DEBUG] Parsed status response: {status_data}")

            # If we have a video ID or the process completed/failed, return immediately
            publicaly_available_post_id = status_data["data"].get(
                "publicaly_available_post_id", []
            )

            print(f"[DEBUG] publicaly_available_post_id: {publicaly_available_post_id}")

            if (
                publicaly_available_post_id and len(publicaly_available_post_id) > 0
            ) or publish_status in ["PUBLISH_COMPLETE", "FAILED"]:
                print(f"[DEBUG] Returning status data with video ID or final status")
                return status_data

            # If still processing and we have more retries, wait and try again
            if attempt < max_retries - 1 and publish_status == "PROCESSING_UPLOAD":
                print(
                    f"[DEBUG] Video still processing, waiting {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(retry_delay)
                continue

            # Return the status even if still processing after all retries
            print(f"[DEBUG] Max retries reached, returning current status")
            return status_data

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(
                f"[DEBUG] Status check failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            await asyncio.sleep(retry_delay)

    # This should not be reached, but just in case
    raise Exception("Max retries exceeded while checking publish status")
