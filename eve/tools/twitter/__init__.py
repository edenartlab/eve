import logging
import os
import time
from datetime import datetime, timedelta, timezone

import requests

from eve.agent.deployments import Deployment


class X:
    def __init__(self, deployment: Deployment):
        if os.getenv("DB") == "PROD":
            raise ValueError("Twitter integration is not available in PROD yet")

        # OAuth 2.0 only mode
        self.access_token = deployment.secrets.twitter.access_token
        self.refresh_token = deployment.secrets.twitter.refresh_token
        self.twitter_id = deployment.secrets.twitter.twitter_id
        self.user_id = deployment.secrets.twitter.twitter_id  # For compatibility
        self.username = deployment.secrets.twitter.username
        self.deployment = deployment  # Store for OAuth 1.0a token lookup

        # Get app credentials from environment
        self.consumer_key = os.getenv("TWITTER_INTEGRATIONS_CLIENT_ID")
        self.consumer_secret = os.getenv("TWITTER_INTEGRATIONS_CLIENT_SECRET")
        self.bearer_token = (
            self.access_token
        )  # OAuth 2.0 access token can be used as bearer

        if not all([self.access_token, self.twitter_id, self.username]):
            raise ValueError("Missing required OAuth 2.0 credentials")

        logging.info(
            f"Initialized X client for user: @{self.username} (ID: {self.twitter_id})\n"
            f"Using OAuth 2.0 access token: {self.access_token[:8]}..."
        )

    def _refresh_token(self):
        """Refresh the OAuth 2.0 access token with race condition protection."""
        from eve.agent.session.models import Deployment
        from eve.mongo import MongoDocumentNotFound

        if os.getenv("DB") == "PROD":
            raise ValueError("Twitter integration is not available in PROD yet")

        logging.info(
            f"Refreshing token for deployment {self.deployment.id} (@{self.username})"
        )

        # Reload deployment from DB using stored deployment ID
        try:
            deployment = Deployment.from_mongo(self.deployment.id)
        except MongoDocumentNotFound:
            logging.error(
                f"Deployment {self.deployment.id} not found in DB, cannot refresh token"
            )
            raise ValueError(
                f"Deployment {self.deployment.id} not found - may need to reconnect Twitter account"
            )

        if not deployment:
            raise ValueError(
                f"Deployment {self.deployment.id} not found for token refresh"
            )

        # Check if token was recently refreshed by another process
        if deployment.secrets.twitter.expires_at:
            time_until_expiry = (
                deployment.secrets.twitter.expires_at - datetime.now(timezone.utc)
            ).total_seconds()
            if time_until_expiry > 60:  # Token still valid for >1 minute
                logging.info(
                    f"Token was recently refreshed (expires in {time_until_expiry:.0f}s), using existing"
                )
                self.access_token = deployment.secrets.twitter.access_token
                self.refresh_token = deployment.secrets.twitter.refresh_token
                return

        # Use latest refresh_token from DB (not potentially stale in-memory copy)
        refresh_token = deployment.secrets.twitter.refresh_token or self.refresh_token
        if not refresh_token:
            raise ValueError("No refresh token available")

        logging.info("Refreshing Twitter OAuth 2.0 token...")

        token_url = "https://api.twitter.com/2/oauth2/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.consumer_key,
        }

        response = requests.post(
            token_url, data=data, auth=(self.consumer_key, self.consumer_secret)
        )

        if not response.ok:
            logging.error(
                f"Token refresh failed: {response.status_code} {response.text}"
            )
            response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data["access_token"]

        # Update refresh token if provided (Twitter uses rotating refresh tokens)
        if "refresh_token" in token_data:
            self.refresh_token = token_data["refresh_token"]

        # Calculate expiry
        expires_in = token_data.get("expires_in", 7200)  # Default 2 hours
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

        # Reload deployment again before saving (prevent overwriting concurrent updates)
        deployment = Deployment.from_mongo(self.deployment.id)

        if deployment:
            deployment.secrets.twitter.access_token = self.access_token
            deployment.secrets.twitter.refresh_token = self.refresh_token
            deployment.secrets.twitter.expires_at = expires_at
            deployment.save()
            logging.info(f"Token refreshed successfully, expires at {expires_at}")
        else:
            logging.warning("Could not find deployment to update tokens")

    def _make_request(self, method, url, **kwargs):
        """Makes a request to the Twitter API using OAuth 2.0."""

        if os.getenv("DB") == "PROD":
            raise ValueError("Twitter integration is not available in PROD yet")

        # Always use bearer token authentication for OAuth 2.0
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"]["Authorization"] = f"Bearer {self.access_token}"

        if method.lower() == "get":
            response = requests.get(url, **kwargs)
        else:
            response = requests.post(url, **kwargs)

        # If 401 Unauthorized, try refreshing token once
        if response.status_code == 401 and self.refresh_token:
            logging.warning("Got 401 Unauthorized, attempting token refresh...")
            try:
                self._refresh_token()
                # Retry request with new token
                kwargs["headers"]["Authorization"] = f"Bearer {self.access_token}"
                if method.lower() == "get":
                    response = requests.get(url, **kwargs)
                else:
                    response = requests.post(url, **kwargs)
            except Exception as e:
                logging.error(f"Token refresh failed: {e}")

        if not response.ok:
            # Try to parse error response as JSON, but handle cases where it's not valid JSON
            try:
                error_data = (
                    response.json() if response.text else "No error details available"
                )
            except Exception:
                error_data = (
                    response.text if response.text else "No error details available"
                )

            logging.error(
                f"Twitter API Error:\n"
                f"Status Code: {response.status_code}\n"
                f"URL: {url}\n"
                f"Method: {method.upper()}\n"
                f"Error Data: {error_data}\n"
            )
            response.raise_for_status()

        return response

    def fetch_mentions(self, start_time=None):
        """Fetches mentions for the user."""

        params = {
            "max_results": 100,
            "tweet.fields": "created_at,author_id,conversation_id",
            "expansions": "author_id",
            "user.fields": "username,name,public_metrics,verified",
        }

        if start_time:
            if isinstance(start_time, datetime):
                start_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["start_time"] = start_time

        response = self._make_request(
            "get",
            f"https://api.twitter.com/2/users/{self.user_id}/mentions",
            params=params,
        )
        return response.json()

    def fetch_followings(self):
        """Fetches the latest followings of the user."""
        response = self._make_request(
            "get",
            f"https://api.twitter.com/2/users/{self.user_id}/following",
        )
        return response.json()

    def get_newest_tweet(self, data):
        """Gets the newest tweet from the data."""

        tweets = [
            tweet
            for tweet in data.get("data", [])
            if tweet["author_id"] != self.user_id
        ]
        return max(tweets, key=lambda tweet: tweet["id"]) if tweets else None

    # ========================================================================
    # Media Upload Helper Methods
    # ========================================================================

    def _detect_media_type(self, content: bytes) -> str:
        """Detect MIME type from file content."""
        # Check magic bytes for common formats
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        elif content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
            return "image/gif"
        elif content.startswith(b"RIFF") and b"WEBP" in content[:20]:
            return "image/webp"
        elif content[4:12] == b"ftypmp42" or content[4:12] == b"ftypisom":
            return "video/mp4"
        else:
            # Default to jpeg for images
            return "image/jpeg"

    def _get_media_category(self, media_type: str, is_gif: bool = False) -> str:
        """Map MIME type to X API media category."""
        if media_type.startswith("video/"):
            return "tweet_video"
        elif is_gif or media_type == "image/gif":
            return "tweet_gif"
        else:
            return "tweet_image"

    def _chunk_data(self, content: bytes, chunk_size: int = 1024 * 1024):
        """Generator that yields chunks of data."""
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    def _poll_processing_status(self, media_id: str, max_wait_seconds: int = 300):
        """
        Poll media processing status until complete or timeout.

        Args:
            media_id: The media ID to check status for
            max_wait_seconds: Maximum time to wait for processing (default 5 minutes)

        Returns:
            True if processing succeeded, False if failed or timed out
        """
        start_time = time.time()
        wait_time = 1  # Start with 1 second

        logging.info(f"Polling processing status for media {media_id}")

        while (time.time() - start_time) < max_wait_seconds:
            try:
                status_response = self._make_request(
                    "get",
                    f"https://api.x.com/2/media/upload/{media_id}/status",
                )

                status_data = status_response.json().get("data", {})
                processing_info = status_data.get("processing_info", {})
                state = processing_info.get("state")

                logging.debug(f"Processing state: {state}")

                if state == "succeeded":
                    logging.info(f"Media processing succeeded for {media_id}")
                    return True
                elif state == "failed":
                    error_msg = processing_info.get("error", {}).get(
                        "message", "Unknown error"
                    )
                    logging.error(f"Media processing failed: {error_msg}")
                    return False
                elif state in ["pending", "in_progress"]:
                    # Use check_after_secs if provided, otherwise exponential backoff
                    check_after = processing_info.get("check_after_secs", wait_time)
                    logging.debug(f"Still processing, waiting {check_after}s...")
                    time.sleep(check_after)
                    # Exponential backoff: 1s -> 2s -> 5s -> 10s -> 10s...
                    wait_time = min(wait_time * 2, 10)
                else:
                    logging.warning(f"Unknown processing state: {state}")
                    return False

            except Exception as e:
                logging.error(f"Error polling status: {e}")
                return False

        logging.error(f"Processing timeout after {max_wait_seconds}s")
        return False

    def _upload_one_shot(
        self, content: bytes, media_type: str, media_category: str
    ) -> str:
        """
        Upload media using one-shot endpoint (for small images/subtitles).

        Args:
            content: Binary content of the media file
            media_type: MIME type (e.g., "image/jpeg")
            media_category: X API category (e.g., "tweet_image")

        Returns:
            media_id string for use in tweets
        """
        logging.info(
            f"Uploading media via one-shot (size: {len(content)} bytes, type: {media_type}, category: {media_category})"
        )

        try:
            response = self._make_request(
                "post",
                "https://api.x.com/2/media/upload",
                files={"media": content},
                data={
                    "media_type": media_type,
                    "media_category": media_category,
                },
            )

            data = response.json().get("data", {})
            media_id = data.get("id")

            if not media_id:
                raise Exception("No media ID returned from upload")

            logging.info(f"One-shot upload successful: {media_id}")
            return media_id

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logging.error(
                    "Media upload failed with 403 Forbidden. "
                    "Ensure the OAuth token has 'media.write' scope."
                )
            raise

    def _upload_chunked(
        self, content: bytes, media_type: str, media_category: str
    ) -> str:
        """
        Upload media using chunked endpoint (for videos, GIFs, large images).

        Args:
            content: Binary content of the media file
            media_type: MIME type (e.g., "video/mp4")
            media_category: X API category ("tweet_video", "tweet_image", "tweet_gif")

        Returns:
            media_id string for use in tweets
        """
        total_bytes = len(content)
        logging.info(
            f"Uploading media via chunked upload "
            f"(size: {total_bytes} bytes, type: {media_type}, category: {media_category})"
        )

        # ===== INITIALIZE =====
        try:
            init_response = self._make_request(
                "post",
                "https://api.x.com/2/media/upload/initialize",
                json={
                    "media_type": media_type,
                    "total_bytes": total_bytes,
                    "media_category": media_category,
                },
            )

            init_data = init_response.json().get("data", {})
            media_id = init_data.get("id")
            expires_after = init_data.get("expires_after_secs", 86400)

            if not media_id:
                raise Exception("No media ID returned from INITIALIZE")

            logging.info(
                f"INITIALIZE successful: media_id={media_id}, expires_in={expires_after}s"
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logging.error(
                    "Media upload INITIALIZE failed with 403 Forbidden. "
                    "Ensure the OAuth token has 'media.write' scope."
                )
            raise

        # ===== APPEND =====
        chunk_size = 1024 * 1024  # 1MB chunks (X API maximum)
        chunks = list(self._chunk_data(content, chunk_size))
        total_chunks = len(chunks)

        logging.info(f"Uploading {total_chunks} chunks...")

        for segment_index, chunk in enumerate(chunks):
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    self._make_request(
                        "post",
                        f"https://api.x.com/2/media/upload/{media_id}/append",
                        data={"segment_index": segment_index},
                        files={"media": chunk},
                    )

                    logging.debug(
                        f"APPEND {segment_index + 1}/{total_chunks} successful"
                    )
                    break  # Success, move to next chunk

                except requests.exceptions.HTTPError:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(f"APPEND failed after {max_retries} retries")
                        raise

                    # Exponential backoff for retries
                    wait_time = 2**retry_count
                    logging.warning(
                        f"APPEND failed, retrying in {wait_time}s (attempt {retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)

        logging.info(f"All {total_chunks} chunks uploaded successfully")

        # ===== FINALIZE =====
        try:
            finalize_response = self._make_request(
                "post", f"https://api.x.com/2/media/upload/{media_id}/finalize"
            )

            finalize_data = finalize_response.json().get("data", {})
            processing_info = finalize_data.get("processing_info")

            logging.info(f"FINALIZE successful for media_id={media_id}")

            # If processing is required, poll for completion
            if processing_info:
                state = processing_info.get("state")
                logging.info(f"Media requires async processing (state: {state})")

                if not self._poll_processing_status(media_id):
                    raise Exception("Media processing failed or timed out")

            return media_id

        except requests.exceptions.HTTPError as e:
            logging.error(f"FINALIZE failed: {e}")
            raise

    # ========================================================================
    # Media Upload Methods
    # ========================================================================

    def tweet_media(self, media_url):
        """Uploads media to Twitter and returns the media ID."""
        # First, download the media (don't use Twitter auth for external URL)
        image_response = requests.get(media_url)
        if not image_response.ok:
            logging.error(f"Failed to download media from: {media_url}")
            return None

        content = image_response.content
        is_video = media_url.lower().endswith(".mp4")

        if is_video:
            return self._upload_video(content)
        else:
            return self._upload_image(content)

    def _upload_image(self, content: bytes) -> str:
        """
        Upload image content to X using v2 API with OAuth 2.0.

        Automatically routes to one-shot (simple) or chunked upload based on file size.
        - Images < 5MB: One-shot upload
        - Images >= 5MB or GIFs > 5MB: Chunked upload

        Args:
            content: Binary content of the image file

        Returns:
            media_id string for use in tweets
        """
        file_size_mb = len(content) / (1024 * 1024)
        media_type = self._detect_media_type(content)
        is_gif = media_type == "image/gif"

        logging.info(f"Uploading image: size={file_size_mb:.2f}MB, type={media_type}")

        # Validate file size limits
        if is_gif and file_size_mb > 15:
            raise ValueError(
                f"GIF file too large ({file_size_mb:.2f}MB). Maximum is 15MB."
            )
        elif not is_gif and file_size_mb > 5:
            raise ValueError(
                f"Image file too large ({file_size_mb:.2f}MB). Maximum is 5MB."
            )

        # Get media category for both upload paths
        media_category = self._get_media_category(media_type, is_gif)

        # Route to appropriate upload method
        # Use one-shot for small images, chunked for large images/GIFs
        if file_size_mb < 5 and not is_gif:
            # Small image: use simple one-shot upload
            return self._upload_one_shot(content, media_type, media_category)
        else:
            # Large image or GIF: use chunked upload
            return self._upload_chunked(content, media_type, media_category)

    def _upload_video(self, content: bytes) -> str:
        """
        Upload video content to X using v2 API with OAuth 2.0.

        Videos always use chunked upload with async processing.

        Args:
            content: Binary content of the video file

        Returns:
            media_id string for use in tweets
        """
        file_size_mb = len(content) / (1024 * 1024)
        media_type = self._detect_media_type(content)

        logging.info(f"Uploading video: size={file_size_mb:.2f}MB, type={media_type}")

        # Validate file size limit
        if file_size_mb > 512:
            raise ValueError(
                f"Video file too large ({file_size_mb:.2f}MB). Maximum is 512MB."
            )

        # Videos always use chunked upload with tweet_video category
        return self._upload_chunked(content, media_type, "tweet_video")

    def post(self, text: str, media_ids: list[str] = None, reply: str = None):
        """Posts a tweet or reply."""
        json = {"text": text}
        if media_ids:
            json["media"] = {"media_ids": media_ids}
        if reply:
            json["reply"] = {"in_reply_to_tweet_id": reply}
        response = self._make_request(
            "post",
            "https://api.twitter.com/2/tweets",
            json=json,
        )
        return response.json()

    def get_tweet(self, tweet_id: str) -> dict:
        """Fetch a single tweet's details including conversation_id."""
        response = self._make_request(
            "get",
            f"https://api.twitter.com/2/tweets/{tweet_id}",
            params={
                "tweet.fields": "conversation_id,author_id,created_at,in_reply_to_user_id,referenced_tweets",
            },
        )
        return response.json()

    def get_following(self, usernames):
        """Fetches the list of accounts each specified username is following."""
        following_data = {}

        for username in usernames:
            response = self._make_request(
                "get",
                f"https://api.twitter.com/2/users/by/username/{username}",
            )

            if not response:
                logging.error(f"Failed to fetch user info for {username}.")
                following_data[username] = []
                continue

            user_id = response.json().get("data", {}).get("id")

            if not user_id:
                logging.error(f"User ID not found for {username}.")
                following_data[username] = []
                continue

            follows_response = self._make_request(
                "get",
                f"https://api.twitter.com/2/users/{user_id}/following",
                params={"max_results": 1000},  # Adjust as needed for pagination.
            )

            if follows_response:
                following_data[username] = [
                    follow.get("username")
                    for follow in follows_response.json().get("data", [])
                ]
            else:
                following_data[username] = []

        return following_data

    def get_recent_tweets(self, usernames, timeframe_minutes=60):
        """Fetches tweets from the given users within the specified timeframe."""
        recent_tweets = {}
        time_threshold = datetime.utcnow() - timedelta(minutes=timeframe_minutes)

        for username in usernames:
            response = self._make_request(
                "get",
                f"https://api.twitter.com/2/users/by/username/{username}",
            )

            if not response:
                logging.error(f"Failed to fetch user info for {username}.")
                recent_tweets[username] = []
                continue

            user_id = response.json().get("data", {}).get("id")

            if not user_id:
                logging.error(f"User ID not found for {username}.")
                recent_tweets[username] = []
                continue

            tweets_response = self._make_request(
                "get",
                f"https://api.twitter.com/2/users/{user_id}/tweets",
                params={"max_results": 100, "tweet.fields": "created_at"},
            )

            if not tweets_response:
                logging.error(f"Failed to fetch tweets for {username}.")
                recent_tweets[username] = []
                continue

            tweets = tweets_response.json().get("data", [])
            recent_tweets[username] = [
                tweet
                for tweet in tweets
                if datetime.strptime(tweet["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
                >= time_threshold
            ]

        return recent_tweets

    def get_all_followings(self, user_ids, max_results=1000):
        """
        Retrieves all followings for each user in user_ids using the Twitter v2 endpoint.
        Returns a dict mapping user_id -> list_of_following_users.

        Note: Each user requires its own call. If a user has a large following list,
              we will paginate until we've retrieved them all, using 'next_token'.
        """
        url_template = "https://api.twitter.com/2/users/{}/following"
        all_followings = {}

        for user_id in user_ids:
            followings = []
            pagination_token = None

            while True:
                params = {"max_results": max_results}
                if pagination_token:
                    params["pagination_token"] = pagination_token

                # Use _make_request with Bearer token
                # (the v2 'following' endpoint uses Bearer token).
                response = self._make_request(
                    "get",  # This should be GET, not POST
                    url_template.format(user_id),
                    params=params,
                )

                if not response:
                    logging.error(f"Error fetching followings for user {user_id}")
                    break

                data = response.json()
                # If Twitter returns an error structure, log it
                if "errors" in data:
                    logging.error(
                        f"Error fetching followings for user {user_id}: {data}"
                    )
                    break

                followings_page = data.get("data", [])
                followings.extend(followings_page)

                meta = data.get("meta", {})
                pagination_token = meta.get("next_token")
                if not pagination_token:
                    break

            # Store all followings for this user
            all_followings[user_id] = followings

        return all_followings

    def get_user_by_username(self, username: str):
        """Gets user info by username."""
        response = self._make_request(
            "get",
            f"https://api.twitter.com/2/users/by/username/{username}",
        )
        return response.json()

    def get_user_tweets(self, user_id: str, max_results=100):
        """Gets tweets for a user."""
        response = self._make_request(
            "get",
            f"https://api.twitter.com/2/users/{user_id}/tweets",
            params={"max_results": max_results, "tweet.fields": "created_at"},
        )
        return response.json()

    def get_user_following(self, user_id: str, max_results=1000):
        """Gets list of users that a user follows."""
        response = self._make_request(
            "get",
            f"https://api.twitter.com/2/users/{user_id}/following",
            params={"max_results": max_results},
        )
        return response.json()
