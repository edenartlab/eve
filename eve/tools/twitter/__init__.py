import logging
import time
from datetime import datetime, timedelta

import requests

from eve.agent.deployments import Deployment


class X:
    def __init__(self, deployment: Deployment):
        import os

        # OAuth 2.0 only mode
        self.access_token = deployment.secrets.twitter.access_token
        self.refresh_token = deployment.secrets.twitter.refresh_token
        self.twitter_id = deployment.secrets.twitter.twitter_id
        self.user_id = deployment.secrets.twitter.twitter_id  # For compatibility
        self.username = deployment.secrets.twitter.username

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

    def _make_request(self, method, url, **kwargs):
        """Makes a request to the Twitter API using OAuth 2.0."""
        # Always use bearer token authentication for OAuth 2.0
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"]["Authorization"] = f"Bearer {self.access_token}"

        if method.lower() == "get":
            response = requests.get(url, **kwargs)
        else:
            response = requests.post(url, **kwargs)

        if not response.ok:
            error_data = (
                response.json() if response.text else "No error details available"
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

    def _upload_image(self, content):
        """Upload image content to Twitter."""
        upload_response = self._make_request(
            "post",
            "https://upload.twitter.com/1.1/media/upload.json",
            files={"media": content},
        )
        return upload_response.json().get("media_id_string")

    def _upload_video(self, content):
        """Upload video content to Twitter using chunked upload."""

        # --- INIT ---
        init_response = self._make_request(
            "post",
            "https://upload.twitter.com/1.1/media/upload.json",
            data={
                "command": "INIT",
                "media_type": "video/mp4",
                "media_category": "tweet_video",
                "total_bytes": len(content),
            },
        )
        media_id = init_response.json().get("media_id_string")

        # --- APPEND ---
        chunk_size = 5 * 1024 * 1024  # 5MB chunks
        for i, start in enumerate(range(0, len(content), chunk_size)):
            chunk = content[start : start + chunk_size]

            self._make_request(
                "post",
                "https://upload.twitter.com/1.1/media/upload.json",
                data={"command": "APPEND", "media_id": media_id, "segment_index": i},
                files={"media": chunk},
            )

        # --- FINALIZE ---
        finalize_response = self._make_request(
            "post",
            "https://upload.twitter.com/1.1/media/upload.json",
            data={"command": "FINALIZE", "media_id": media_id},
        )

        response_json = finalize_response.json()
        processing_info = response_json.get("processing_info")

        if not processing_info:
            logging.debug("No processing_info, video may be ready immediately.")
            return media_id

        state = processing_info.get("state")
        while state in ["pending", "in_progress"]:
            check_after_secs = processing_info.get("check_after_secs", 5)
            logging.debug(
                f"Video still processing ({state}). Waiting {check_after_secs}s."
            )
            time.sleep(check_after_secs)

            status_response = self._make_request(
                "get",
                "https://upload.twitter.com/1.1/media/upload.json",
                params={"command": "STATUS", "media_id": media_id},
            )
            if not status_response:
                logging.error("STATUS check failed.")
                return None

            status_json = status_response.json()
            processing_info = status_json.get("processing_info")
            if not processing_info:
                # No further processing info => presumably done
                logging.debug("No processing_info in STATUS, assuming success.")
                return media_id

            state = processing_info.get("state")

        if state == "succeeded":
            logging.debug("Video upload succeeded!")
            return media_id
        else:
            logging.error(f"Video upload failed with state: {state}")
            return None

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

    def get_following222(self, usernames):
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
