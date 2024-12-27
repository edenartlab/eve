import time
import logging
import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1Session

from ...agent import Agent


class X:
    def __init__(self, agent: Agent):
        """Initializes an X/Twitter session."""
        
        self.user_id = agent.secrets["CLIENT_TWITTER_USER_ID"].get_secret_value()
        self.bearer_token = agent.secrets["CLIENT_TWITTER_BEARER_TOKEN"].get_secret_value()
        self.consumer_key = agent.secrets["CLIENT_TWITTER_CONSUMER_KEY"].get_secret_value()
        self.consumer_secret = agent.secrets["CLIENT_TWITTER_CONSUMER_SECRET"].get_secret_value()
        self.access_token = agent.secrets["CLIENT_TWITTER_ACCESS_TOKEN"].get_secret_value()
        self.access_token_secret = agent.secrets["CLIENT_TWITTER_ACCESS_TOKEN_SECRET"].get_secret_value()

        self.last_processed_id = None
        self.oauth = self._init_oauth_session()

    def _init_oauth_session(self):
        """Initializes OAuth1 session."""
        return OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=self.access_token,
            resource_owner_secret=self.access_token_secret
        )

    def _make_request(self, method, url, retries=1, oauth=True, **kwargs):
        """Generic request handler with error handling and retries."""
        logging.debug(f"[REQUEST] {method.upper()} {url}\nkwargs={kwargs}\n")
        try:
            if method.lower() == 'get':
                response = self.oauth.get(url, **kwargs) if oauth else requests.get(url, **kwargs)
            else:
                response = self.oauth.post(url, **kwargs) if oauth else requests.post(url, **kwargs)

            logging.debug(
                f"[RESPONSE] {response.status_code} {response.text}\n"
            )

            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error occurred: {e}")
            return None
        except requests.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None

    def fetch_mentions(self):
        """Fetches the latest mentions of the user."""
        params = {
            "expansions": "author_id",
            "user.fields": "username",
            "max_results": 5
        }
        if self.last_processed_id:
            params["since_id"] = self.last_processed_id

        response = self._make_request(
            'get',
            f"https://api.twitter.com/2/users/{self.user_id}/mentions",
            headers={"Authorization": f"Bearer {self.bearer_token}"},
            params=params
        )
        return response.json() if response else {}

    def get_newest_tweet(self, data):
        """Gets the newest tweet from the data."""
        
        tweets = [
            tweet for tweet in data.get("data", [])
            if tweet["author_id"] != self.user_id
        ]
        return max(tweets, key=lambda tweet: tweet["id"]) if tweets else None

    def tweet_media(self, media_url):
        """Uploads media to Twitter and returns the media ID."""
        # First, download the media
        image_response = self._make_request('get', media_url, oauth=False)
        if not image_response:
            logging.error(f"Failed to download media from: {media_url}")
            return None

        content = image_response.content
        is_video = media_url.lower().endswith('.mp4')

        if is_video:
            return self._upload_video(content)
        else:
            return self._upload_image(content)

    def _upload_image(self, content):
        """Upload image content to Twitter."""
        upload_response = self._make_request(
            'post',
            "https://upload.twitter.com/1.1/media/upload.json",
            files={"media": content}
        )
        if not upload_response:
            return None
        
        return upload_response.json().get("media_id_string")

    def _upload_video(self, content):
        """Upload video content to Twitter using chunked upload."""

        # --- INIT ---
        init_response = self._make_request(
            'post',
            "https://upload.twitter.com/1.1/media/upload.json",
            data={
                'command': 'INIT',
                'media_type': 'video/mp4',
                'media_category': 'tweet_video',
                'total_bytes': len(content)
            }
        )
        if not init_response:
            return None
        
        media_id = init_response.json().get('media_id_string')
        if not media_id:
            logging.error("No media_id_string found in INIT response.")
            return None
        
        # --- APPEND ---
        chunk_size = 5 * 1024 * 1024  # 5MB chunks
        for i, start in enumerate(range(0, len(content), chunk_size)):
            chunk = content[start:start + chunk_size]

            append_response = self._make_request(
                'post',
                "https://upload.twitter.com/1.1/media/upload.json",
                data={
                    'command': 'APPEND',
                    'media_id': media_id,
                    'segment_index': i
                },
                files={'media': chunk}
            )
            if not append_response:
                logging.error("APPEND step failed.")
                return None

        # --- FINALIZE ---
        finalize_response = self._make_request(
            'post',
            "https://upload.twitter.com/1.1/media/upload.json",
            data={
                'command': 'FINALIZE',
                'media_id': media_id
            }
        )
        if not finalize_response:
            logging.error("FINALIZE step failed.")
            return None

        response_json = finalize_response.json()
        processing_info = response_json.get("processing_info")

        # If there's no "processing_info", Twitter might have fully processed it already
        if not processing_info:
            logging.debug("No processing_info, video may be ready immediately.")
            return media_id

        # --- POLL for processing (only if state != succeeded) ---
        state = processing_info.get("state")
        while state in ["pending", "in_progress"]:
            check_after_secs = processing_info.get("check_after_secs", 5)
            logging.debug(f"Video still processing ({state}). Waiting {check_after_secs}s.")
            time.sleep(check_after_secs)
        
            status_response = self._make_request(
                'get',
                "https://upload.twitter.com/1.1/media/upload.json",
                params={
                    'command': 'STATUS',
                    'media_id': media_id
                }
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

    def post(self, tweet_text, media_urls=None, reply_to_tweet_id=None):
        """
        Posts a tweet with optional media, optionally as a reply to another tweet.

        :param tweet_text: The text of the tweet to post.
        :param media_url: (optional) A direct link to an image/MP4 to upload and attach.
        :param reply_to_tweet_id: (optional) The tweet ID to reply to.
        """
        media_ids = []
        media_urls = [media_urls] if isinstance(media_urls, str) else media_urls
        for media_url in media_urls:
            media_id = self.tweet_media(media_url)
            if not media_id:
                logging.error("Media upload was unsuccessful. Cannot post tweet with media.")
                return
            media_ids.append(media_id)

        payload = {"text": tweet_text}

        # If we have a successfully uploaded media_id, add it
        if media_ids:
            payload["media"] = {"media_ids": media_ids}

        # If we have a tweet_id to reply to, add that
        if reply_to_tweet_id:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to_tweet_id}

        response = self._make_request(
            'post',
            "https://api.twitter.com/2/tweets",
            json=payload
        )

        if response is not None:
            logging.info("Tweet sent successfully")
            response = response.json()
            assert "data" in response, f"No data in response: {response}"
            return response["data"]
        else:
            logging.error("Failed to post tweet: None response from _make_request.")
            raise Exception("Failed to post tweet. See logs for details.")
