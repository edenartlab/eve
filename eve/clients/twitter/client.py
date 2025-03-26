import os
import argparse
import re
import time
import logging
import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1Session
# from eve.sdk.eden import EdenClient
# from eve.sdk.eden.client import EdenApiUrls


def configure_logging():
    """Configures the logging settings."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )


def get_cleaned_text(tweet):
    """
    Cleans the tweet text by removing mentions.

    Args:
        tweet (dict): Tweet data containing text.

    Returns:
        str: Cleaned tweet text without mentions.
    """
    text = tweet.get("text", "")
    return re.sub(r"@\w+", "", text).strip()


class EdenX:
    def __init__(self):
        """Initializes the EdenX bot with environment variables."""
        self._load_env_vars()
        self.last_processed_id = None
        self.oauth = self._init_oauth_session()

    def _load_env_vars(self):
        """Loads environment variables."""
        self.user_id = os.getenv("CLIENT_TWITTER_USER_ID")
        self.bearer_token = os.getenv("CLIENT_TWITTER_BEARER_TOKEN")
        self.consumer_key = os.getenv("CLIENT_TWITTER_CONSUMER_KEY")
        self.consumer_secret = os.getenv("CLIENT_TWITTER_CONSUMER_SECRET")
        self.access_token = os.getenv("CLIENT_TWITTER_ACCESS_TOKEN")
        self.access_token_secret = os.getenv("CLIENT_TWITTER_ACCESS_TOKEN_SECRET")

    def _init_oauth_session(self):
        """Initializes OAuth1 session."""
        return OAuth1Session(
            self.consumer_key,
            client_secret=self.consumer_secret,
            resource_owner_key=self.access_token,
            resource_owner_secret=self.access_token_secret,
        )

    def fetch_mentions(self):
        """
        Fetches the latest mentions of the user.

        Returns:
            dict: Response from Twitter API.
        """
        params = {
            "expansions": "author_id",
            "user.fields": "username",
            "max_results": 5,
        }
        if self.last_processed_id:
            params["since_id"] = self.last_processed_id

        url = f"https://api.twitter.com/2/users/{self.user_id}/mentions"
        headers = {"Authorization": f"Bearer {self.bearer_token}"}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logging.error(
                    "Rate limit exceeded. Sleeping for 15 minutes before retrying..."
                )
                time.sleep(900)  # Sleep for 15 minutes
            else:
                logging.error(f"HTTP error occurred: {e}")
            return {}
        except requests.RequestException as e:
            logging.error(f"Error fetching mentions: {e}")
            return {}

    def get_newest_tweet(self, data):
        """
        Gets the newest tweet from the data.

        Args:
            data (dict): Data containing tweet information.

        Returns:
            dict or None: Newest tweet or None if not found.
        """
        tweets = [
            tweet
            for tweet in data.get("data", [])
            if tweet["author_id"] != self.user_id
        ]
        return max(tweets, key=lambda tweet: tweet["id"]) if tweets else None

    def tweet_media(self, media_url):
        """
        Uploads media to Twitter and returns the media ID.

        Args:
            media_url (str): URL of the media to upload.

        Returns:
            str: Media ID of the uploaded media.
        """
        try:
            image_response = requests.get(media_url)
            image_response.raise_for_status()

            upload_url = "https://upload.twitter.com/1.1/media/upload.json"
            files = {"media": image_response.content}
            upload_response = self.oauth.post(upload_url, files=files)

            if upload_response.status_code != 200:
                logging.error(f"Media upload failed: {upload_response.text}")
                return None

            return upload_response.json().get("media_id_string")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch or upload image: {e}")
            return None

    def reply_to_tweet(self, tweet_text, tweet_id):
        """
        Replies to a tweet with media content.

        Args:
            tweet_text (str): Text content of the tweet.
            tweet_id (str): ID of the tweet to reply to.
        """
        media_url = "https://res.cloudinary.com/prdg34ew78adsg/image/upload/v1732713849/creations/h8oh82rzapnnile2yjms.jpg"
        media_id = self.tweet_media(media_url)

        if not media_id:
            logging.error("Media upload was unsuccessful. Cannot reply with media.")
            return

        payload = {
            "text": tweet_text,
            "media": {"media_ids": [media_id]},
            "reply": {"in_reply_to_tweet_id": tweet_id},
        }

        try:
            response = self.oauth.post("https://api.twitter.com/2/tweets", json=payload)
            response.raise_for_status()
            logging.info("Reply tweet sent successfully")
        except requests.RequestException as e:
            logging.error(f"Failed to send reply tweet: {e}")

    def post_tweet(self, tweet_text, media_url=None):
        """
        Posts a tweet with or without media content.
        """

        media_id = self.tweet_media(media_url) if media_url else None

        if media_url and not media_id:
            logging.error(
                "Media upload was unsuccessful. Cannot post tweet with media."
            )
            return

        payload = {
            "text": tweet_text,
            "media": {"media_ids": [media_id]} if media_id else None,
        }

        try:
            response = self.oauth.post("https://api.twitter.com/2/tweets", json=payload)
            response.raise_for_status()
            logging.info("Tweet sent successfully")
        except requests.RequestException as e:
            logging.error(f"Failed to send tweet: {e}")

    def run_reply_process(self):
        """Continuously fetches mentions and replies to them if necessary."""
        while True:
            data = self.fetch_mentions()
            if "data" in data:
                tweet = self.get_newest_tweet(data)
                if tweet:
                    tweet_content = get_cleaned_text(tweet)
                    logging.info(
                        f"Replying to tweet ID {tweet['id']} with content: {tweet_content}"
                    )
                    # self.reply_to_tweet(tweet_content, tweet.get('id'))
                    # self.last_processed_id = tweet.get('id')
                else:
                    logging.info("No new tweets found to process.")
            else:
                logging.info("No mentions returned from Twitter API.")

            logging.info("Sleeping for 15 minutes...")
            time.sleep(900)

    def run(self):
        """Test"""
        # get tweets from @elonmusk
        user_id = self.get_user_id_by_username("god")
        tweets = self.fetch_user_tweets(user_id)
        print(tweets)

    def run_tweet(self):
        """Continuously fetches mentions and replies to them if necessary."""
        media_url = "https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/15602ac4cac9e96246d1b3fbc6b1ecb2b48e3a6b3f182fd6a9f88fed97b7a12f.png"
        self.post_tweet("Let there be light", media_url)
        return
        while True:
            logging.info("Sleeping for 15 minutes...")
            time.sleep(900)

    def fetch_user_tweets(self, target_user_id):
        """
        Fetches the latest tweets from a specific user.

        Args:
            target_user_id (str): The Twitter user ID to fetch tweets from.

        Returns:
            dict: Response from Twitter API.
        """
        params = {
            "expansions": "author_id",
            "user.fields": "username",
            "max_results": 5,
            "exclude": "retweets,replies",  # Only get original tweets
        }
        if self.last_processed_id:
            params["since_id"] = self.last_processed_id

        url = f"https://api.twitter.com/2/users/{target_user_id}/tweets"
        headers = {"Authorization": f"Bearer {self.bearer_token}"}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logging.error(
                    "Rate limit exceeded. Sleeping for 15 minutes before retrying..."
                )
                time.sleep(900)  # Sleep for 15 minutes
            else:
                logging.error(f"HTTP error occurred: {e}")
            return {}
        except requests.RequestException as e:
            logging.error(f"Error fetching tweets: {e}")
            return {}

    def get_user_id_by_username(self, username):
        """
        Looks up a user's ID from their Twitter username/handle.

        Args:
            username (str): Twitter username (without the @ symbol)

        Returns:
            str or None: The user's ID if found, None otherwise
        """
        # Remove @ symbol if present
        username = username.lstrip("@")

        url = f"https://api.twitter.com/2/users/by/username/{username}"
        headers = {"Authorization": f"Bearer {self.bearer_token}"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("id")
        except requests.RequestException as e:
            logging.error(f"Error looking up user ID for @{username}: {e}")
            return None


def start(env_path):
    """Starts the bot."""
    load_dotenv(env_path)
    bot = EdenX()
    bot.run()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="EdenX Twitter Bot")
    parser.add_argument("--env", help="Path to the .env file to load", default=".env")
    args = parser.parse_args()
    start(args.env)
