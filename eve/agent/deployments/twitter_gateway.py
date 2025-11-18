#!/usr/bin/env python3
"""
Twitter Account Tracking Service

Continuously polls Twitter API to track mentions, replies, and quotes for specified accounts.
Saves all data to MongoDB and downloads media to local disk.

Usage:
  DB=PROD python twitter_tracker.py --deployment-id <id>

Prerequisites:
- Connect Twitter account via staging.app.eden.art to get deployment ID
- Set DB environment variable (STAGE or PROD)
- Ensure MongoDB is accessible
"""

import eve
import argparse
import time
import hashlib
import requests
from loguru import logger
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from bson import ObjectId
from pydantic import Field

from eve.agent.session.models import Deployment
from eve.tools.twitter import X
from eve.mongo import Document, Collection


# ============================================================================
# CONFIGURATION
# ============================================================================

# API field and expansion configuration (from PLAN.md Section 2)
TWEET_FIELDS = "id,text,created_at,author_id,conversation_id,in_reply_to_user_id,referenced_tweets,entities,attachments,lang,possibly_sensitive,public_metrics,source,reply_settings,edit_history_tweet_ids"
USER_FIELDS = "id,username,name,verified,verified_type,profile_image_url,public_metrics,protected,created_at,description,url,location"
MEDIA_FIELDS = "media_key,type,url,preview_image_url,width,height,duration_ms,alt_text,public_metrics,variants"
EXPANSIONS = "author_id,attachments.media_keys,referenced_tweets.id,referenced_tweets.id.author_id,entities.mentions.username"

# Polling and pagination configuration
POLL_INTERVAL_SECONDS = 60
MAX_PAGES_PER_QUERY = 5
MAX_RESULTS_PER_PAGE = 100
MEDIA_DOWNLOAD_DIR = Path("./x-media")
MAX_MEDIA_DOWNLOAD_ATTEMPTS = 3

# Rate limiting
HARD_CAP_CALLS_PER_LOOP = 20


# ============================================================================
# MONGODB MODELS (Section 4)
# ============================================================================

@Collection("X_tracked_handles")
class TrackedHandle(Document):
    username: str
    user_id: Optional[str] = None
    active: bool = True
    created_at: datetime = None
    updated_at: datetime = None

    def __init__(self, **data):
        if 'created_at' not in data:
            data['created_at'] = datetime.now(timezone.utc)
        if 'updated_at' not in data:
            data['updated_at'] = datetime.now(timezone.utc)
        super().__init__(**data)


@Collection("X_tweets")
class Tweet(Document):
    id: Optional[str] = Field(None, alias="_id")  # Override to use string instead of ObjectId
    author_id: str
    text: str
    created_at: datetime
    conversation_id: Optional[str] = None
    in_reply_to_user_id: Optional[str] = None
    referenced_tweets: Optional[List[Dict[str, Any]]] = None
    entities: Optional[Dict[str, Any]] = None
    attachments: Optional[Dict[str, Any]] = None
    lang: Optional[str] = None
    possibly_sensitive: Optional[bool] = None
    source: Optional[str] = None
    reply_settings: Optional[str] = None
    public_metrics: Optional[Dict[str, int]] = None
    raw: Dict[str, Any]
    first_seen_at: datetime = None
    last_seen_at: datetime = None
    processed: Optional[bool] = False  # Track if tweet has been processed by agent

    def __init__(self, **data):
        if 'first_seen_at' not in data:
            data['first_seen_at'] = datetime.now(timezone.utc)
        if 'last_seen_at' not in data:
            data['last_seen_at'] = datetime.now(timezone.utc)
        super().__init__(**data)


@Collection("X_users")
class User(Document):
    id: Optional[str] = Field(None, alias="_id")  # Override to use string instead of ObjectId
    username: str
    name: str
    verified: Optional[bool] = None
    verified_type: Optional[str] = None
    protected: Optional[bool] = None
    profile_image_url: Optional[str] = None
    public_metrics: Optional[Dict[str, int]] = None
    bio_snapshot: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    last_updated_at: datetime = None
    raw: Dict[str, Any] = None

    def __init__(self, **data):
        if 'last_updated_at' not in data:
            data['last_updated_at'] = datetime.now(timezone.utc)
        super().__init__(**data)


@Collection("X_media")
class Media(Document):
    id: Optional[str] = Field(None, alias="_id")  # Override to use string instead of ObjectId
    type: str
    url: Optional[str] = None
    preview_image_url: Optional[str] = None
    variants: Optional[List[Dict[str, Any]]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_ms: Optional[int] = None
    alt_text: Optional[str] = None
    public_metrics: Optional[Dict[str, int]] = None
    download: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = None

    def __init__(self, **data):
        if 'download' not in data:
            data['download'] = {
                'status': 'pending',
                'stored_path': None,
                'byte_size': None,
                'checksum': None,
                'last_attempt_at': None,
                'last_success_at': None,
                'attempts': 0
            }
        super().__init__(**data)


@Collection("X_state")
class State(Document):
    id: Optional[str] = Field(None, alias="_id")  # Override to use string instead of ObjectId
    value: Any
    updated_at: datetime = None

    def __init__(self, **data):
        if 'updated_at' not in data:
            data['updated_at'] = datetime.now(timezone.utc)
        super().__init__(**data)


# ============================================================================
# TWITTER API WRAPPERS (Section 11)
# ============================================================================

class TwitterAPIWrapper:
    """Wrapper for Twitter API using Eve's X client."""

    def __init__(self, deployment_id: str):
        self.deployment_id = deployment_id
        self.deployment = Deployment.load(_id=ObjectId(deployment_id))
        if not self.deployment:
            raise Exception(f"Deployment not found: {deployment_id}")
        self.twitter = X(self.deployment)
        self.calls_this_loop = 0

    def refresh_access_token(self):
        """Refresh the OAuth 2.0 access token using the refresh token."""
        import os

        client_id = os.getenv("TWITTER_INTEGRATIONS_CLIENT_ID")
        client_secret = os.getenv("TWITTER_INTEGRATIONS_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise Exception("TWITTER_INTEGRATIONS_CLIENT_ID and TWITTER_INTEGRATIONS_CLIENT_SECRET must be set")

        logger.info("🔄 Refreshing access token...")

        # Make token refresh request
        token_url = "https://api.twitter.com/2/oauth2/token"
        auth = (client_id, client_secret)
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.twitter.refresh_token
        }

        response = requests.post(token_url, auth=auth, data=data)
        response.raise_for_status()

        token_data = response.json()
        new_access_token = token_data["access_token"]
        new_refresh_token = token_data.get("refresh_token", self.twitter.refresh_token)

        # Update deployment with new tokens
        self.deployment.secrets.twitter.access_token = new_access_token
        self.deployment.secrets.twitter.refresh_token = new_refresh_token
        self.deployment.save()

        # Update the twitter client
        self.twitter.access_token = new_access_token
        self.twitter.refresh_token = new_refresh_token
        self.twitter.bearer_token = new_access_token

        logger.info(f"✅ Token refreshed successfully: {new_access_token[:8]}...")

        return new_access_token

    def reset_call_counter(self):
        """Reset the call counter for a new polling loop."""
        self.calls_this_loop = 0

    def _make_api_call(self, method: str, url: str, params: Dict[str, Any], retry_on_401: bool = True) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Make an API call and return response data + rate limit headers.
        Automatically refreshes token on 401 Unauthorized errors.

        Returns:
            Tuple of (response_json, rate_limit_headers)
        """
        if self.calls_this_loop >= HARD_CAP_CALLS_PER_LOOP:
            raise Exception(f"Hard cap of {HARD_CAP_CALLS_PER_LOOP} calls per loop reached")

        self.calls_this_loop += 1

        try:
            response = self.twitter._make_request(method, url, params=params)
        except requests.exceptions.HTTPError as e:
            # If 401 Unauthorized and we haven't retried yet, refresh token and retry
            if e.response.status_code == 401 and retry_on_401:
                logger.info("⚠️  401 Unauthorized - attempting to refresh token...")
                self.refresh_access_token()
                # Retry once with refreshed token (set retry_on_401=False to prevent infinite loop)
                return self._make_api_call(method, url, params, retry_on_401=False)
            else:
                raise

        # Extract rate limit headers
        rate_limit_headers = {
            'remaining': response.headers.get('x-rate-limit-remaining'),
            'reset': response.headers.get('x-rate-limit-reset'),
            'limit': response.headers.get('x-rate-limit-limit'),
        }

        return response.json(), rate_limit_headers

    def fetch_search_recent(
        self,
        query: str,
        since_id: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: int = MAX_RESULTS_PER_PAGE
    ) -> Dict[str, Any]:
        """
        Fetch recent tweets matching a search query.

        Args:
            query: Twitter search query
            since_id: Return tweets after this ID
            next_token: Pagination token
            max_results: Max results per page (up to 100)

        Returns:
            API response with data, includes, and meta
        """
        params = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": TWEET_FIELDS,
            "user.fields": USER_FIELDS,
            "media.fields": MEDIA_FIELDS,
            "expansions": EXPANSIONS,
        }

        if since_id:
            params["since_id"] = since_id
        if next_token:
            params["next_token"] = next_token

        response_data, rate_headers = self._make_api_call(
            "get",
            "https://api.twitter.com/2/tweets/search/recent",
            params=params
        )

        # Attach rate limit info
        response_data["_rate_limit"] = rate_headers
        return response_data

    def fetch_users_by_usernames(self, usernames: List[str]) -> Dict[str, Any]:
        """
        Fetch user objects by usernames.

        Args:
            usernames: List of usernames to fetch

        Returns:
            API response with user data
        """
        params = {
            "usernames": ",".join(usernames),
            "user.fields": USER_FIELDS,
        }

        response_data, rate_headers = self._make_api_call(
            "get",
            "https://api.twitter.com/2/users/by",
            params=params
        )

        response_data["_rate_limit"] = rate_headers
        return response_data

    def fetch_quote_tweets(
        self,
        tweet_id: str,
        next_token: Optional[str] = None,
        max_results: int = MAX_RESULTS_PER_PAGE
    ) -> Dict[str, Any]:
        """
        Fetch quote tweets for a specific tweet.

        Args:
            tweet_id: The tweet ID to get quotes for
            next_token: Pagination token
            max_results: Max results per page (up to 100)

        Returns:
            API response with quote tweet data
        """
        params = {
            "max_results": min(max_results, 100),
            "tweet.fields": TWEET_FIELDS,
            "user.fields": USER_FIELDS,
            "media.fields": MEDIA_FIELDS,
            "expansions": EXPANSIONS,
        }

        if next_token:
            params["next_token"] = next_token

        response_data, rate_headers = self._make_api_call(
            "get",
            f"https://api.twitter.com/2/tweets/{tweet_id}/quote_tweets",
            params=params
        )

        response_data["_rate_limit"] = rate_headers
        return response_data

    def check_rate_limit_and_sleep(self, rate_limit_headers: Dict[str, str]):
        """
        Check rate limit headers and sleep if needed.

        Args:
            rate_limit_headers: Headers from API response
        """
        remaining = rate_limit_headers.get('remaining')
        reset = rate_limit_headers.get('reset')

        if remaining is not None and int(remaining) == 0 and reset is not None:
            reset_time = datetime.fromtimestamp(int(reset), tz=timezone.utc)
            now = datetime.now(timezone.utc)
            sleep_seconds = (reset_time - now).total_seconds() + 5  # Add 5s jitter

            if sleep_seconds > 0:
                logger.info(f"⏱️  Rate limit reached. Sleeping {sleep_seconds:.0f}s until reset...")
                time.sleep(sleep_seconds)


# ============================================================================
# QUERY CONSTRUCTION (Section 8)
# ============================================================================

def build_combined_query(usernames: List[str]) -> str:
    """
    Build a combined query to find mentions, replies, and tweets from tracked accounts.

    Example: (@alice OR to:alice OR from:alice OR @bob OR to:bob OR from:bob) -is:retweet

    This combines what were previously two separate queries:
    1. Mentions/replies: (@alice OR to:alice)
    2. From users: (from:alice)
    """
    parts = []
    for username in usernames:
        parts.append(f"@{username}")
        parts.append(f"to:{username}")
        parts.append(f"from:{username}")

    query = "(" + " OR ".join(parts) + ") -is:retweet"
    return query


# ============================================================================
# DATA PERSISTENCE (Section 4)
# ============================================================================

class DataPersister:
    """Handles upserting data to MongoDB."""

    @staticmethod
    def upsert_tweets(response_data: Dict[str, Any]) -> List[str]:
        """
        Upsert tweets from API response.

        Returns:
            List of tweet IDs that were processed
        """
        if "data" not in response_data or not response_data["data"]:
            return []

        tweet_ids = []
        now = datetime.now(timezone.utc)

        for tweet_data in response_data["data"]:
            tweet_id = tweet_data["id"]
            tweet_ids.append(tweet_id)

            # Check if tweet already exists
            existing = Tweet.find({"_id": tweet_id})

            if existing:
                # Update last_seen_at
                existing[0].last_seen_at = now
                existing[0].save()
            else:
                # Create new tweet
                tweet_created_at = datetime.fromisoformat(tweet_data.get("created_at", "").replace("Z", "+00:00")) if tweet_data.get("created_at") else now
                tweet = Tweet(
                    author_id=tweet_data.get("author_id"),
                    text=tweet_data.get("text", ""),
                    created_at=tweet_created_at,
                    conversation_id=tweet_data.get("conversation_id"),
                    in_reply_to_user_id=tweet_data.get("in_reply_to_user_id"),
                    referenced_tweets=tweet_data.get("referenced_tweets"),
                    entities=tweet_data.get("entities"),
                    attachments=tweet_data.get("attachments"),
                    lang=tweet_data.get("lang"),
                    possibly_sensitive=tweet_data.get("possibly_sensitive"),
                    source=tweet_data.get("source"),
                    reply_settings=tweet_data.get("reply_settings"),
                    public_metrics=tweet_data.get("public_metrics"),
                    raw=tweet_data,
                    first_seen_at=now,
                    last_seen_at=now
                )
                tweet.id = tweet_id
                tweet.save()

        return tweet_ids

    @staticmethod
    def upsert_users(response_data: Dict[str, Any]):
        """Upsert users from API response includes."""
        if "includes" not in response_data or "users" not in response_data["includes"]:
            return

        now = datetime.now(timezone.utc)

        for user_data in response_data["includes"]["users"]:
            user_id = user_data["id"]

            # Check if user exists
            existing = User.find({"_id": user_id})

            # Prepare bio snapshot
            bio_snapshot = {
                "description": user_data.get("description"),
                "url": user_data.get("url"),
                "location": user_data.get("location")
            }

            user_created_at = None
            if "created_at" in user_data:
                user_created_at = datetime.fromisoformat(user_data["created_at"].replace("Z", "+00:00"))

            if existing:
                # Update existing user
                existing[0].username = user_data.get("username")
                existing[0].name = user_data.get("name")
                existing[0].verified = user_data.get("verified")
                existing[0].verified_type = user_data.get("verified_type")
                existing[0].protected = user_data.get("protected")
                existing[0].profile_image_url = user_data.get("profile_image_url")
                existing[0].public_metrics = user_data.get("public_metrics")
                existing[0].bio_snapshot = bio_snapshot
                existing[0].last_updated_at = now
                existing[0].raw = user_data
                existing[0].save()
            else:
                # Create new user
                user = User(
                    username=user_data.get("username"),
                    name=user_data.get("name"),
                    verified=user_data.get("verified"),
                    verified_type=user_data.get("verified_type"),
                    protected=user_data.get("protected"),
                    profile_image_url=user_data.get("profile_image_url"),
                    public_metrics=user_data.get("public_metrics"),
                    bio_snapshot=bio_snapshot,
                    created_at=user_created_at,
                    last_updated_at=now,
                    raw=user_data
                )
                user.id = user_id
                user.save()

    @staticmethod
    def upsert_media(response_data: Dict[str, Any]) -> List[str]:
        """
        Upsert media from API response includes.

        Returns:
            List of media_keys that need downloading
        """
        if "includes" not in response_data or "media" not in response_data["includes"]:
            return []

        media_keys_to_download = []

        for media_data in response_data["includes"]["media"]:
            media_key = media_data["media_key"]

            # Check if media exists
            existing = Media.find({"_id": media_key})

            if existing:
                # Update metadata but preserve download status
                existing[0].type = media_data.get("type")
                existing[0].url = media_data.get("url")
                existing[0].preview_image_url = media_data.get("preview_image_url")
                existing[0].variants = media_data.get("variants")
                existing[0].width = media_data.get("width")
                existing[0].height = media_data.get("height")
                existing[0].duration_ms = media_data.get("duration_ms")
                existing[0].alt_text = media_data.get("alt_text")
                existing[0].public_metrics = media_data.get("public_metrics")
                existing[0].raw = media_data
                existing[0].save()

                # Mark for download if not yet successful
                if existing[0].download.get("status") != "ok":
                    media_keys_to_download.append(media_key)
            else:
                # Create new media
                media = Media(
                    type=media_data.get("type"),
                    url=media_data.get("url"),
                    preview_image_url=media_data.get("preview_image_url"),
                    variants=media_data.get("variants"),
                    width=media_data.get("width"),
                    height=media_data.get("height"),
                    duration_ms=media_data.get("duration_ms"),
                    alt_text=media_data.get("alt_text"),
                    public_metrics=media_data.get("public_metrics"),
                    raw=media_data
                )
                media.id = media_key
                media.save()
                media_keys_to_download.append(media_key)

        return media_keys_to_download

    @staticmethod
    def get_since_id(key: str) -> Optional[str]:
        """Get the since_id for a given query key."""
        state = State.find({"_id": key})
        if state:
            return state[0].value
        return None

    @staticmethod
    def set_since_id(key: str, value: str):
        """Set the since_id for a given query key."""
        state = State.find({"_id": key})
        now = datetime.now(timezone.utc)

        if state:
            state[0].value = value
            state[0].updated_at = now
            state[0].save()
        else:
            new_state = State(value=value, updated_at=now)
            new_state.id = key
            new_state.save()


# ============================================================================
# MEDIA DOWNLOADING (Section 5)
# ============================================================================

class MediaDownloader:
    """Handles downloading media files to local disk."""

    def __init__(self):
        MEDIA_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def download_media(self, media_key: str):
        """
        Download media file for the given media_key.

        Args:
            media_key: The media key to download
        """
        media_docs = Media.find({"_id": media_key})
        if not media_docs:
            return

        media = media_docs[0]

        # Check if already downloaded successfully
        if media.download.get("status") == "ok":
            return

        # Check if max attempts reached
        if media.download.get("attempts", 0) >= MAX_MEDIA_DOWNLOAD_ATTEMPTS:
            return

        # Determine download URL and extension
        download_url = None
        extension = None

        if media.type == "photo":
            download_url = media.url
            extension = "jpg"
        elif media.type in ("video", "animated_gif"):
            # Find highest bitrate variant
            if media.variants:
                mp4_variants = [v for v in media.variants if v.get("content_type") == "video/mp4"]
                if mp4_variants:
                    highest = max(mp4_variants, key=lambda v: v.get("bit_rate", 0))
                    download_url = highest.get("url")
                    extension = "mp4"

        if not download_url:
            media.download["status"] = "failed"
            media.download["attempts"] = media.download.get("attempts", 0) + 1
            media.download["last_attempt_at"] = datetime.now(timezone.utc)
            media.save()
            return

        # Attempt download
        try:
            media.download["attempts"] = media.download.get("attempts", 0) + 1
            media.download["last_attempt_at"] = datetime.now(timezone.utc)

            response = requests.get(download_url, timeout=30)
            response.raise_for_status()

            content = response.content
            checksum = hashlib.sha256(content).hexdigest()

            # Save to disk
            file_path = MEDIA_DOWNLOAD_DIR / f"{media_key}.{extension}"
            file_path.write_bytes(content)

            # Update media document
            media.download["status"] = "ok"
            media.download["stored_path"] = str(file_path)
            media.download["byte_size"] = len(content)
            media.download["checksum"] = checksum
            media.download["last_success_at"] = datetime.now(timezone.utc)
            media.save()

            logger.info(f"  📥 Downloaded {media_key}.{extension} ({len(content)} bytes)")

        except Exception as e:
            logger.info(f"  ❌ Failed to download {media_key}: {e}")
            media.download["status"] = "failed"
            media.save()


# ============================================================================
# PAGINATION AND POLLING (Section 6)
# ============================================================================

class TwitterPoller:
    """Main polling logic for Twitter account tracking."""

    def __init__(self, api: TwitterAPIWrapper):
        self.api = api
        self.persister = DataPersister()
        self.downloader = MediaDownloader()

    def paginate_search(self, query: str, since_id_key: str, max_pages: int = MAX_PAGES_PER_QUERY) -> int:
        """
        Paginate through search results, upserting data and handling since_id.

        Args:
            query: Twitter search query
            since_id_key: State key for tracking since_id
            max_pages: Maximum pages to fetch

        Returns:
            Number of new tweets processed
        """
        since_id = self.persister.get_since_id(since_id_key)

        total_tweets = 0
        max_tweet_id = since_id
        next_token = None
        page = 0

        while page < max_pages:
            try:
                response = self.api.fetch_search_recent(
                    query=query,
                    since_id=since_id,
                    next_token=next_token
                )

                # Check rate limits
                if "_rate_limit" in response:
                    self.api.check_rate_limit_and_sleep(response["_rate_limit"])

                # Upsert data
                tweet_ids = self.persister.upsert_tweets(response)
                self.persister.upsert_users(response)
                media_keys = self.persister.upsert_media(response)

                # Download media
                for media_key in media_keys:
                    self.downloader.download_media(media_key)

                # Track max tweet ID
                if tweet_ids:
                    total_tweets += len(tweet_ids)
                    max_id_in_batch = max(tweet_ids, key=lambda x: int(x))
                    if not max_tweet_id or int(max_id_in_batch) > int(max_tweet_id):
                        max_tweet_id = max_id_in_batch

                # Check for next page
                next_token = response.get("meta", {}).get("next_token")
                if not next_token:
                    break

                page += 1

            except Exception as e:
                logger.info(f"  ⚠️  Error during pagination: {e}")
                break

        # Update since_id only after successful batch
        if max_tweet_id and max_tweet_id != since_id:
            self.persister.set_since_id(since_id_key, max_tweet_id)

        return total_tweets

    def poll_tweets_mentions_replies(self, usernames: List[str]) -> int:
        """
        Poll for mentions, replies, and tweets from tracked accounts (combined query).

        Returns:
            Number of new tweets processed
        """
        query = build_combined_query(usernames)
        logger.info(f"\n🔍 Searching mentions/replies/from users: {query[:100]}...")

        count = self.paginate_search(query, "since_id:combined", max_pages=MAX_PAGES_PER_QUERY)
        logger.info(f"  ✓ Found {count} new tweets")
        return count

    def poll_quote_tweets(self, usernames: List[str]) -> int:
        """
        Poll for quote tweets of recent tweets from tracked accounts.

        Strategy: Smart prioritization based on engagement metrics
        - Gets recent tweets from tracked users (last 50)
        - Filters for tweets with quotes OR high engagement (like_count > 10)
        - Sorts by quote_count DESC, then like_count DESC
        - Checks top 5 tweets for new quotes
        - 4-hour cache to avoid redundant checks
        - Respects rate limits

        Returns:
            Number of new quote tweets processed
        """
        logger.info(f"\n🔍 Searching quote tweets of popular posts...")

        # Get recent tweets from tracked users to check for quotes
        recent_tweets = []
        for username in usernames:
            # Get user_id for this username
            handles = TrackedHandle.find({"username": username})
            if not handles:
                continue

            user_id = handles[0].user_id
            if not user_id:
                continue

            # Get their recent tweets from our database (last 50 tweets)
            user_tweets = Tweet.find(
                {"author_id": user_id},
                sort="created_at",
                desc=True,
                limit=50
            )
            recent_tweets.extend(user_tweets)

        if not recent_tweets:
            logger.info(f"  ✓ No recent source tweets to check")
            return 0

        # Filter for tweets worth checking:
        # - Has known quotes (quote_count > 0), OR
        # - High engagement (like_count > 10)
        candidates = []
        for tweet in recent_tweets:
            metrics = tweet.public_metrics or {}
            quote_count = metrics.get('quote_count', 0)
            like_count = metrics.get('like_count', 0)

            if quote_count > 0 or like_count > 10:
                candidates.append(tweet)

        if not candidates:
            logger.info(f"  ✓ No tweets meet quote check criteria (quote_count > 0 or like_count > 10)")
            return 0

        # Sort by quote_count DESC (prioritize known quotes), then like_count DESC
        candidates.sort(
            key=lambda t: (
                t.public_metrics.get('quote_count', 0) if t.public_metrics else 0,
                t.public_metrics.get('like_count', 0) if t.public_metrics else 0
            ),
            reverse=True
        )

        # Take top 5
        tweets_to_check = candidates[:5]

        total_quotes = 0
        checked_count = 0

        for tweet in tweets_to_check:
            tweet_id = tweet.id

            # Check if we've already checked this tweet recently
            state_key = f"quote_checked:{tweet_id}"
            last_checked = self.persister.get_since_id(state_key)

            # Skip if checked in the last 4 hours (to avoid redundant checks)
            if last_checked:
                try:
                    from datetime import datetime, timezone
                    last_checked_time = datetime.fromisoformat(last_checked)
                    now = datetime.now(timezone.utc)
                    if (now - last_checked_time).total_seconds() < 14400:  # 4 hours = 14400 seconds
                        continue
                except:
                    pass

            # Check rate limit before making call
            # If we've already made many calls this loop, be conservative
            if self.api.calls_this_loop >= HARD_CAP_CALLS_PER_LOOP - 2:
                logger.info(f"  ⚠️  Approaching call limit ({self.api.calls_this_loop}/{HARD_CAP_CALLS_PER_LOOP}), stopping quote checks")
                break

            try:
                # Fetch quote tweets for this tweet (just first page to keep it light)
                response = self.api.fetch_quote_tweets(tweet_id, max_results=100)

                # Check rate limits
                if "_rate_limit" in response:
                    rate_limit = response["_rate_limit"]
                    remaining = rate_limit.get('remaining')

                    # If rate limit is low, stop checking more quotes
                    if remaining and int(remaining) < 10:
                        logger.info(f"  ⚠️  Quote tweets rate limit low ({remaining} remaining), stopping checks")
                        break

                    self.api.check_rate_limit_and_sleep(rate_limit)

                # Upsert data
                quote_tweet_ids = self.persister.upsert_tweets(response)
                self.persister.upsert_users(response)
                media_keys = self.persister.upsert_media(response)

                # Download media
                for media_key in media_keys:
                    self.downloader.download_media(media_key)

                if quote_tweet_ids:
                    total_quotes += len(quote_tweet_ids)
                    metrics = tweet.public_metrics or {}
                    logger.info(f"  📝 Found {len(quote_tweet_ids)} quotes of tweet {tweet_id[:10]}... (❤️ {metrics.get('like_count', 0)}, 💬 {metrics.get('quote_count', 0)})")

                # Mark as checked
                checked_count += 1
                self.persister.set_since_id(state_key, datetime.now(timezone.utc).isoformat())

            except Exception as e:
                logger.info(f"  ⚠️  Error checking quotes for tweet {tweet_id}: {e}")
                continue

        logger.info(f"  ✓ Found {total_quotes} new quote tweets (checked {checked_count}/{len(candidates)} eligible tweets)")
        return total_quotes

    def run_poll_loop(self, usernames: List[str]):
        """Run a single polling loop."""
        logger.info(f"\n{'='*80}")
        logger.info(f"⏰ Poll started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(f"{'='*80}")

        self.api.reset_call_counter()

        total_tweets = 0

        # Combined mentions/replies/from:users
        total_tweets += self.poll_tweets_mentions_replies(usernames)

        # Quote tweets
        total_tweets += self.poll_quote_tweets(usernames)

        logger.info(f"\n📊 Total new tweets this loop: {total_tweets}")
        logger.info(f"📞 API calls this loop: {self.api.calls_this_loop}")


# ============================================================================
# BOOTSTRAP (Section 7)
# ============================================================================

def bootstrap_tracked_handles(api: TwitterAPIWrapper, usernames: List[str]):
    """
    Bootstrap tracked handles by resolving user IDs.

    Args:
        api: Twitter API wrapper
        usernames: List of usernames to track
    """
    logger.info("\n🚀 Bootstrapping tracked handles...")

    # Fetch user data
    response = api.fetch_users_by_usernames(usernames)

    if "data" not in response:
        logger.info("  ⚠️  No user data returned")
        return

    # Upsert users to users collection
    DataPersister.upsert_users({"includes": {"users": response["data"]}})

    # Create/update tracked handles
    for user_data in response["data"]:
        username = user_data["username"]
        user_id = user_data["id"]

        existing = TrackedHandle.find({"username": username})
        if existing:
            existing[0].user_id = user_id
            existing[0].active = True
            existing[0].updated_at = datetime.now(timezone.utc)
            existing[0].save()
            logger.info(f"  ✓ Updated handle: @{username} (id: {user_id})")
        else:
            handle = TrackedHandle(username=username, user_id=user_id, active=True)
            handle.save()
            logger.info(f"  ✓ Added handle: @{username} (id: {user_id})")

    logger.info("  ✓ Bootstrap complete")


# ============================================================================
# MAIN - Legacy standalone mode (for reference only)
# ============================================================================
# NOTE: This is kept for reference but not used in production.
# The gateway now runs via poll_twitter_gateway() in twitter.py,
# deployed as a Modal scheduled function.
#
# For local testing, use the API endpoint:
#   POST http://localhost:8000/dev/twitter/poll
#
# Or run the gateway directly:
#   DB=STAGE python -c "import asyncio; from eve.agent.deployments.twitter import poll_twitter_gateway; asyncio.run(poll_twitter_gateway())"
