import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from fastapi import Request
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.deployments import PlatformClient
from eve.agent.deployments.twitter_gateway import Tweet
from eve.agent.deployments.utils import get_api_url
from eve.agent.session.context import (
    add_chat_message,
    add_user_to_session,
    build_llm_context,
)
from eve.agent.session.models import (
    Channel,
    ChatMessage,
    ChatMessageRequestInput,
    Deployment,
    DeploymentConfig,
    DeploymentSecrets,
    LLMConfig,
    PromptSessionContext,
    Session,
    SessionUpdateConfig,
    UpdateType,
)
from eve.agent.session.runtime import async_prompt_session
from eve.api.errors import APIError
from eve.mongo import MongoDocumentNotFound
from eve.s3 import upload_file_from_url
from eve.tool import Tool
from eve.user import User, increment_message_count

if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest


# ========================================================================
# Twitter Webhook Processing Helpers
# ========================================================================


def extract_media_urls_from_tweet(tweet_data: Dict[str, Any]) -> List[str]:
    """Extract media URLs from tweet data"""
    media_urls = []

    # Check if tweet has media attachments
    if "includes" in tweet_data and "media" in tweet_data["includes"]:
        for media_item in tweet_data["includes"]["media"]:
            media_type = media_item.get("type")

            if media_type == "photo":
                # For photos, use the url field
                if "url" in media_item:
                    media_urls.append(media_item["url"])
            elif media_type in ("video", "animated_gif"):
                # For videos, get the highest bitrate variant
                variants = media_item.get("variants", [])
                mp4_variants = [
                    v for v in variants if v.get("content_type") == "video/mp4"
                ]
                if mp4_variants:
                    # Get highest bitrate variant
                    highest = max(mp4_variants, key=lambda v: v.get("bit_rate", 0))
                    if "url" in highest:
                        media_urls.append(highest["url"])

    return media_urls


def upload_media_to_s3(media_urls: List[str]) -> List[str]:
    """Upload media URLs to S3 and return S3 URLs"""
    uploaded_urls = []
    for media_url in media_urls:
        try:
            uploaded_url, _ = upload_file_from_url(media_url)
            uploaded_urls.append(uploaded_url)
        except Exception as e:
            logger.error(f"Error uploading {media_url} to S3: {e}")
    return uploaded_urls


async def fetch_tweet_ancestry(
    tweet_id: str, conversation_id: str, twitter_api
) -> List[Dict[str, Any]]:
    """
    Fetch tweet ancestry by getting all tweets in the conversation.
    Returns tweets in chronological order (oldest first).
    """
    try:
        import httpx

        # Use Twitter API v2 conversation search
        # We'll search for tweets in this conversation that came before this tweet
        params = {
            "query": f"conversation_id:{conversation_id}",
            "max_results": 100,
            "tweet.fields": "id,text,created_at,author_id,conversation_id,in_reply_to_user_id,referenced_tweets,attachments",
            "user.fields": "id,username,name,profile_image_url",
            "media.fields": "media_key,type,url,preview_image_url,variants",
            "expansions": "author_id,attachments.media_keys,referenced_tweets.id",
        }

        headers = {
            "Authorization": f"Bearer {twitter_api.access_token}",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://api.twitter.com/2/tweets/search/recent",
                headers=headers,
                params=params,
            )

        response.raise_for_status()
        data = response.json()

        tweets = data.get("data", [])
        # Sort by created_at to get chronological order
        tweets.sort(key=lambda t: t.get("created_at", ""))

        # Filter to only include tweets before the current one
        for i, tweet in enumerate(tweets):
            if tweet["id"] == tweet_id:
                # Return all tweets before this one
                return {"data": tweets[:i], "includes": data.get("includes", {})}

        # If we didn't find the current tweet, return all tweets
        return data

    except Exception as e:
        logger.error(f"Error fetching tweet ancestry: {e}")
        return {"data": [], "includes": {}}


async def unpack_tweet(tweet: Dict[str, Any], includes: Dict[str, Any]):
    """Unpack tweet into components"""
    tweet_id = tweet.get("id")
    author_id = tweet.get("author_id")
    text = tweet.get("text") or ""
    created_at = tweet.get("created_at")

    # Get author info from includes
    author_username = None
    author_data = None
    if "users" in includes:
        for user in includes["users"]:
            if user["id"] == author_id:
                author_username = user.get("username")
                author_data = user
                break

    # Get media URLs from includes
    media_urls = []
    if "attachments" in tweet and "media_keys" in tweet["attachments"]:
        media_keys = tweet["attachments"]["media_keys"]
        if "media" in includes:
            for media_item in includes["media"]:
                if media_item["media_key"] in media_keys:
                    media_type = media_item.get("type")
                    if media_type == "photo" and "url" in media_item:
                        media_urls.append(media_item["url"])
                    elif media_type in ("video", "animated_gif"):
                        variants = media_item.get("variants", [])
                        mp4_variants = [
                            v for v in variants if v.get("content_type") == "video/mp4"
                        ]
                        if mp4_variants:
                            highest = max(
                                mp4_variants, key=lambda v: v.get("bit_rate", 0)
                            )
                            if "url" in highest:
                                media_urls.append(highest["url"])

    return (
        tweet_id,
        author_id,
        author_username,
        author_data,
        text,
        media_urls,
        created_at,
    )


async def induct_user(user: User, author_data: Dict[str, Any]):
    """Update user metadata from Twitter profile"""
    if not author_data:
        return

    pfp = author_data.get("profile_image_url")
    if pfp and pfp != user.userImage:
        try:
            # Upload profile image to S3
            pfp_url, _ = upload_file_from_url(pfp)
            user.update(userImage=pfp_url.split("/")[-1])
        except Exception as e:
            logger.error(f"Error uploading pfp {pfp} for user {str(user.id)}: {str(e)}")


async def process_twitter_tweet(
    tweet_id: str,
    tweet_data: Dict[str, Any],
    deployment_id: str,
):
    """Process a Twitter tweet event - main processing logic"""
    try:
        # Load deployment and agent
        deployment = Deployment.from_mongo(deployment_id)
        if not deployment or not deployment.agent:
            raise Exception("Deployment or agent not found")

        agent = Agent.from_mongo(deployment.agent)
        if not agent:
            raise Exception("Agent not found")

        # Extract tweet info
        tweet = tweet_data.get("data") if "data" in tweet_data else tweet_data
        includes = tweet_data.get("includes", {})

        conversation_id = tweet.get("conversation_id")
        author_id = tweet.get("author_id")
        text = tweet.get("text", "")

        # Get author info
        author_username = None
        author_data = None
        if "users" in includes:
            for user in includes["users"]:
                if user["id"] == author_id:
                    author_username = user.get("username")
                    author_data = user
                    break

        if not author_username:
            logger.error(f"Could not find author username for tweet {tweet_id}")
            return {"status": "failed", "error": "Author not found"}

        # Get or create user and update profile
        user = User.from_twitter(author_id, author_username)
        await induct_user(user, author_data)

        # Handle attachments/media - upload to S3
        media_urls = []
        if "attachments" in tweet and "media_keys" in tweet["attachments"]:
            media_keys = tweet["attachments"]["media_keys"]
            if "media" in includes:
                for media_item in includes["media"]:
                    if media_item["media_key"] in media_keys:
                        media_type = media_item.get("type")
                        if media_type == "photo" and "url" in media_item:
                            media_urls.append(media_item["url"])
                        elif media_type in ("video", "animated_gif"):
                            variants = media_item.get("variants", [])
                            mp4_variants = [
                                v
                                for v in variants
                                if v.get("content_type") == "video/mp4"
                            ]
                            if mp4_variants:
                                highest = max(
                                    mp4_variants, key=lambda v: v.get("bit_rate", 0)
                                )
                                if "url" in highest:
                                    media_urls.append(highest["url"])

        # Upload media to S3
        media_urls = upload_media_to_s3(media_urls)

        # Create session key based on conversation_id
        session_key = f"TWITTER-{conversation_id}"
        session = None

        # Strategy 1: Look up by standard Twitter session_key
        try:
            session = Session.load(session_key=session_key)
            if session.platform != "twitter":
                session.update(platform="twitter")

            # Reactivate if deleted or archived
            if session.deleted:
                session.update(deleted=False)

        except MongoDocumentNotFound:
            logger.info(f"Session not found for conversation ID: {conversation_id}")
            pass

        # Strategy 2: Check if any Tweet in this conversation has a linked Eden session
        # (handles case where agent tweeted from Eden, user replies on Twitter)
        if not session:
            try:
                linked_tweet = Tweet.find_one(
                    {"conversation_id": conversation_id, "session_id": {"$ne": None}}
                )
                if linked_tweet and linked_tweet.session_id:
                    logger.info(
                        f"Found linked session {linked_tweet.session_id} via Tweet record"
                    )
                    session = Session.from_mongo(linked_tweet.session_id)
                    if session.deleted or session.status == "archived":
                        session.update(deleted=False, status="active")
                    # Update session_key for future lookups
                    # if not session.session_key or not session.session_key.startswith("TWITTER-"):
                    #    session.update(session_key=session_key, platform="twitter")
                    # Reactivate if needed
            except Exception as e:
                logger.debug(f"Could not find linked session via Tweet: {e}")

        # Strategy 3: Create new session if none found
        if not session:
            session = Session(
                owner=agent.owner,
                agents=[agent.id],
                title="Twitter conversation",
                session_key=session_key,
                platform="twitter",
                status="active",
            )
            session.save()

            # Reconstruct thread: if this tweet is not the start of conversation, get previous tweets
            if conversation_id and conversation_id != tweet_id:
                logger.info(f"Reconstructing thread for tweet {tweet_id}")
                try:
                    # Get Twitter API client
                    from eve.tools.twitter import X

                    twitter_api = X(deployment)

                    prev_tweets_data = await fetch_tweet_ancestry(
                        tweet_id, conversation_id, twitter_api
                    )
                    prev_tweets = prev_tweets_data.get("data", [])
                    prev_includes = prev_tweets_data.get("includes", {})

                    agent_twitter_id = deployment.secrets.twitter.twitter_id

                    for prev_tweet in prev_tweets:
                        (
                            tweet_id_,
                            author_id_,
                            author_username_,
                            author_data_,
                            text_,
                            media_urls_,
                            created_at_,
                        ) = await unpack_tweet(prev_tweet, prev_includes)

                        # Upload media to S3
                        media_urls_ = upload_media_to_s3(media_urls_)

                        # Parse created_at
                        created_at = (
                            datetime.fromisoformat(created_at_.replace("Z", "+00:00"))
                            if created_at_
                            else datetime.now(timezone.utc)
                        )

                        # Determine role
                        if author_id_ == agent_twitter_id:
                            role = "assistant"
                            tweet_user = agent
                        else:
                            role = "user"
                            tweet_user = User.from_twitter(author_id_, author_username_)

                        message = ChatMessage(
                            createdAt=created_at,
                            session=session.id,
                            channel=Channel(type="twitter", key=tweet_id_),
                            role=role,
                            content=text_,
                            sender=tweet_user.id,
                            attachments=media_urls_,
                        )
                        message.save()

                        # Increment message count for sender
                        increment_message_count(tweet_user.id)

                        # Add user to Session.users for user role messages
                        if role == "user":
                            add_user_to_session(session, tweet_user.id)
                except Exception as e:
                    logger.error(f"Error reconstructing thread: {e}")

        # Load twitter tool
        twitter_tool = Tool.load("tweet")

        # Create prompt context
        prompt_context = PromptSessionContext(
            session=session,
            initiating_user_id=str(user.id),
            message=ChatMessageRequestInput(
                channel=Channel(type="twitter", key=tweet_id),
                content=text,
                sender_name=author_username,
                attachments=media_urls if media_urls else None,
            ),
            update_config=SessionUpdateConfig(
                deployment_id=str(deployment.id),
                update_endpoint=f"{get_api_url()}/v2/deployments/emission",
                twitter_tweet_id=tweet_id,
                twitter_author_id=author_id,
            ),
            llm_config=LLMConfig(model="claude-sonnet-4-5"),
            extra_tools={twitter_tool.name: twitter_tool},
        )

        # Add user message to session
        message = await add_chat_message(session, prompt_context)

        # Update Tweet record with Eden session/message linkage
        try:
            existing_tweet = Tweet.find_one({"_id": tweet_id})
            if existing_tweet:
                existing_tweet.update(
                    session_id=str(session.id),
                    message_id=str(message.id),
                )
                logger.info(
                    f"Linked tweet {tweet_id} to session {session.id}, message {message.id}"
                )
        except Exception as e:
            logger.warning(f"Could not update Tweet with session/message linkage: {e}")

        # Build LLM context
        llm_context = await build_llm_context(
            session,
            agent,
            prompt_context,
            trace_id=str(uuid.uuid4()),
        )

        # Execute prompt session
        new_messages = []
        async for update in async_prompt_session(
            session, llm_context, agent, context=prompt_context, is_client_platform=True
        ):
            if update.type == UpdateType.ASSISTANT_MESSAGE:
                new_messages.append(update.message)

        return {
            "status": "completed",
            "session_id": str(session.id),
            "message_id": str(message.id),
        }

    except Exception as e:
        logger.exception(f"Error processing Twitter tweet {tweet_id}: {e}")
        return {"status": "failed", "error": str(e)}


class TwitterClient(PlatformClient):
    TOOLS = [
        "tweet",
        "twitter_search",
        "twitter_mentions",
        "twitter_trends",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Twitter OAuth 2.0 credentials and add Twitter tools"""
        try:
            # Simple validation - just check we have the required OAuth 2.0 fields
            if not secrets.twitter.access_token:
                raise ValueError("Missing OAuth 2.0 access token")

            if not secrets.twitter.twitter_id:
                raise ValueError("Missing Twitter user ID")

            if not secrets.twitter.username:
                raise ValueError("Missing Twitter username")
            # Add Twitter tools to agent
            self.add_tools()

            # Add twitter username to agent's social_accounts
            self.agent.get_collection().update_one(
                {"_id": self.agent.id},
                {
                    "$set": {"social_accounts.twitter": secrets.twitter.username},
                    "$currentDate": {"updatedAt": True},
                },
            )

            return secrets, config
        except Exception as e:
            logger.error(f"Invalid Twitter credentials: {str(e)}")
            raise APIError(f"Invalid Twitter credentials: {str(e)}", status_code=400)

    async def postdeploy(self) -> None:
        """No post-deployment actions needed for Twitter"""
        pass

    async def stop(self) -> None:
        """Stop Twitter client"""
        self.remove_tools()

        # Remove twitter from agent's social_accounts
        self.agent.get_collection().update_one(
            {"_id": self.agent.id},
            {
                "$unset": {"social_accounts.twitter": ""},
                "$currentDate": {"updatedAt": True},
            },
        )

    async def interact(self, request: Request) -> None:
        """Interact with the Twitter client"""
        pass

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the Twitter client"""
        pass


# ========================================================================
# Twitter Gateway Polling
# ========================================================================


async def poll_twitter_gateway(local_mode: bool = False):
    """
    Main Twitter polling gateway function.
    Polls Twitter API for new mentions/replies/tweets and spawns processing tasks.

    Args:
        local_mode: If True, processes tweets synchronously for local testing.
                   If False (default), spawns Modal tasks for production.

    This runs as a scheduled Modal function in production.
    For local testing, set local_mode=True to process synchronously.
    """
    import modal

    from eve import db

    # Import MongoDB collections from twitter_gateway
    from eve.agent.deployments.twitter_gateway import (
        EXPANSIONS,
        MAX_PAGES_PER_QUERY,
        MAX_RESULTS_PER_PAGE,
        MEDIA_FIELDS,
        TWEET_FIELDS,
        USER_FIELDS,
        DataPersister,
        Tweet,
        build_combined_query,
    )

    logger.info("=== Starting Twitter polling gateway ===")

    try:
        # Get all Twitter deployments with enable_tweet enabled
        active_twitter_deployments = [
            d
            for d in Deployment.find({"platform": "twitter"})
            if (
                d.config
                and d.config.twitter
                and d.config.twitter.enable_tweet
                and d.config.twitter.username
            )
        ]

        if not active_twitter_deployments:
            logger.info("No active Twitter deployments found")
            return {"status": "ok", "message": "No active deployments!"}

        # Build list of usernames to track
        tracked_usernames = [
            d.config.twitter.username for d in active_twitter_deployments
        ]
        logger.info(
            f"Tracking {len(tracked_usernames)} Twitter accounts: {tracked_usernames}"
        )

        # Build deployment map for quick lookup (username -> deployment)
        deployment_map = {
            d.config.twitter.username: d for d in active_twitter_deployments
        }

        # Get agent Twitter IDs to skip self-tweets
        agent_twitter_ids = set()
        for d in active_twitter_deployments:
            if d.secrets and d.secrets.twitter and d.secrets.twitter.twitter_id:
                agent_twitter_ids.add(d.secrets.twitter.twitter_id)

        # Use app-only bearer token for search (no user auth needed, no token refresh issues)
        import os

        import requests

        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer_token:
            logger.error("TWITTER_BEARER_TOKEN environment variable not set")
            return {"status": "error", "error": "TWITTER_BEARER_TOKEN not set"}

        logger.info(f"Using app bearer token: {bearer_token[:10]}...")

        # Build combined query for all tracked accounts
        query = build_combined_query(tracked_usernames)
        logger.info(f"Using query: {query}")

        # Get since_id from state
        persister = DataPersister()
        since_id = persister.get_since_id("since_id:combined")

        total_tweets = 0
        max_tweet_id = since_id
        next_token = None
        page = 0

        # Paginate through results
        while page < MAX_PAGES_PER_QUERY:
            try:
                # Build request params
                params = {
                    "query": query,
                    "max_results": min(MAX_RESULTS_PER_PAGE, 100),
                    "tweet.fields": TWEET_FIELDS,
                    "user.fields": USER_FIELDS,
                    "media.fields": MEDIA_FIELDS,
                    "expansions": EXPANSIONS,
                }

                if since_id:
                    params["since_id"] = since_id
                if next_token:
                    params["next_token"] = next_token

                # Make API request using app bearer token (no user auth needed)
                try:
                    headers = {"Authorization": f"Bearer {bearer_token}"}
                    response = requests.get(
                        "https://api.twitter.com/2/tweets/search/recent",
                        params=params,
                        headers=headers,
                    )
                    response.raise_for_status()
                except Exception as api_error:
                    logger.error(f"Twitter API request failed: {api_error}")
                    logger.error("Check TWITTER_BEARER_TOKEN validity.")
                    raise

                # Debug: log response
                logger.info(f"Twitter API response status: {response.status_code}")
                logger.debug(f"Response text: {response.text[:500]}")

                # Parse JSON response
                try:
                    response_data = response.json()
                except Exception as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Response status: {response.status_code}")
                    logger.error(f"Response text: {response.text}")
                    break

                # Check for errors in response
                if "errors" in response_data:
                    logger.error(
                        f"Twitter API returned errors: {response_data['errors']}"
                    )
                    break

                # Log results
                result_count = len(response_data.get("data", []))
                logger.info(f"Got {result_count} tweets from API")

                # Persist tweets, users, media to MongoDB
                tweet_ids = persister.upsert_tweets(response_data)
                persister.upsert_users(response_data)
                persister.upsert_media(response_data)

                if not tweet_ids:
                    logger.info("No new tweets found, stopping pagination")
                    break

                total_tweets += len(tweet_ids)

                # Track max tweet ID
                max_id_in_batch = max(tweet_ids, key=lambda x: int(x))
                if not max_tweet_id or int(max_id_in_batch) > int(max_tweet_id):
                    max_tweet_id = max_id_in_batch

                # Process each new tweet
                tweets = response_data.get("data", [])
                includes = response_data.get("includes", {})

                for tweet in tweets:
                    tweet_id = tweet["id"]
                    author_id = tweet.get("author_id")

                    # Skip if tweet is from any agent itself (prevent loops)
                    if author_id in agent_twitter_ids:
                        logger.info(
                            f"Skipping tweet from agent itself (author_id: {author_id})"
                        )
                        continue

                    # Determine which agent should respond
                    # Check for mentions and replies
                    mentioned_usernames = []
                    entities = tweet.get("entities", {})
                    if "mentions" in entities:
                        mentioned_usernames = [
                            m["username"] for m in entities["mentions"]
                        ]

                    # Check if replying to any tracked agent
                    in_reply_to_user_id = tweet.get("in_reply_to_user_id")
                    reply_to_username = None
                    if in_reply_to_user_id:
                        # Find username from includes
                        for user in includes.get("users", []):
                            if user["id"] == in_reply_to_user_id:
                                reply_to_username = user.get("username")
                                break

                    # Find matching deployment (mentioned or replied to)
                    matching_deployment = None
                    match_reason = None

                    # First check mentions
                    for username in mentioned_usernames:
                        if username in deployment_map:
                            matching_deployment = deployment_map[username]
                            match_reason = "mention"
                            break

                    # Then check replies
                    if not matching_deployment and reply_to_username in deployment_map:
                        matching_deployment = deployment_map[reply_to_username]
                        match_reason = "reply"

                    if not matching_deployment:
                        logger.debug(f"No matching deployment for tweet {tweet_id}")
                        continue

                    logger.info(
                        f"Found matching deployment {matching_deployment.id} via {match_reason} for tweet {tweet_id}"
                    )

                    # Check if already processed (use X_tweets collection)
                    # If tweet exists and has been processed, skip
                    existing_tweet = Tweet.find_one({"_id": tweet_id})
                    if (
                        existing_tweet
                        and hasattr(existing_tweet, "processed")
                        and existing_tweet.processed
                    ):
                        logger.info(f"Tweet {tweet_id} already processed, skipping")
                        continue

                    # Mark as processed
                    if existing_tweet:
                        existing_tweet.processed = True
                        existing_tweet.save()

                    # Build tweet_data with full context
                    tweet_data = {"data": tweet, "includes": includes}

                    # Process tweet (local mode = synchronous, production = spawn Modal task)
                    try:
                        if local_mode:
                            # Local development: process synchronously
                            logger.info(
                                f"Processing tweet {tweet_id} locally (synchronous)"
                            )
                            result = await process_twitter_tweet(
                                tweet_id, tweet_data, str(matching_deployment.id)
                            )
                            logger.info(f"Local processing result: {result}")
                        else:
                            # Production: spawn Modal function for parallel processing
                            func = modal.Function.from_name(
                                f"api-{db.lower()}",
                                "process_twitter_tweet_fn",
                                environment_name="main",
                            )
                            func.spawn(
                                tweet_id, tweet_data, str(matching_deployment.id)
                            )
                            logger.info(f"Spawned processing task for tweet {tweet_id}")
                    except Exception as e:
                        logger.error(f"Failed to process tweet {tweet_id}: {e}")

                # Check for next page
                next_token = response_data.get("meta", {}).get("next_token")
                if not next_token:
                    break

                page += 1

            except Exception as e:
                logger.error(f"Error during pagination: {e}")
                break

        # Update since_id only after successful batch
        if max_tweet_id and max_tweet_id != since_id:
            persister.set_since_id("since_id:combined", max_tweet_id)

        logger.info(f"=== Twitter polling complete: {total_tweets} new tweets ===")
        return {
            "status": "ok",
            "tweets_processed": total_tweets,
            "tracked_accounts": len(tracked_usernames),
        }

    except Exception as e:
        logger.exception(f"Error in Twitter polling gateway: {e}")
        return {"status": "error", "error": str(e)}
