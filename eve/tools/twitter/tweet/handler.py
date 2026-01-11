import logging
from datetime import datetime, timezone

from eve.agent.agent import Agent
from eve.agent.deployments.twitter_gateway import Tweet
from eve.agent.session.models import Deployment, Session
from eve.tool import ToolContext
from eve.tools.twitter import X


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent_obj = Agent.from_mongo(context.agent)

    # Auto-set session platform to twitter if not already set
    if context.session:
        session = Session.from_mongo(context.session)
        if session.platform is None:
            session.update(platform="twitter")

    # Enforce reply_to after first tweet in session
    reply_to = context.args.get("reply_to")
    if context.session and not reply_to:
        prior_tweet = Tweet.find_one({"session_id": context.session})
        if prior_tweet:
            raise Exception(
                "reply_to is required after your first tweet. Use the tweet ID from "
                "your previous tweet or the tweet you're responding to. For tweetstorms, "
                "reply to your own previous tweet."
            )

    deployment = Deployment.load(agent=agent_obj.id, platform="twitter")
    if not deployment:
        raise Exception("No valid twitter deployments found")

    x = X(deployment)

    if context.args.get("images"):
        media_ids = [x.tweet_media(image) for image in context.args.get("images", [])]
        response = x.post(
            text=context.args.get("content") or "",
            media_ids=media_ids,
            reply=context.args.get("reply_to"),
        )
    elif context.args.get("video"):
        media_ids = [x.tweet_media(context.args.get("video"))]
        response = x.post(
            text=context.args.get("content") or "",
            media_ids=media_ids,
            reply=context.args.get("reply_to"),
        )
    else:
        response = x.post(
            text=context.args.get("content"), reply=context.args.get("reply_to")
        )

    tweet_id = response.get("data", {}).get("id")
    url = f"https://x.com/{deployment.config.twitter.username}/status/{tweet_id}"

    # Save Tweet record with Eden session/message linkage
    try:
        # Fetch tweet details to get conversation_id
        tweet_details = x.get_tweet(tweet_id)
        tweet_data = tweet_details.get("data", {})

        now = datetime.now(timezone.utc)
        tweet_record = Tweet(
            author_id=deployment.secrets.twitter.twitter_id,
            text=context.args.get("content") or "",
            created_at=now,
            conversation_id=tweet_data.get("conversation_id", tweet_id),
            in_reply_to_user_id=tweet_data.get("in_reply_to_user_id"),
            referenced_tweets=tweet_data.get("referenced_tweets"),
            raw=tweet_data,
            first_seen_at=now,
            last_seen_at=now,
            processed=True,  # Mark as processed since we created it
            session_id=context.session,
            message_id=context.message,
        )
        tweet_record.id = tweet_id
        tweet_record.save()
        logging.info(
            f"Saved Tweet {tweet_id} with session={context.session}, message={context.message}"
        )
    except Exception as e:
        logging.warning(f"Could not save Tweet record: {e}")

    return {
        "output": [
            {
                "id": tweet_id,
                "url": url,
                "text": context.args.get("content"),
                "author_id": response.get("data", {}).get("author_id", "unknown"),
            }
        ]
    }
