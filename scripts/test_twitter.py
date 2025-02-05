#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional
from bson import ObjectId

from eve.tools.twitter import X
from eve.agent import Agent


def test_twitter_mentions(user_id: str, start_time: Optional[datetime] = None):
    """Test Twitter mentions functionality for a specific user."""

    # Default to 15 minutes ago if no start time provided
    if not start_time:
        start_time = datetime.now(timezone.utc) - timedelta(minutes=15)

    # Load agent from MongoDB
    agent = Agent.from_mongo(ObjectId(user_id))
    print(f"\nTesting Twitter mentions for agent: {agent.name}")

    # Initialize Twitter client
    twitter = X(agent)
    print("✓ Twitter client initialized")

    # formatted_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    # print(f"\nFetching mentions since {formatted_time}")
    # mentions = twitter.fetch_mentions(start_time=formatted_time)

    # if not mentions.get("data"):
    #     print("No mentions found in the specified timeframe")
    #     return

    # print(f"\nFound {len(mentions['data'])} mentions:")
    # for tweet in mentions["data"]:
    #     print(f"\nTweet ID: {tweet['id']}")
    #     print(f"Author ID: {tweet['author_id']}")
    #     print(f"Text: {tweet.get('text', 'No text available')}")
    #     print("-" * 50)

    twitter.post("Hello, world!")


def main():
    parser = argparse.ArgumentParser(description="Test Twitter mentions functionality")
    parser.add_argument(
        "--user-id",
        type=str,
        required=True,
        help="MongoDB ObjectId of the agent to test",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        help="ISO format datetime to start fetching mentions from (default: 15 minutes ago)",
    )

    args = parser.parse_args()

    start_time = None
    if args.start_time:
        try:
            start_time = datetime.fromisoformat(args.start_time)
        except ValueError:
            print(
                "Error: Invalid start time format. Please use ISO format (YYYY-MM-DDTHH:MM:SS+00:00)"
            )
            return

    test_twitter_mentions(user_id=args.user_id, start_time=start_time)


if __name__ == "__main__":
    main()
