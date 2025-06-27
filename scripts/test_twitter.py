#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional
from bson import ObjectId

from eve.agent.deployments import Deployment
from eve.tools.twitter import X


def test_twitter_mentions(
    deployment: Deployment, start_time: Optional[datetime] = None
):
    """Test Twitter mentions functionality for a specific user."""

    # Default to 15 minutes ago if no start time provided
    if not start_time:
        start_time = datetime.now(timezone.utc) - timedelta(minutes=15)

    print(f"\nTesting Twitter mentions for deployment: {deployment.id}")

    # Initialize Twitter client
    twitter = X(deployment)
    print("✓ Twitter client initialized")

    formatted_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\nFetching mentions since {formatted_time}")
    mentions = twitter.fetch_mentions(start_time=formatted_time)

    if not mentions.get("data"):
        print("No mentions found in the specified timeframe")
        return

    print(f"\nFound {len(mentions['data'])} mentions:")
    for tweet in mentions["data"]:
        print(f"\nTweet ID: {tweet['id']}")
        print(f"Author ID: {tweet['author_id']}")
        print(f"Text: {tweet.get('text', 'No text available')}")
        print("-" * 50)


def test_twitter_post(deployment: Deployment):
    print(f"\nTesting Twitter post for deployment: {deployment.id}")
    twitter = X(deployment)
    print("✓ Twitter client initialized")
    response = twitter.post("Hello, world!")
    print(f"Tweet ID: {response.get('data', {}).get('id')}")
    print(response)


def main():
    parser = argparse.ArgumentParser(description="Test Twitter mentions functionality")
    parser.add_argument(
        "--deployment-id",
        type=str,
        required=True,
        help="MongoDB ObjectId of the agent to test",
    )
    parser.add_argument(
        "--method",
        choices=["mentions", "post"],
        type=str,
        help="Method to test",
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

    deployment = Deployment.load(ObjectId(args.deployment_id))
    if args.method == "mentions":
        test_twitter_mentions(deployment=deployment, start_time=start_time)
    elif args.method == "post":
        test_twitter_post(deployment=deployment)


if __name__ == "__main__":
    main()
