import asyncio
from typing import Dict, Any
from eve.agent import Agent
from eve.deploy import ClientType, Deployment
from eve.tools.twitter import X


async def generate_response(agent: Agent, mention: Dict[str, Any]) -> str:
    """
    Asynchronously generate a response to a mention.
    Replace this placeholder with your actual response generation logic.
    """
    # Placeholder - replace with actual async response generation
    return "Thanks for reaching out! I'm an AI assistant and I'll be happy to help."


async def process_mention(client: X, agent: Agent, mention: Dict[str, Any]) -> None:
    """Process a single mention asynchronously"""
    try:
        # Generate response asynchronously
        response_text = await generate_response(agent, mention)

        # Post reply asynchronously
        # Note: client.post is synchronous, so we run it in a thread pool
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: client.post(response_text, reply_to_tweet_id=mention["id"])
        )
    except Exception as e:
        print(f"Error processing mention {mention['id']}: {e}")


async def handle_twitter_mentions():
    """Handle Twitter mentions for all agents with Twitter deployments"""
    # Get all agents with Twitter deployments
    deployments = Deployment.find({"platform": ClientType.TWITTER.value})

    # Create tasks for all deployments
    tasks = []
    for deployment in deployments:
        try:
            # Get the agent
            agent = Agent.find_one({"_id": deployment.agent})
            if not agent:
                continue

            # Initialize Twitter client
            client = X(agent)

            # Get mentions (run in thread pool since it's synchronous)
            mentions = await asyncio.get_event_loop().run_in_executor(
                None, client.fetch_mentions
            )

            if not mentions.get("data"):
                continue

            # Track users we've replied to
            replied_users = set()

            # Create tasks for each unique user mention
            for mention in mentions["data"]:
                author_id = mention["author_id"]

                # Skip if we've already replied to this user
                if author_id in replied_users:
                    continue

                # Add task for processing this mention
                tasks.append(process_mention(client, agent, mention))
                replied_users.add(author_id)

        except Exception as e:
            print(f"Error handling Twitter mentions for agent {deployment.agent}: {e}")
            continue

    # Run all tasks concurrently
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
