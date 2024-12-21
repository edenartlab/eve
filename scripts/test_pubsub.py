import os
import argparse
from ably import AblyRealtime
import asyncio
import time
from dotenv import load_dotenv

load_dotenv()


async def publish_test_messages(channel_name: str):
    # Initialize Ably with publisher key
    client = AblyRealtime(os.getenv("ABLY_PUBLISHER_KEY"))
    channel = client.channels.get(channel_name)

    try:
        # Publish a few test messages
        for i in range(5):
            data = {
                "type": "test",
                "message": f"Test message {i}",
                "timestamp": time.time(),
            }
            await channel.publish("update", data)
            print(f"Published message {i} to channel {channel_name}")
            await asyncio.sleep(2)  # Wait 2 seconds between messages

    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("channel", help="Channel name to publish to")
    args = parser.parse_args()

    asyncio.run(publish_test_messages(args.channel))
