import os
import asyncio
import json
from typing import Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("EDEN_API_URL")
HEADERS = {"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"}


async def test_chat(agent_id: Optional[str] = None, user_id: Optional[str] = None):
    """Test the /chat endpoint against running API"""
    if not agent_id or not user_id:
        raise ValueError("Agent ID and User ID are required")

    async with httpx.AsyncClient(timeout=60.0) as client:
        chat_request = {
            "user_id": user_id,
            "agent_id": agent_id,
            "user_message": {
                "content": "Hello!",
                "name": "test_user",
                "attachments": [],
            },
            "force_reply": True,
        }

        response = await client.post(
            f"{API_URL}/chat", json=chat_request, headers=HEADERS
        )
        print("API Response:", response.status_code)
        print(json.dumps(response.json(), indent=2))


async def main(
    type: str, agent_id: Optional[str] = None, user_id: Optional[str] = None
):
    if not os.getenv("EDEN_ADMIN_KEY"):
        raise ValueError("EDEN_ADMIN_KEY environment variable not set")

    if type == "chat":
        print("\nTesting chat endpoint...")
        await test_chat(agent_id, user_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Eve API")
    parser.add_argument("--type", help="Type of test", required=True, choices=["chat"])
    parser.add_argument("--user_id", help="User ID", required=True)
    parser.add_argument("--agent_id", help="Agent ID", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.type, args.agent_id, args.user_id))
