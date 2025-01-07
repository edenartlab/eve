import os
import aiohttp
import modal
import logging

logger = logging.getLogger(__name__)

db = os.getenv("DB", "STAGE").upper()
if db not in ["PROD", "STAGE"]:
    raise Exception(f"Invalid environment: {db}. Must be PROD or STAGE")
API_URL = os.getenv("EDEN_API_URL")

app = modal.App(
    name=f"[AGENT_USERNAME]-cron-[CRON_ID]-{db}",
    secrets=[
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
    ],
)


@app.function(schedule=modal.Period(days=1))
async def chat_cron():
    async with aiohttp.ClientSession() as session:
        request_data = {
            "user_id": "[USER_ID]",
            "agent_id": "[AGENT_ID]",
            "thread_id": "[THREAD_ID]",
            "force_reply": True,
            "user_message": {
                "content": "[CONTENT]",
                "name": "[NAME]",
                "attachments": "[ATTACHMENTS]",
            },
            "update_config": "[UPDATE_CONFIG]",
        }

        print(f"Sending request: {request_data}")
        async with session.post(
            f"{API_URL}/chat",
            json=request_data,
            headers={"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"},
        ) as response:
            if response.status != 200:
                logger.error(f"Failed to send request: {response.status}")
