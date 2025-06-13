import os
import re
from typing import List, Optional

from ably import AblyRest
from pydantic import BaseModel


from eve.agent.agent import Agent
from eve.api.api_requests import ChatRequest
from eve.api.errors import APIError
from eve.agent.deployments import (
    Deployment,
    PlatformClient,
    DeploymentSecrets,
    DeploymentConfig,
)
from eve.user import User

db = os.getenv("DB", "STAGE").upper()


class TelegramAllowlistItem(BaseModel):
    id: str
    note: Optional[str] = None


class DeploymentSecretsTelegram(BaseModel):
    token: str
    webhook_secret: Optional[str] = None


class DeploymentSettingsTelegram(BaseModel):
    topic_allowlist: Optional[List[TelegramAllowlistItem]] = None


class TelegramClient(PlatformClient):
    TOOLS = {
        "telegram_post": {},
    }

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Telegram token, generate webhook secret and add Telegram tools"""
        from telegram import Bot
        import secrets as python_secrets

        # Validate bot token
        try:
            bot = Bot(secrets.telegram.token)
            bot_info = await bot.get_me()
            print(f"Verified Telegram bot: {bot_info.username}")
        except Exception as e:
            raise APIError(f"Invalid Telegram token: {str(e)}", status_code=400)

        webhook_secret = python_secrets.token_urlsafe(32)
        secrets.telegram.webhook_secret = webhook_secret

        # Add Telegram tools to agent
        self.add_tools()

        return secrets, config

    async def postdeploy(self) -> None:
        """Set Telegram webhook and notify gateway"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        from telegram import Bot

        webhook_url = f"{os.getenv('EDEN_API_URL')}/updates/platform/telegram"

        # Update bot webhook
        response = await Bot(self.deployment.secrets.telegram.token).set_webhook(
            url=webhook_url,
            secret_token=self.deployment.secrets.telegram.webhook_secret,
            drop_pending_updates=True,
            max_connections=100,
        )

        if not response:
            raise Exception("Failed to set Telegram webhook")

        # Notify gateway about the new deployment
        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-{db}")

            await channel.publish(
                "command",
                {
                    "command": "register_telegram",
                    "deployment_id": str(self.deployment.id),
                    "token": self.deployment.secrets.telegram.token,
                },
            )
            print(
                f"Sent Telegram registration command for deployment {self.deployment.id} via Ably"
            )
        except Exception as e:
            raise Exception(f"Failed to notify gateway service for Telegram: {e}")

    async def stop(self) -> None:
        """Stop Telegram client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-{db}")

            await channel.publish(
                "command",
                {
                    "command": "unregister_telegram",
                    "deployment_id": str(self.deployment.id),
                },
            )
            print(
                f"Sent unregister command for Telegram deployment {self.deployment.id} via Ably"
            )

            self.remove_tools()
        except Exception as e:
            print(f"Failed to notify gateway service for Telegram unregistration: {e}")


async def create_telegram_chat_request(
    update_data: dict, deployment: Deployment
) -> Optional[ChatRequest]:
    message = update_data.get("message", {})
    if not message:
        return None

    chat_id = message.get("chat", {}).get("id")
    message_thread_id = message.get("message_thread_id")

    # Check allowlist if it exists
    if deployment.config and deployment.config.telegram:
        allowlist = deployment.config.telegram.topic_allowlist or []
        if allowlist:
            current_id = (
                f"{chat_id}_{message_thread_id}" if message_thread_id else str(chat_id)
            )
            if not any(item.id == current_id for item in allowlist):
                return None

    agent = Agent.from_mongo(deployment.agent)

    # Get user info
    from_user = message.get("from", {})
    user_id = str(from_user.get("id"))
    username = from_user.get("username", "unknown")
    user = User.from_telegram(user_id, username)

    # Process text and attachments
    text = message.get("text", "")
    attachments = []

    # Handle photos
    photos = message.get("photo", [])
    if photos:
        # Get the largest photo (last in array)
        largest_photo = photos[-1]
        file_id = largest_photo.get("file_id")

        # Initialize bot to get file path
        from telegram import Bot

        bot = Bot(deployment.secrets.telegram.token)
        file = await bot.get_file(file_id)
        photo_url = file.file_path
        attachments.append(photo_url)

        # Use caption as text if available
        if message.get("caption"):
            text = message.get("caption")

    # Create thread
    thread_key = (
        f"telegram-{chat_id}-topic-{message_thread_id}"
        if message_thread_id
        else f"telegram-{chat_id}"
    )
    thread = agent.request_thread(key=thread_key)

    # Clean message text (remove bot mention)
    cleaned_text = text
    if text:
        bot_username = f"@{agent.username.lower()}_bot"
        pattern = rf"\s*{re.escape(bot_username)}\b"
        cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return {
        "user_id": str(user.id),
        "agent_id": str(deployment.agent),
        "thread_id": str(thread.id),
        "user_is_bot": from_user.get("is_bot", False),
        "force_reply": True,
        "user_message": {
            "content": cleaned_text,
            "name": username,
            "attachments": attachments,
        },
        "update_config": {
            "update_endpoint": f"{os.getenv('EDEN_API_URL')}/emissions/platform/telegram",
            "deployment_id": str(deployment.id),
            "telegram_chat_id": str(chat_id),
            "telegram_message_id": str(message.get("message_id")),
            "telegram_thread_id": str(message_thread_id) if message_thread_id else None,
        },
    }
