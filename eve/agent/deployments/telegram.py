import os
from typing import List, Optional

from ably import AblyRest
from pydantic import BaseModel


from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient, DeploymentSecrets, DeploymentConfig

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
