import os
import re
import json
from typing import Optional

from ably import AblyRest
from fastapi import Request

from eve.agent.agent import Agent
from eve.agent.session.models import (
    ChatMessageRequestInput,
    SessionUpdateConfig,
    Deployment,
    DeploymentSecrets,
    DeploymentConfig,
)
from eve.api.api_requests import ChatRequest, PromptSessionRequest
from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import UpdateType
from eve.user import User
from eve.utils import prepare_result
from loguru import logger

db = os.getenv("DB", "STAGE").upper()


class TelegramClient(PlatformClient):
    TOOLS = [
        "telegram_post",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Telegram token, generate webhook secret and add Telegram tools"""
        from telegram import Bot
        import secrets as python_secrets

        # Validate bot token
        try:
            bot = Bot(secrets.telegram.token)
            await bot.get_me()
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

        # Point webhook to the Modal gateway service
        # Modal URL format: https://edenartlab--discord-gateway-v2-{db}-gateway-app.modal.run
        gateway_url = os.getenv("GATEWAY_URL")
        if not gateway_url:
            raise Exception("GATEWAY_URL is not set")

        logger.info(f"Gateway URL: {gateway_url}")
        webhook_url = f"{gateway_url}/telegram/webhook"
        logger.info(f"Webhook URL: {webhook_url}")

        # Update bot webhook
        bot = Bot(self.deployment.secrets.telegram.token)

        # First check the bot is valid
        try:
            bot_info = await bot.get_me()
            logger.info(f"Bot info: {bot_info}")
        except Exception as e:
            logger.error(f"Failed to get bot info: {e}")
            raise Exception(f"Invalid bot token: {e}")

        # Try to get current webhook info
        try:
            current_webhook = await bot.get_webhook_info()
            logger.info(f"Current webhook: {current_webhook}")
        except Exception as e:
            logger.warning(f"Failed to get current webhook info: {e}")

        # Set webhook with detailed error logging
        try:
            # Try with minimal parameters first
            logger.info("Attempting to set webhook with minimal parameters...")
            response = await bot.set_webhook(url=webhook_url)
            logger.info(f"Set webhook response (minimal): {response}")

            # If that worked, update with full parameters
            if response:
                logger.info("Updating webhook with full parameters...")
                response = await bot.set_webhook(
                    url=webhook_url,
                    secret_token=self.deployment.secrets.telegram.webhook_secret,
                    drop_pending_updates=True,
                    max_connections=100,
                )
                logger.info(f"Set webhook response (full): {response}")
        except Exception as e:
            logger.error(f"Failed to set webhook: {e}")
            logger.error(f"Webhook URL: {webhook_url}")
            logger.error(f"URL length: {len(webhook_url)}")
            logger.error(f"Secret token length: {len(self.deployment.secrets.telegram.webhook_secret) if self.deployment.secrets.telegram.webhook_secret else 0}")
            raise

        if not response:
            raise Exception("Failed to set Telegram webhook")

        # Notify gateway about the new deployment
        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-v2-{db}")

            await channel.publish(
                "command",
                {
                    "command": "register_telegram",
                    "deployment_id": str(self.deployment.id),
                    "token": self.deployment.secrets.telegram.token,
                },
            )
        except Exception as e:
            raise Exception(f"Failed to notify gateway service for Telegram: {e}")

    async def stop(self) -> None:
        """Stop Telegram client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-v2-{db}")

            await channel.publish(
                "command",
                {
                    "command": "unregister_telegram",
                    "deployment_id": str(self.deployment.id),
                },
            )

            self.remove_tools()
        except Exception as e:
            logger.error(
                f"Failed to notify gateway service for Telegram unregistration: {e}"
            )

    async def interact(self, request: Request) -> None:
        """Handle Telegram webhook interactions"""
        try:
            import aiohttp
            import eve.mongo

            # Parse the webhook update
            update_data = await request.json()
            logger.info(
                f"Processing Telegram webhook update: {update_data.get('update_id')}"
            )

            # Extract message data
            message = update_data.get("message")
            if not message:
                logger.debug("No message in update, ignoring")
                return

            # Skip bot messages
            if message.get("from", {}).get("is_bot", False):
                logger.debug("Skipping bot message")
                return

            chat_id = message.get("chat", {}).get("id")
            message_thread_id = message.get("message_thread_id")

            # Check allowlist if it exists
            if self.deployment.config and self.deployment.config.telegram:
                allowlist = self.deployment.config.telegram.topic_allowlist or []
                if allowlist:
                    current_id = (
                        f"{chat_id}_{message_thread_id}"
                        if message_thread_id
                        else str(chat_id)
                    )
                    if not any(item.id == current_id for item in allowlist):
                        logger.debug(f"Chat {current_id} not in allowlist")
                        return

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

                bot = Bot(self.deployment.secrets.telegram.token)
                file = await bot.get_file(file_id)
                photo_url = file.file_path
                attachments.append(photo_url)

                # Use caption as text if available
                if message.get("caption"):
                    text = message.get("caption")

            # Clean message text (remove bot mention)
            agent = Agent.from_mongo(self.deployment.agent)
            cleaned_text = text
            if text:
                bot_username = f"@{agent.username.lower()}_bot"
                pattern = rf"\s*{re.escape(bot_username)}\b"
                cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

            # Create session key
            is_dm = message.get("chat", {}).get("type") == "private"
            if is_dm:
                session_key = f"telegram-dm-{user_id}"
            else:
                session_key = (
                    f"telegram-{chat_id}-topic-{message_thread_id}"
                    if message_thread_id
                    else f"telegram-{chat_id}"
                )

            # Try to load existing session
            from eve.agent.session.models import Session

            session = None
            try:
                session = Session.load(session_key=session_key)
                # Reactivate if needed
                if hasattr(session, "deleted") and session.deleted:
                    session.deleted = False
                    session.status = "active"
                    session.save()
                elif hasattr(session, "status") and session.status == "archived":
                    session.status = "active"
                    session.save()
            except eve.mongo.MongoDocumentNotFound:
                pass

            # Build session request
            from eve.api.api_requests import SessionCreationArgs

            api_url = os.getenv("EDEN_API_URL")
            session_request = PromptSessionRequest(
                user_id=str(user.id),
                actor_agent_ids=[str(self.deployment.agent)],
                message=ChatMessageRequestInput(
                    content=cleaned_text, sender_name=username, attachments=attachments
                ),
                update_config=SessionUpdateConfig(
                    update_endpoint=f"{api_url}/v2/deployments/emission",
                    deployment_id=str(self.deployment.id),
                    telegram_chat_id=str(chat_id),
                    telegram_message_id=str(message.get("message_id")),
                    telegram_thread_id=str(message_thread_id)
                    if message_thread_id
                    else None,
                ),
            )

            if session:
                session_request.session_id = str(session.id)
            else:
                session_request.creation_args = SessionCreationArgs(
                    owner_id=str(user.id),
                    agents=[str(self.deployment.agent)],
                    title=f"Telegram {session_key}",
                    session_key=session_key,
                    platform="telegram",
                )

            # Send to sessions API
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(
                    f"{api_url}/sessions/prompt",
                    json=session_request.model_dump(),
                    headers={
                        "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                        "Content-Type": "application/json",
                        "X-Client-Platform": "telegram",
                        "X-Client-Deployment-Id": str(self.deployment.id),
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Session request failed: {error_text}")
                        raise Exception(f"Failed to process message: {error_text}")

                    logger.info(
                        f"Successfully handled Telegram interaction for deployment {self.deployment.id}"
                    )

        except Exception as e:
            logger.error(
                f"Error handling Telegram interaction: {str(e)}", exc_info=True
            )
            raise

    async def handle_emission(self, emission) -> None:
        """Handle an emission from the platform client"""
        try:
            if not self.deployment:
                raise ValueError("Deployment is required for handle_emission")

            # Extract context from update_config
            chat_id = emission.update_config.telegram_chat_id
            message_id = emission.update_config.telegram_message_id
            thread_id = emission.update_config.telegram_thread_id

            if not chat_id:
                logger.error("Missing telegram_chat_id in update_config")
                return

            update_type = emission.type
            logger.debug(
                f"Handling emission type: {update_type} (type: {type(update_type)})"
            )

            # Initialize Telegram Bot
            from telegram import Bot

            bot = Bot(self.deployment.secrets.telegram.token)

            # Build message kwargs
            message_kwargs = {
                "chat_id": int(chat_id),
            }

            if message_id:
                message_kwargs["reply_to_message_id"] = int(message_id)

            if thread_id:
                message_kwargs["message_thread_id"] = int(thread_id)

            # Compare with string value instead of enum
            if (
                update_type == UpdateType.ASSISTANT_MESSAGE.value
                or update_type == UpdateType.ASSISTANT_MESSAGE
            ):
                content = emission.content
                if content:
                    await bot.send_message(text=content, **message_kwargs)
                    logger.info(f"Sent Telegram message to chat {chat_id}")

            elif (
                update_type == UpdateType.TOOL_COMPLETE.value
                or update_type == UpdateType.TOOL_COMPLETE
            ):
                result = emission.result
                if not result:
                    logger.debug("No tool result to post")
                    return

                # Process result to extract media URLs
                processed_result = prepare_result(json.loads(result))

                if (
                    processed_result.get("result")
                    and len(processed_result["result"]) > 0
                    and "output" in processed_result["result"][0]
                ):
                    outputs = processed_result["result"][0]["output"]

                    # Send each output as appropriate media type
                    for output in outputs[:4]:  # Send up to 4 media items
                        if isinstance(output, dict) and "url" in output:
                            url = output["url"]
                            video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")

                            if any(
                                url.lower().endswith(ext) for ext in video_extensions
                            ):
                                await bot.send_video(video=url, **message_kwargs)
                            else:
                                await bot.send_photo(photo=url, **message_kwargs)

                    logger.info(f"Sent Telegram media to chat {chat_id}")
                else:
                    logger.warning(
                        "Unexpected tool result structure for Telegram emission"
                    )

            elif (
                update_type == UpdateType.ERROR.value or update_type == UpdateType.ERROR
            ):
                error_msg = emission.error or "Unknown error occurred"
                await bot.send_message(text=f"Error: {error_msg}", **message_kwargs)
                logger.info(f"Sent Telegram error message to chat {chat_id}")

            else:
                logger.debug(f"Ignoring emission type: {update_type}")

        except Exception as e:
            logger.error(f"Error handling Telegram emission: {str(e)}", exc_info=True)
            raise


async def create_telegram_session_request(
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

    # Clean message text (remove bot mention)
    cleaned_text = text
    if text:
        bot_username = f"@{agent.username.lower()}_bot"
        pattern = rf"\s*{re.escape(bot_username)}\b"
        cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return PromptSessionRequest(
        user_id=str(user.id),
        actor_agent_ids=[str(deployment.agent)],
        message=ChatMessageRequestInput(
            content=cleaned_text, sender_name=username, attachments=attachments
        ),
        update_config=SessionUpdateConfig(
            update_endpoint=f"{os.getenv('EDEN_API_URL')}/emissions/platform/telegram",
            deployment_id=str(deployment.id),
            telegram_chat_id=str(chat_id),
            telegram_message_id=str(message.get("message_id")),
            telegram_thread_id=str(message_thread_id) if message_thread_id else None,
        ),
    )
