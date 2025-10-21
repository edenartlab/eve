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

        # Validate bot token and get bot info
        try:
            bot = Bot(secrets.telegram.token)
            bot_info = await bot.get_me()

            # Store the actual bot username in config
            if not config.telegram:
                from eve.agent.session.models import DeploymentSettingsTelegram
                config.telegram = DeploymentSettingsTelegram()

            config.telegram.bot_username = bot_info.username
            logger.info(f"[TELEGRAM-PREDEPLOY] Bot username set to: {bot_info.username}")

        except Exception as e:
            logger.error(f"Token validation failed: {e}")
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

        # Point webhook to the Modal gateway service
        # Modal URL format: https://edenartlab--discord-gateway-v2-{db}-gateway-app.modal.run
        gateway_url = os.getenv("GATEWAY_URL")
        if not gateway_url:
            raise Exception("GATEWAY_URL is not set")

        webhook_url = f"{gateway_url}/telegram/webhook"

        # Set webhook directly via HTTP instead of using telegram library
        import aiohttp

        telegram_api_url = f"https://api.telegram.org/bot{self.deployment.secrets.telegram.token}/setWebhook"

        payload = {
            "url": webhook_url,
            "secret_token": self.deployment.secrets.telegram.webhook_secret,
            "drop_pending_updates": "true",
            "max_connections": "100",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(telegram_api_url, data=payload) as response:
                response_data = await response.json()
                logger.info(f"Telegram API response: {response_data}")

                if not response_data.get("ok"):
                    error_msg = response_data.get("description", "Unknown error")
                    raise Exception(f"Failed to set Telegram webhook: {error_msg}")

                logger.info("Webhook set successfully via HTTP")

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
                f"[TELEGRAM-INTERACT] Processing Telegram webhook update: {update_data.get('update_id')}"
            )
            logger.info(f"[TELEGRAM-INTERACT] Full update data: {json.dumps(update_data, indent=2)}")

            # Extract message data
            message = update_data.get("message")
            if not message:
                logger.info("[TELEGRAM-INTERACT] No message in update, ignoring")
                return

            # Skip bot messages
            if message.get("from", {}).get("is_bot", False):
                logger.info("[TELEGRAM-INTERACT] Skipping bot message")
                return

            chat_id = message.get("chat", {}).get("id")
            chat_type = message.get("chat", {}).get("type")
            message_thread_id = message.get("message_thread_id")
            is_dm = chat_type == "private"

            logger.info(f"[TELEGRAM-INTERACT] Chat ID: {chat_id}, Type: {chat_type}, Thread ID: {message_thread_id}, Is DM: {is_dm}")

            # NEVER respond to DMs - only groups and topics
            if is_dm:
                logger.info(f"[TELEGRAM-INTERACT] Skipping DM from user {message.get('from', {}).get('username')} - bot does not respond to DMs")
                return

            # Check allowlist if it exists
            if self.deployment.config and self.deployment.config.telegram:
                allowlist = self.deployment.config.telegram.topic_allowlist or []
                logger.info(f"[TELEGRAM-INTERACT] Topic allowlist configured: {[item.id for item in allowlist] if allowlist else 'None'}")
                if allowlist:
                    current_id = (
                        f"{chat_id}_{message_thread_id}"
                        if message_thread_id
                        else str(chat_id)
                    )
                    logger.info(f"[TELEGRAM-INTERACT] Checking current_id '{current_id}' against allowlist")
                    if not any(item.id == current_id for item in allowlist):
                        logger.info(f"[TELEGRAM-INTERACT] Chat {current_id} not in allowlist - skipping")
                        return
                    logger.info(f"[TELEGRAM-INTERACT] Chat {current_id} IS in allowlist - continuing")
            else:
                logger.info("[TELEGRAM-INTERACT] No allowlist configured, accepting all chats")

            # Get user info
            from_user = message.get("from", {})
            user_id = str(from_user.get("id"))
            username = from_user.get("username", "unknown")
            user = User.from_telegram(user_id, username)
            logger.info(f"[TELEGRAM-INTERACT] User: {username} (ID: {user_id})")

            # Get agent and bot info for mention checking
            agent = Agent.from_mongo(self.deployment.agent)
            logger.info(f"[TELEGRAM-INTERACT] Agent username: {agent.username}")

            # Get bot ID to check if replies are to this bot
            from telegram import Bot
            bot = Bot(self.deployment.secrets.telegram.token)
            bot_info = await bot.get_me()
            bot_id = bot_info.id

            # Get bot username from config (stored during deployment)
            bot_username = None
            if self.deployment.config and self.deployment.config.telegram:
                bot_username = self.deployment.config.telegram.bot_username

            # Fallback to fetched bot info if not in config
            if not bot_username:
                bot_username = bot_info.username
                logger.warning(f"[TELEGRAM-INTERACT] Bot username not in config, using fetched: {bot_username}")
            else:
                logger.info(f"[TELEGRAM-INTERACT] Bot username from config: {bot_username}")

            # Ensure bot_username has @ prefix for comparison
            if bot_username and not bot_username.startswith("@"):
                bot_username = f"@{bot_username}"

            logger.info(f"[TELEGRAM-INTERACT] Bot ID: {bot_id}, Bot username: {bot_username}")

            # Check if we should reply (only in groups/channels)
            # Reply to:
            # 1. Replies to the bot's messages
            # 2. @mentions of the bot's username
            force_reply = False

            # Check if this is a reply to bot's message
            reply_to = message.get("reply_to_message")
            if reply_to:
                replied_to_user = reply_to.get("from", {})
                replied_to_bot_id = replied_to_user.get("id")
                replied_to_is_bot = replied_to_user.get("is_bot")
                logger.info(f"[TELEGRAM-INTERACT] This is a reply to message from user ID: {replied_to_bot_id}, is_bot: {replied_to_is_bot}")

                if replied_to_is_bot and replied_to_bot_id == bot_id:
                    force_reply = True
                    logger.info(f"[TELEGRAM-INTERACT] ✓ Message is reply to THIS bot (ID {bot_id}) - WILL RESPOND")
                else:
                    logger.info(f"[TELEGRAM-INTERACT] ✗ Message is reply to different user (ID {replied_to_bot_id}) - will not respond based on reply")
            else:
                logger.info("[TELEGRAM-INTERACT] Not a reply to any message")

            # Check if bot is mentioned via @username in text
            if not force_reply:
                text = message.get("text") or message.get("caption") or ""
                logger.info(f"[TELEGRAM-INTERACT] Message text: '{text}'")
                logger.info(f"[TELEGRAM-INTERACT] Checking if '{bot_username}' is in text")

                if bot_username.lower() in text.lower():
                    force_reply = True
                    logger.info(f"[TELEGRAM-INTERACT] ✓ Bot username '{bot_username}' found in text - WILL RESPOND")
                else:
                    logger.info(f"[TELEGRAM-INTERACT] ✗ Bot username '{bot_username}' NOT found in text")

            # Also check entities for mentions
            if not force_reply:
                entities = message.get("entities", [])
                logger.info(f"[TELEGRAM-INTERACT] Checking {len(entities)} entities for mentions")

                for i, entity in enumerate(entities):
                    entity_type = entity.get("type")
                    logger.info(f"[TELEGRAM-INTERACT] Entity {i}: type='{entity_type}'")

                    if entity_type == "mention":
                        # Extract the mentioned username from text
                        offset = entity.get("offset", 0)
                        length = entity.get("length", 0)
                        text = message.get("text") or message.get("caption") or ""
                        mentioned = text[offset:offset + length].lower()
                        logger.info(f"[TELEGRAM-INTERACT] Mention entity found: '{mentioned}'")

                        if mentioned == bot_username.lower():
                            force_reply = True
                            logger.info(f"[TELEGRAM-INTERACT] ✓ Bot mentioned in entities ('{mentioned}') - WILL RESPOND")
                            break
                        else:
                            logger.info(f"[TELEGRAM-INTERACT] ✗ Different user mentioned: '{mentioned}'")
                    elif entity_type == "text_mention":
                        # Direct user mention (not via @username)
                        mentioned_user = entity.get("user", {})
                        mentioned_user_id = mentioned_user.get("id")
                        logger.info(f"[TELEGRAM-INTERACT] Text mention entity found for user ID: {mentioned_user_id}")

                        if mentioned_user_id == bot_id:
                            force_reply = True
                            logger.info(f"[TELEGRAM-INTERACT] ✓ Bot mentioned via text_mention (ID {bot_id}) - WILL RESPOND")
                            break
                        else:
                            logger.info(f"[TELEGRAM-INTERACT] ✗ Different user mentioned via text_mention: {mentioned_user_id}")

            # Skip if not force_reply (not a reply to bot, not mentioned)
            if not force_reply:
                logger.info("[TELEGRAM-INTERACT] ✗✗✗ NOT responding - message is neither a reply to bot nor a mention of bot - SKIPPING")
                return

            logger.info("[TELEGRAM-INTERACT] ✓✓✓ WILL RESPOND to this message")

            # Process text and attachments
            text = message.get("text", "")
            attachments = []

            # Handle photos
            photos = message.get("photo", [])
            if photos:
                logger.info(f"[TELEGRAM-INTERACT] Processing {len(photos)} photos")
                # Get the largest photo (last in array)
                largest_photo = photos[-1]
                file_id = largest_photo.get("file_id")

                file = await bot.get_file(file_id)
                photo_url = file.file_path
                attachments.append(photo_url)
                logger.info(f"[TELEGRAM-INTERACT] Added photo attachment: {photo_url}")

                # Use caption as text if available
                if message.get("caption"):
                    text = message.get("caption")
                    logger.info(f"[TELEGRAM-INTERACT] Using photo caption as text: '{text}'")

            # Clean message text (remove bot mention)
            cleaned_text = text
            if text:
                pattern = rf"\s*{re.escape(bot_username)}\b"
                cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
                logger.info(f"[TELEGRAM-INTERACT] Original text: '{text}'")
                logger.info(f"[TELEGRAM-INTERACT] Cleaned text: '{cleaned_text}'")

            # Create session key
            session_key = (
                f"telegram-{chat_id}-topic-{message_thread_id}"
                if message_thread_id
                else f"telegram-{chat_id}"
            )
            logger.info(f"[TELEGRAM-INTERACT] Session key: {session_key}")

            # Try to load existing session
            from eve.agent.session.models import Session

            session = None
            try:
                logger.info(f"[TELEGRAM-INTERACT] Attempting to load session with key: {session_key}")
                session = Session.load(session_key=session_key)
                logger.info(f"[TELEGRAM-INTERACT] Found existing session: {session.id}")

                # Reactivate if needed
                if hasattr(session, "deleted") and session.deleted:
                    logger.info(f"[TELEGRAM-INTERACT] Session was deleted, reactivating")
                    session.deleted = False
                    session.status = "active"
                    session.save()
                elif hasattr(session, "status") and session.status == "archived":
                    logger.info(f"[TELEGRAM-INTERACT] Session was archived, reactivating")
                    session.status = "active"
                    session.save()
            except eve.mongo.MongoDocumentNotFound:
                logger.info(f"[TELEGRAM-INTERACT] No existing session found, will create new one")
                pass

            # Build session request
            from eve.api.api_requests import SessionCreationArgs

            api_url = os.getenv("EDEN_API_URL")
            logger.info(f"[TELEGRAM-INTERACT] API URL: {api_url}")

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
                logger.info(f"[TELEGRAM-INTERACT] Using existing session ID: {session.id}")
            else:
                session_request.creation_args = SessionCreationArgs(
                    owner_id=str(user.id),
                    agents=[str(self.deployment.agent)],
                    title=f"Telegram {session_key}",
                    session_key=session_key,
                    platform="telegram",
                )
                logger.info(f"[TELEGRAM-INTERACT] Creating new session with key: {session_key}")

            logger.info(f"[TELEGRAM-INTERACT] Session request payload: {json.dumps(session_request.model_dump(), indent=2)}")

            # Send to sessions API
            logger.info(f"[TELEGRAM-INTERACT] Sending request to {api_url}/sessions/prompt")
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
                    logger.info(f"[TELEGRAM-INTERACT] Response status: {response.status}")
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"[TELEGRAM-INTERACT] Session request failed: {error_text}")
                        raise Exception(f"Failed to process message: {error_text}")

                    response_data = await response.text()
                    logger.info(f"[TELEGRAM-INTERACT] Response data: {response_data}")
                    logger.info(
                        f"[TELEGRAM-INTERACT] ✓✓✓ Successfully handled Telegram interaction for deployment {self.deployment.id}"
                    )

        except Exception as e:
            logger.error(
                f"[TELEGRAM-INTERACT] ✗✗✗ ERROR handling Telegram interaction: {str(e)}", exc_info=True
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
        # Get bot username from config (stored during deployment)
        bot_username = None
        if deployment.config and deployment.config.telegram:
            bot_username = deployment.config.telegram.bot_username

        # Fallback to fetching bot info if not in config
        if not bot_username:
            from telegram import Bot
            bot = Bot(deployment.secrets.telegram.token)
            bot_info = await bot.get_me()
            bot_username = bot_info.username

        # Ensure bot_username has @ prefix for comparison
        if bot_username and not bot_username.startswith("@"):
            bot_username = f"@{bot_username}"

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
