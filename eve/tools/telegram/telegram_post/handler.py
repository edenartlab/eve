from loguru import logger
from telegram import Bot

from eve.agent import Agent
from eve.agent.deployments import Deployment
from eve.agent.session.models import Session
from eve.tool import ToolContext


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    deployment = Deployment.load(agent=agent.id, platform="telegram")
    if not deployment:
        raise Exception("No valid Telegram deployments found")

    # Get allowed chats from deployment config
    allowed_chats = deployment.config.telegram.topic_allowlist
    if not allowed_chats:
        raise Exception("No chats configured for this deployment")

    # Get channel ID and content from args (using channel_id to match discord_post)
    channel_id = context.args.get("channel_id")
    content = context.args.get("content")
    media_urls = context.args.get("media_urls", [])

    if not channel_id:
        raise Exception("channel_id is required")

    # Validate that the channel matches the session's Telegram target
    # This prevents agents from posting to wrong chats based on memories
    if context.session:
        try:
            session = Session.from_mongo(context.session)
            if session and session.telegram_chat_id:
                if channel_id != session.telegram_chat_id:
                    raise Exception(
                        f"This Telegram session only allows posting to chat {session.telegram_chat_id}. "
                        f"You tried to post to chat {channel_id}. "
                        f"Please use channel_id={session.telegram_chat_id}."
                    )
        except Exception as e:
            # If it's our validation error, re-raise it
            if "This Telegram session" in str(e):
                raise
            # Otherwise log and continue (session lookup failed)
            logger.warning(f"telegram_post: Failed to validate session target: {e}")

    # Verify the channel is in the allowlist
    if not any(str(chat.id) == channel_id for chat in allowed_chats):
        raise Exception(f"Channel {channel_id} is not in the allowlist")

    # Create Telegram bot
    bot = Bot(token=deployment.secrets.telegram.token)

    try:
        # Send media if provided
        if media_urls:
            from telegram import InputMediaPhoto, InputMediaVideo

            media_group = []
            for i, url in enumerate(
                media_urls[:10]
            ):  # Telegram allows up to 10 media items
                # Determine media type based on extension
                video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")

                # Add caption to first media item only
                caption = content if i == 0 and content else None

                if any(url.lower().endswith(ext) for ext in video_extensions):
                    media = InputMediaVideo(media=url, caption=caption)
                else:
                    media = InputMediaPhoto(media=url, caption=caption)

                media_group.append(media)

            messages = await bot.send_media_group(chat_id=channel_id, media=media_group)
            message = messages[0]  # Use first message for response
        else:
            # Send text message
            message = await bot.send_message(chat_id=channel_id, text=content)

        result = {
            "output": [
                {
                    "message_id": message.message_id,
                    "chat_id": message.chat.id,
                    "url": f"https://t.me/{message.chat.username}/{message.message_id}"
                    if hasattr(message.chat, "username") and message.chat.username
                    else None,
                }
            ]
        }

        await bot.close()
        return result

    except Exception as e:
        try:
            await bot.close()
        except Exception:
            pass  # Ignore errors closing bot
        raise e
