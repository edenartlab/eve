from eve.agent.deployments import Deployment
from eve.agent import Agent
from eve.tool import ToolContext
from telegram import Bot


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
