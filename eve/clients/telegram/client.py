import os
import argparse
import re
from ably import AblyRealtime
import aiohttp

# import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    Application,
)

from ...clients import common
from ...llm import UserMessage, async_prompt_thread, UpdateType
from ...eden_utils import prepare_result
from ...agent import Agent
from ...user import User
from ...tool import get_tools_from_mongo
from ...models import ClientType


async def handler_mention_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Determine if the bot is mentioned or replied to.
    """
    message = update.message
    chat_type = message.chat.type
    is_direct_message = chat_type == "private"
    bot_username = (await context.bot.get_me()).username.lower()

    is_bot_mentioned = any(
        entity.type == "mention"
        and message.text[entity.offset : entity.offset + entity.length].lower()
        == f"@{bot_username}"
        for entity in message.entities or []
    )

    is_replied_to_bot = bool(
        message.reply_to_message
        and message.reply_to_message.from_user.username.lower() == bot_username
    )
    return (
        message.chat.id,
        chat_type,
        is_direct_message,
        is_bot_mentioned,
        is_replied_to_bot,
    )


def get_user_info(update: Update):
    """
    Retrieve user information from the update.
    """
    user = update.message.from_user
    full_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return (
        user.id,
        user.username,
        user.first_name or "",
        user.last_name or "",
        full_name,
    )


def remove_bot_mentions(message_text: str, bot_username: str) -> str:
    """
    Remove bot mentions from the message text.
    """
    pattern = rf"\s*@{re.escape(bot_username)}\b"
    return (
        re.sub(pattern, "", message_text, flags=re.IGNORECASE)
        .strip()
        .replace("  ", " ")
    )


def replace_bot_mentions(message_text: str, bot_username: str, replacement: str) -> str:
    """
    Replace bot mentions with a replacement string.
    """
    pattern = rf"\s*@{re.escape(bot_username)}\b"
    return (
        re.sub(pattern, replacement, message_text, flags=re.IGNORECASE)
        .strip()
        .replace("  ", " ")
    )


async def send_response(
    message_type: str, chat_id: int, response: list, context: ContextTypes.DEFAULT_TYPE
):
    """
    Send messages, photos, or videos based on the type of response.
    """
    for item in response:
        if item.startswith("https://"):
            # Common video file extensions
            video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
            if any(item.lower().endswith(ext) for ext in video_extensions):
                # logging.info(f"Sending video to {chat_id}")
                await context.bot.send_video(chat_id=chat_id, video=item)
            else:
                # logging.info(f"Sending photo to {chat_id}")
                await context.bot.send_photo(chat_id=chat_id, photo=item)
        else:
            # logging.info(f"Sending message to {chat_id}")
            await context.bot.send_message(chat_id=chat_id, text=item)


class EdenTG:
    def __init__(self, token: str, agent: Agent, db: str = "STAGE"):
        self.token = token
        self.agent = agent
        self.db = db
        self.tools = get_tools_from_mongo(db=self.db)
        self.known_users = {}
        self.known_threads = {}
        self.channel_name = common.get_ably_channel_name(
            agent.name, ClientType.TELEGRAM
        )

        # Don't initialize Ably here - we'll do it in setup_ably
        self.ably_client = None
        self.channel = None

    async def initialize(self, application):
        """Initialize the bot including Ably setup"""
        # Setup Ably
        self.ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
        self.channel = self.ably_client.channels.get(self.channel_name)

        # Setup Ably subscriptions
        await self.setup_ably(application)

    def __del__(self):
        """Cleanup when the instance is destroyed"""
        if hasattr(self, "ably_client") and self.ably_client:
            self.ably_client.close()

    async def setup_ably(self, application):
        """Initialize Ably client and subscribe to updates"""

        async def async_callback(message):
            print(f"Received update in Telegram client: {message.data}")

            data = message.data
            if not isinstance(data, dict) or "type" not in data:
                print("Invalid message format:", data)
                return

            update_type = data["type"]
            update_config = data.get("update_config", {})
            telegram_chat_id = update_config.get("telegram_chat_id")

            if not telegram_chat_id:
                print("No telegram_chat_id in update_config:", data)
                return

            print(f"Processing update type: {update_type} for chat: {telegram_chat_id}")

            if update_type == UpdateType.START_PROMPT:
                pass

            elif update_type == UpdateType.ERROR:
                error_msg = data.get("error", "Unknown error occurred")
                await application.bot.send_message(
                    chat_id=telegram_chat_id, text=f"Error: {error_msg}"
                )

            elif update_type == UpdateType.ASSISTANT_MESSAGE:
                content = data.get("content")
                if content:
                    await application.bot.send_message(
                        chat_id=telegram_chat_id, text=content
                    )

            elif update_type == UpdateType.TOOL_COMPLETE:
                result = data.get("result", {})
                result["result"] = prepare_result(result["result"], db=self.db)
                url = result["result"][0]["output"][0]["url"]

                # Determine if it's a video or image
                video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
                if any(url.lower().endswith(ext) for ext in video_extensions):
                    await application.bot.send_video(
                        chat_id=telegram_chat_id, video=url
                    )
                else:
                    await application.bot.send_photo(
                        chat_id=telegram_chat_id, photo=url
                    )
            else:
                print(f"Unknown update type: {update_type}")

        # Subscribe using the async callback
        await self.channel.subscribe(async_callback)
        print(f"Subscribed to Ably channel: {self.channel_name}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handler for the /start command.
        """
        await update.message.reply_text("Hello! I am your asynchronous bot.")

    async def echo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle incoming messages and process bot mentions or direct messages.
        """
        (
            chat_id,
            _,
            is_direct_message,
            is_bot_mentioned,
            is_replied_to_bot,
        ) = await handler_mention_type(update, context)
        user_id, username, _, _, _ = get_user_info(update)

        # Determine message type
        message_type = (
            "dm"
            if is_direct_message
            else "mention"
            if is_bot_mentioned
            else "reply"
            if is_replied_to_bot
            else None
        )
        force_reply = message_type in ["dm", "reply", "mention"]

        # Lookup thread
        thread_key = f"telegram-{chat_id}"
        if thread_key not in self.known_threads:
            self.known_threads[thread_key] = self.agent.request_thread(
                key=thread_key,
                db=self.db,
            )
        thread = self.known_threads[thread_key]

        # Lookup user
        if user_id not in self.known_users:
            self.known_users[user_id] = User.from_telegram(
                user_id, username, db=self.db
            )
        user = self.known_users[user_id]

        # Check if user rate limits
        if common.user_over_rate_limits(user):
            message = (
                "I'm sorry, you've hit your rate limit. Please try again a bit later!",
            )
            await send_response(message_type, chat_id, [message], context)
            return

        # Lookup bot
        me_bot = await context.bot.get_me()

        # Process text or photo messages
        message_text = update.message.text or ""
        attachments = []
        cleaned_text = message_text
        if update.message.photo:
            photo_url = (await update.message.photo[-1].get_file()).file_path
            attachments.append(photo_url)
        else:
            cleaned_text = replace_bot_mentions(
                message_text, me_bot.username, self.agent.name
            )

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        # Make API request
        api_url = os.getenv("EDEN_API_URL")
        request_data = {
            "user_id": str(user.id),
            "agent_id": str(self.agent.id),
            "thread_id": str(thread.id),
            "user_message": {
                "content": cleaned_text,
                "name": username,
                "attachments": attachments,
            },
            "update_config": {
                "sub_channel_name": self.channel_name,
                "telegram_chat_id": str(chat_id),
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/chat",
                json=request_data,
            ) as response:
                if response.status != 200:
                    await send_response(
                        message_type,
                        chat_id,
                        ["Sorry, something went wrong processing your request."],
                        context,
                    )
                    return


def start(env: str, db: str = "STAGE") -> None:
    load_dotenv(env)

    agent_name = os.getenv("EDEN_AGENT_USERNAME")
    agent = Agent.load(agent_name, db=db)

    bot_token = os.getenv("CLIENT_TELEGRAM_TOKEN")
    if not bot_token:
        raise ValueError("CLIENT_TELEGRAM_TOKEN not found in environment variables")

    application = ApplicationBuilder().token(bot_token).build()
    bot = EdenTG(bot_token, agent, db=db)

    # Setup handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.echo))
    application.add_handler(MessageHandler(filters.PHOTO, bot.echo))

    # Create a post init callback to setup Ably
    async def post_init(application: Application) -> None:
        await bot.initialize(application)

    application.post_init = post_init

    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eden Telegram Bot")
    parser.add_argument("--env", help="Path to the .env file to load", default=".env")
    parser.add_argument("--agent", help="Agent username", default="eve")
    parser.add_argument("--db", help="Database to use", default="STAGE")
    args = parser.parse_args()
    start(args.env, args.agent, args.db)
