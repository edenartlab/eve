import os
import argparse
import re
import time
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction

from eve.tool import get_tools_from_mongo
from eve.llm import UserMessage, async_prompt_thread, UpdateType
from eve.thread import Thread
from eve.eden_utils import prepare_result
from eve.agent import Agent
from eve.user import User

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Constants
LONG_RUNNING_TOOLS = {
    "txt2vid",
    "style_mixing",
    "img2vid",
    "vid2vid",
    "video_upscale",
    "vid2vid_sdxl",
    "lora_trainer",
    "animate_3D",
    "reel",
    "story",
}
VIDEO_TOOLS = {
    "animate_3D",
    "txt2vid",
    "img2vid",
    "vid2vid_sdxl",
    "style_mixing",
    "video_upscaler",
    "reel",
    "story",
    "lora_trainer",
}

# Rate limits
HOUR_IMAGE_LIMIT = 50
HOUR_VIDEO_LIMIT = 10
DAY_IMAGE_LIMIT = 200
DAY_VIDEO_LIMIT = 40

hour_timestamps = {}
day_timestamps = {}


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


def user_over_rate_limits(user_id):
    """
    Check if the user has exceeded the rate limits.
    """
    current_time = time.time()

    # Filter timestamps within valid intervals
    hour_timestamps[user_id] = [
        t for t in hour_timestamps.get(user_id, []) if current_time - t["time"] < 3600
    ]
    day_timestamps[user_id] = [
        t for t in day_timestamps.get(user_id, []) if current_time - t["time"] < 86400
    ]

    hour_video_count = sum(
        1 for t in hour_timestamps[user_id] if t["tool"] in VIDEO_TOOLS
    )
    hour_image_count = len(hour_timestamps[user_id]) - hour_video_count

    day_video_count = sum(
        1 for t in day_timestamps[user_id] if t["tool"] in VIDEO_TOOLS
    )
    day_image_count = len(day_timestamps[user_id]) - day_video_count

    return (
        hour_video_count >= HOUR_VIDEO_LIMIT
        or hour_image_count >= HOUR_IMAGE_LIMIT
        or day_video_count >= DAY_VIDEO_LIMIT
        or day_image_count >= DAY_IMAGE_LIMIT
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
                logging.info(f"Sending video to {chat_id}")
                await context.bot.send_video(chat_id=chat_id, video=item)
            else:
                logging.info(f"Sending photo to {chat_id}")
                await context.bot.send_photo(chat_id=chat_id, photo=item)
        else:
            logging.info(f"Sending message to {chat_id}")
            await context.bot.send_message(chat_id=chat_id, text=item)


class EdenTG:
    def __init__(self, token: str, agent: Agent, db: str = "STAGE"):
        self.token = token
        self.agent = agent
        self.db = db
        self.tools = get_tools_from_mongo(db=self.db)
        self.known_users = {}
        self.known_threads = {}

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

        # Get or create user
        if user_id not in self.known_users:
            self.known_users[user_id] = User.from_telegram(
                user_id, username, db=self.db
            )
        user = self.known_users[user_id]

        # Lookup thread
        thread_key = f"telegram-{chat_id}"
        if thread_key not in self.known_threads:
            self.known_threads[thread_key] = self.agent.request_thread(
                key=thread_key, 
                db=self.db,
            )
        thread = self.known_threads[thread_key]
        
        # Process text or photo messages
        message_text = update.message.text or ""
        attachments = []
        if update.message.photo:
            photo_url = (await update.message.photo[-1].get_file()).file_path
            logging.info(f"Received photo from {username}: {photo_url}")
            attachments.append(photo_url)
        else:
            cleaned_text = replace_bot_mentions(
                message_text, (await context.bot.get_me()).username, self.agent.name
            )
            logging.info(f"Received message from {username}: {cleaned_text}")

        user_message = UserMessage(
            name=username, content=cleaned_text, attachments=attachments
        )

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        async for update in async_prompt_thread(
            db=self.db,
            user=user,
            agent=self.agent,
            thread=thread,
            user_messages=user_message,
            tools=self.tools,
            force_reply=force_reply,
        ):
            if update.type == UpdateType.ASSISTANT_MESSAGE:
                await send_response(
                    message_type, chat_id, [update.message.content], context
                )
            elif update.type == UpdateType.TOOL_COMPLETE:
                update.result["result"] = prepare_result(
                    update.result["result"], db=self.db
                )
                url = update.result["result"][0]["output"][0]["url"]
                await send_response(message_type, chat_id, [url], context)
            elif update.type == UpdateType.ERROR:
                await send_response(message_type, chat_id, [update.error], context)


def start(
    env: str, 
    db: str = "STAGE"
) -> None:
    logger.info("Launching Telegram bot...")
    load_dotenv(env)
    
    agent_key = os.environ.get("CLIENT_AGENT_KEY", "eve")
    agent = Agent.load(agent_key, db=db)
    
    logging.info(f"Using agent: {agent.name}")
    bot_token = os.getenv("CLIENT_TELEGRAM_TOKEN")
    application = ApplicationBuilder().token(bot_token).build()
    bot = EdenTG(bot_token, agent, db=db)

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, bot.echo)
    )
    application.add_handler(MessageHandler(filters.PHOTO, bot.echo))

    application.add_error_handler(
        lambda update, context: logging.error("Exception:", exc_info=context.error)
    )

    logging.info("Bot started.")
    application.run_polling()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eden Telegram Bot")
    parser.add_argument("--env", help="Path to the .env file to load", default=".env")
    parser.add_argument("--agent", help="Agent key to use", default="eve")
    parser.add_argument("--db", help="Database to use", default="STAGE")
    args = parser.parse_args()
    start(args.env, args.agent, args.db)
