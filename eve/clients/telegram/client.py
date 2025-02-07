from contextlib import asynccontextmanager
import os
import argparse
import re
from ably import AblyRealtime
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI
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
import asyncio

from ...clients import common
from ...agent import Agent
from ...llm import UpdateType
from ...user import User
from ...eden_utils import prepare_result
from ...deploy import ClientType, Deployment, DeploymentConfig


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


@common.client_context("telegram")
class EdenTG:
    def __init__(self, token: str, agent: Agent, local: bool = False):
        self.token = token
        self.agent = agent

        # Parse allowlist into tuples of (group_id, topic_id)
        self.telegram_topic_allowlist = []
        self.telegram_group_allowlist = []

        self.deployment_config = self._get_deployment_config(agent)

        if hasattr(self.deployment_config, "telegram") and hasattr(
            self.deployment_config.telegram, "topic_allowlist"
        ):
            for entry in self.deployment_config.telegram.topic_allowlist:
                try:
                    if "/" in entry:  # Topic format: "group_id/topic_id"
                        group_id, topic_id = entry.split("/")
                        internal_group_id = -int(f"100{group_id}")

                        # Special case: if topic_id is "1", this is the main channel
                        if topic_id == "1":
                            self.telegram_group_allowlist.append(internal_group_id)
                        else:
                            self.telegram_topic_allowlist.append(
                                (internal_group_id, int(topic_id))
                            )
                    else:  # Group format: "group_id"
                        self.telegram_group_allowlist.append(int(entry))
                except ValueError:
                    raise ValueError(f"Invalid format in telegram allowlist: {entry}")

        self.tools = agent.get_tools()
        self.known_users = {}
        self.known_threads = {}
        if local:
            self.api_url = "http://localhost:8000"
        else:
            self.api_url = os.getenv("EDEN_API_URL")
        self.channel_name = common.get_ably_channel_name(
            agent.name, ClientType.TELEGRAM
        )

        # Don't initialize Ably here - we'll do it in setup_ably
        self.ably_client = None
        self.channel = None

        self.typing_tasks = {}

    def _get_deployment_config(self, agent: Agent) -> DeploymentConfig:
        deployment = Deployment.load(agent=agent.id, platform="discord")
        if not deployment:
            raise Exception("No deployment config found")
        return deployment.config

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

    async def _typing_loop(
        self, chat_id: int, thread_id: int, application: Application
    ):
        """Keep sending typing action until stopped"""
        try:
            while True:
                await application.bot.send_chat_action(
                    chat_id=chat_id,
                    action=ChatAction.TYPING,
                    message_thread_id=thread_id,
                )
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

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
            telegram_message_id = update_config.get("telegram_message_id")
            telegram_thread_id = update_config.get("telegram_thread_id")

            if not telegram_chat_id:
                print("No telegram_chat_id in update_config:", data)
                return

            print(f"Processing update type: {update_type} for chat: {telegram_chat_id}")

            if update_type == UpdateType.START_PROMPT:
                # Start continuous typing
                chat_key = f"{telegram_chat_id}_{telegram_thread_id}"
                if chat_key not in self.typing_tasks:
                    self.typing_tasks[chat_key] = asyncio.create_task(
                        self._typing_loop(
                            int(telegram_chat_id),
                            int(telegram_thread_id) if telegram_thread_id else None,
                            application,
                        )
                    )

            elif update_type == UpdateType.ERROR:
                error_msg = data.get("error", "Unknown error occurred")
                await application.bot.send_message(
                    chat_id=telegram_chat_id, text=f"Error: {error_msg}"
                )

            elif update_type == UpdateType.ASSISTANT_MESSAGE:
                content = data.get("content")
                if content and not self.agent.mute:
                    await application.bot.send_message(
                        chat_id=telegram_chat_id,
                        text=content,
                        reply_to_message_id=telegram_message_id,
                        message_thread_id=telegram_thread_id,
                    )

            elif update_type == UpdateType.TOOL_COMPLETE:
                print(f"Tool complete: {data}")
                result = data.get("result", {})
                result["result"] = prepare_result(result["result"])
                outputs = result["result"][0]["output"]
                urls = [output["url"] for output in outputs[:4]]  # Get up to 4 URLs

                # Send each URL as appropriate media type
                for url in urls:
                    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
                    if any(url.lower().endswith(ext) for ext in video_extensions):
                        await application.bot.send_video(
                            chat_id=telegram_chat_id,
                            video=url,
                            reply_to_message_id=telegram_message_id,
                            message_thread_id=telegram_thread_id,
                        )
                    else:
                        await application.bot.send_photo(
                            chat_id=telegram_chat_id,
                            photo=url,
                            reply_to_message_id=telegram_message_id,
                            message_thread_id=telegram_thread_id,
                        )

            elif update_type == UpdateType.END_PROMPT:
                # Stop typing
                chat_key = f"{telegram_chat_id}_{telegram_thread_id}"
                if chat_key in self.typing_tasks:
                    self.typing_tasks[chat_key].cancel()
                    del self.typing_tasks[chat_key]

        # Subscribe using the async callback
        await self.channel.subscribe(async_callback)
        print(f"Subscribed to Ably channel: {self.channel_name}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handler for the /start command.
        """
        await update.message.reply_text(f"Hello! I am {self.agent.name}.")

    async def echo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle incoming messages and process bot mentions or direct messages.
        """

        message = update.message
        if not message:
            return

        chat_id = message.chat_id
        message_thread_id = message.message_thread_id

        # Always allow DMs (private chats)
        if message.chat.type == "private":
            pass
        # For messages in topics
        elif message_thread_id:
            # Only check allowlist if it exists
            if (
                self.telegram_topic_allowlist
                and (chat_id, message_thread_id) not in self.telegram_topic_allowlist
            ):
                return  # Silently ignore messages from non-allowlisted topics
        # For messages in regular groups or main channel
        else:
            # Only check allowlist if it exists
            if (
                self.telegram_group_allowlist
                and chat_id not in self.telegram_group_allowlist
            ):
                return  # Silently ignore messages from non-allowlisted groups

        (
            chat_id,
            chat_type,
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

        force_reply = False
        if is_bot_mentioned or is_replied_to_bot or is_direct_message:
            force_reply = True

        if is_direct_message:
            # print author
            force_reply = False  # No DMs
            return

        # Update thread key to include both group and topic IDs if present
        thread_key = (
            f"telegram-{chat_id}-topic-{message_thread_id}"
            if message_thread_id
            else f"telegram-{chat_id}"
        )

        if thread_key not in self.known_threads:
            self.known_threads[thread_key] = self.agent.request_thread(key=thread_key)
        thread = self.known_threads[thread_key]

        # Lookup user
        if user_id not in self.known_users:
            self.known_users[user_id] = User.from_telegram(user_id, username)
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
        message_text = message.text or ""
        attachments = []
        cleaned_text = message_text
        if message.photo:
            photo_url = (await message.photo[-1].get_file()).file_path
            attachments.append(photo_url)
        else:
            cleaned_text = replace_bot_mentions(
                message_text, me_bot.username, self.agent.name
            )

        # Make API request
        request_data = {
            "user_id": str(user.id),
            "agent_id": str(self.agent.id),
            "thread_id": str(thread.id),
            "force_reply": force_reply,
            "user_message": {
                "content": cleaned_text,
                "name": username,
                "attachments": attachments,
            },
            "update_config": {
                "sub_channel_name": self.channel_name,
                "telegram_chat_id": str(chat_id),
                "telegram_message_id": str(message.message_id),
                "telegram_thread_id": str(message_thread_id)
                if message_thread_id
                else None,
            },
        }

        async with aiohttp.ClientSession() as session:
            print(f"Sending request to {self.api_url}/chat")
            async with session.post(
                f"{self.api_url}/chat",
                json=request_data,
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "X-Client-Platform": "telegram",
                    "X-Client-Agent": self.agent.username,
                },
            ) as response:
                print(f"Response from {self.api_url}/chat: {response.status}")
                # json
                print(await response.json())
                if response.status != 200:
                    await send_response(
                        message_type,
                        chat_id,
                        ["Sorry, something went wrong processing your request."],
                        context,
                    )
                    return

    async def send_typing(self, chat_id: int, message_thread_id: int = None):
        """Send typing indicator to Telegram."""
        try:
            await self.bot.send_chat_action(
                chat_id=chat_id, action="typing", message_thread_id=message_thread_id
            )
        except Exception as e:
            print(f"Error sending typing indicator: {e}")


def init(env: str, local: bool = False) -> None:
    print("Starting Telegram client...")
    load_dotenv(env)

    agent_name = os.getenv("AGENT_ID")
    agent = Agent.from_mongo(agent_name)

    bot_token = os.getenv("CLIENT_TELEGRAM_TOKEN")
    if not bot_token:
        raise ValueError("CLIENT_TELEGRAM_TOKEN not found in environment variables")

    application = ApplicationBuilder().token(bot_token).build()
    bot = EdenTG(bot_token, agent, local=local)

    # Setup handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.echo))
    application.add_handler(MessageHandler(filters.PHOTO, bot.echo))

    # Create a post init callback to setup Ably
    async def post_init(application: Application) -> None:
        await bot.initialize(application)

    application.post_init = post_init

    # Run the bot
    return application


def start(env: str, local: bool = False) -> None:
    app = init(env, local)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Just yield immediately - we'll start the bot separately
    yield


def create_telegram_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    # Start the bot in a background thread. Without setting an event loop,
    # run_polling() will fail in a newly spawned thread.
    application = init(env=".env", local=False)

    def run_bot():
        import asyncio

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Disable signal handling in this background thread to avoid errors.
        loop.run_until_complete(
            application.run_polling(
                allowed_updates=Update.ALL_TYPES, handle_signals=False
            )
        )

    import threading

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eden Telegram Bot")
    parser.add_argument("--env", help="Path to the .env file to load", default=".env")
    parser.add_argument("--local", help="Run locally", action="store_true")
    args = parser.parse_args()
    start(args.env, args.local)
