import argparse
import asyncio
import logging
import os
import re
import sentry_sdk
import aiohttp
import discord
from discord.ext import commands
from dotenv import load_dotenv
from fastapi import FastAPI
from contextlib import asynccontextmanager

from ... import load_env
from ...clients import common
from ...agent import Agent
from ...llm import UpdateType
from ...user import User
from ...eden_utils import prepare_result
from ...deploy import ClientType

logger = logging.getLogger(__name__)


def replace_mentions_with_usernames(
    message_content: str,
    mentions,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """
    Replaces all mentions with their usernames.
    :param message_content: The message to replace mentions in.
    :return: The message with all mentions replaced with their usernames.
    """
    for mention in mentions:
        message_content = re.sub(
            f"<@[!&]?{mention.id}>",
            f"{prefix}{mention.display_name}{suffix}",
            message_content,
        )
    return message_content.strip()


@common.client_context("discord")
class Eden2Cog(commands.Cog):
    def __init__(
        self,
        bot: commands.bot,
        agent: Agent,
        local: bool = False,
    ) -> None:
        self.bot = bot
        self.agent = agent
        self.discord_channel_allowlist = (
            [int(channel) for channel in agent.discord_channel_allowlist]
            if agent.discord_channel_allowlist
            else None
        )
        self.tools = agent.get_tools()
        self.known_users = {}
        self.known_threads = {}
        if local:
            self.api_url = "http://localhost:8000"
        else:
            self.api_url = os.getenv("EDEN_API_URL")
        self.channel_name = common.get_ably_channel_name(
            agent.username, ClientType.DISCORD
        )

        # Setup will be done in on_ready
        self.ably_client = None
        self.channel = None

        self.typing_tasks = {}

    async def setup_ably(self):
        """Configure Ably realtime client"""
        try:
            from ably import AblyRealtime

            self.ably_client = AblyRealtime(key=os.getenv("ABLY_SUBSCRIBER_KEY"))
            self.channel = self.ably_client.channels.get(self.channel_name)
            await self.subscribe_to_updates()
            logger.info(f"Ably configured for channel: {self.channel_name}")
        except Exception as e:
            logger.error("Failed to setup Ably", exc_info=True)
            sentry_sdk.capture_exception(e)
            raise

    async def subscribe_to_updates(self):
        """Subscribe to Ably channel updates"""

        async def async_callback(message):
            try:
                data = message.data
                if not isinstance(data, dict):
                    logger.warning(f"Received invalid message format: {data}")
                    return

                update_type = data["type"]
                update_config = data.get("update_config", {})
                discord_channel_id = update_config.get("discord_channel_id")
                message_id = update_config.get("message_id")

                if not discord_channel_id:
                    raise Exception("No discord_channel_id in update_config")

                # Try to get channel first (for regular channels)
                channel = self.bot.get_channel(int(discord_channel_id))

                # If channel not found, try to get user (for DMs)
                if not channel:
                    user = self.bot.get_user(int(discord_channel_id))
                    if user:
                        channel = await user.create_dm()

                if not channel:
                    raise Exception(
                        f"Could not find channel or user with id {discord_channel_id}"
                    )

                logger.info(
                    f"Processing update type: {update_type} for channel: {channel.name if hasattr(channel, 'name') else 'DM'}"
                )

                try:
                    # Get the original message if message_id is provided
                    reference = None
                    if message_id and not isinstance(channel, discord.DMChannel):
                        try:
                            original_message = await channel.fetch_message(
                                int(message_id)
                            )
                            reference = original_message.to_reference()
                        except Exception as e:
                            logger.error(
                                f"Could not fetch original message {message_id}",
                                exc_info=True,
                            )
                            sentry_sdk.capture_exception(e)

                    if update_type == UpdateType.START_PROMPT:
                        await self.start_typing(channel)

                    elif update_type == UpdateType.ERROR:
                        error_msg = data.get("error", "Unknown error occurred")
                        await self.send_message(
                            channel, f"Error: {error_msg}", reference=reference
                        )

                    elif update_type == UpdateType.ASSISTANT_MESSAGE:
                        content = data.get("content")
                        if content:
                            await self.send_message(
                                channel, content, reference=reference
                            )

                    elif update_type == UpdateType.TOOL_COMPLETE:
                        result = data.get("result", {})
                        result["result"] = prepare_result(result["result"])
                        outputs = result["result"][0]["output"]
                        urls = [
                            output["url"] for output in outputs[:4]
                        ]  # Get up to 4 URLs

                        # Get creation ID from the first output
                        creation_id = str(outputs[0].get("creation"))

                        if creation_id:
                            eden_url = common.get_eden_creation_url(creation_id)
                            view = discord.ui.View()
                            view.add_item(
                                discord.ui.Button(
                                    label="View on Eden",
                                    url=eden_url,
                                    style=discord.ButtonStyle.link,
                                )
                            )
                            await self.send_message(
                                channel, "\n".join(urls), reference=reference, view=view
                            )
                        else:
                            await self.send_message(
                                channel, "\n".join(urls), reference=reference
                            )

                    elif update_type == UpdateType.END_PROMPT:
                        await self.stop_typing(channel)

                except Exception as e:
                    logger.error("Error processing update", exc_info=True)
                    sentry_sdk.capture_exception(e)

            except Exception as e:
                logger.error("Error in async_callback", exc_info=True)
                sentry_sdk.capture_exception(e)

        await self.channel.subscribe(async_callback)
        logger.info(f"Subscribed to Ably channel: {self.channel_name}")

    @commands.Cog.listener()
    async def on_ready(self):
        """Called when the bot is ready and connected"""
        try:
            await self.setup_ably()
            logger.info(f"Bot {self.bot.user.name} is ready and Ably is configured")
        except Exception as e:
            logger.error("Failed to initialize bot", exc_info=True)
            sentry_sdk.capture_exception(e)

    def __del__(self):
        """Cleanup when the cog is destroyed"""
        try:
            if hasattr(self, "ably_client") and self.ably_client:
                self.ably_client.close()
        except Exception as e:
            logger.error("Error during cleanup", exc_info=True)
            sentry_sdk.capture_exception(e)

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        try:
            if (
                self.discord_channel_allowlist
                and message.channel.id not in self.discord_channel_allowlist
            ):
                return

            if message.author.id == self.bot.user.id:
                return

            # Add check for bot messages
            dont_reply = message.author.bot

            dm = message.channel.type == discord.ChannelType.private
            if dm:
                thread_key = f"discord-dm-{message.author.name}-{message.author.id}"
                if message.author.id not in common.DISCORD_DM_WHITELIST:
                    return
            else:
                thread_key = f"discord-{message.guild.id}-{message.channel.id}"

            # Stop typing if dont_reply is true
            # if dont_reply:
            #     await self.stop_typing(message.channel)

            # Lookup thread
            if thread_key not in self.known_threads:
                self.known_threads[thread_key] = self.agent.request_thread(
                    key=thread_key
                )
            thread = self.known_threads[thread_key]

            # Lookup user
            if message.author.id not in self.known_users:
                self.known_users[message.author.id] = User.from_discord(
                    message.author.id, message.author.name
                )
            user = self.known_users[message.author.id]

            if common.user_over_rate_limits(user):
                await message.reply(
                    "I'm sorry, you've hit your rate limit. Please try again a bit later!",
                )
                return

            # check if bot is mentioned in the message or replied to
            force_reply = False
            if self.bot.user in message.mentions:
                force_reply = True

            content = replace_mentions_with_usernames(message.content, message.mentions)
            content = re.sub(
                rf"\b{re.escape(self.bot.user.display_name)}\b",
                self.agent.name,
                content,
                flags=re.IGNORECASE,
            )

            if message.reference:
                source_message = await message.channel.fetch_message(
                    message.reference.message_id
                )
                content = f"(Replying to message: {source_message.content[:100]} ...)\n\n{content}"

            async with aiohttp.ClientSession() as session:
                request_data = {
                    "user_id": str(user.id),
                    "agent_id": str(self.agent.id),
                    "thread_id": str(thread.id),
                    "force_reply": force_reply,
                    "dont_reply": dont_reply,
                    "user_message": {
                        "content": content,
                        "name": message.author.name,
                        "attachments": [
                            attachment.proxy_url for attachment in message.attachments
                        ],
                    },
                    "update_config": {
                        "sub_channel_name": self.channel_name,
                        "discord_channel_id": str(
                            message.author.id if dm else message.channel.id
                        ),
                        "message_id": str(message.id),
                    },
                }

                logger.info(f"Sending request for user {message.author.name}")
                async with session.post(
                    f"{self.api_url}/chat",
                    json=request_data,
                    headers={
                        "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                        "X-Client-Platform": "discord",
                        "X-Client-Agent": self.agent.username,
                    },
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        logger.error(f"API request failed: {error_msg}")
                        await message.reply(
                            "Sorry, something went wrong processing your request.",
                        )
                        return

        except Exception as e:
            logger.error("Error processing message", exc_info=True)
            sentry_sdk.capture_exception(e)
            await message.reply(
                "Sorry, an error occurred while processing your message."
            )

    @commands.Cog.listener()
    async def on_member_join(self, member):
        logger.info(f"Member {member} has joined guild {member.guild.id}")

    async def send_message(
        self, channel, content, reference=None, limit=2000, view=None
    ):
        """Send a message to a channel with proper chunking and error handling"""
        try:
            for i in range(0, len(content), limit):
                chunk = content[i : i + limit]
                await channel.send(chunk, reference=reference, view=view)
        except Exception as e:
            logger.error("Failed to send message", exc_info=True)
            sentry_sdk.capture_exception(e)
            raise

    async def start_typing(self, channel):
        """Start or resume indefinite typing in a given channel"""
        try:
            existing_task = self.typing_tasks.get(channel.id)
            if existing_task and not existing_task.done():
                return

            async def keep_typing(ch):
                try:
                    while True:
                        await ch.trigger_typing()
                        await asyncio.sleep(5)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error("Error in typing loop", exc_info=True)
                    sentry_sdk.capture_exception(e)

            self.typing_tasks[channel.id] = asyncio.create_task(keep_typing(channel))
            channel_name = getattr(channel, "name", "DM")
            logger.info(f"Started typing in channel: {channel_name} ({channel.id})")
        except Exception as e:
            logger.error("Failed to start typing", exc_info=True)
            sentry_sdk.capture_exception(e)

    async def stop_typing(self, channel):
        """Cancel the indefinite typing task for a given channel"""
        try:
            typing_task = self.typing_tasks.pop(channel.id, None)
            if typing_task and not typing_task.done():
                typing_task.cancel()
        except Exception as e:
            logger.error("Failed to stop typing", exc_info=True)
            sentry_sdk.capture_exception(e)


class DiscordBot(commands.Bot):
    def __init__(self) -> None:
        try:
            intents = discord.Intents.default()
            self.set_intents(intents)
            commands.Bot.__init__(
                self,
                command_prefix="!",
                intents=intents,
            )
        except Exception as e:
            logger.error("Failed to initialize Discord bot", exc_info=True)
            sentry_sdk.capture_exception(e)
            raise

    def set_intents(self, intents: discord.Intents) -> None:
        intents.message_content = True
        intents.messages = True
        intents.presences = True
        intents.members = True

    async def on_ready(self) -> None:
        logger.info("Discord bot is running")

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        await self.process_commands(message)


def init(
    env: str,
    local: bool = False,
) -> None:
    try:
        load_dotenv(env)

        agent_name = os.getenv("EDEN_AGENT_USERNAME")
        agent = Agent.load(agent_name)

        logger.info(f"Launching Discord bot {agent.username}...")

        bot_token = os.getenv("CLIENT_DISCORD_TOKEN")
        bot = DiscordBot()
        bot.add_cog(Eden2Cog(bot, agent, local=local))
        return bot, bot_token
    except Exception as e:
        logger.error("Failed to start Discord bot", exc_info=True)
        sentry_sdk.capture_exception(e)
        raise


def start(env: str, local: bool = False) -> None:
    bot, bot_token = init(env, local)
    asyncio.create_task(bot.run(bot_token))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    bot, bot_token = init(env=".env", local=False)

    # Create a background task for the bot
    bot_task = None

    async def run_bot():
        try:
            await bot.start(bot_token)
        except Exception as e:
            logger.error("Bot crashed", exc_info=True)
            sentry_sdk.capture_exception(e)

    # Start bot in background task
    bot_task = asyncio.create_task(run_bot())

    yield

    # Cleanup
    if bot_task and not bot_task.done():
        bot_task.cancel()
    if not bot.is_closed():
        await bot.close()


def create_discord_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiscordBot")
    parser.add_argument("--agent", help="Agent username")
    parser.add_argument("--db", help="Database to use", default="STAGE")
    parser.add_argument(
        "--env", help="Path to a different .env file not in agent directory"
    )
    parser.add_argument("--local", help="Run locally", action="store_true")
    args = parser.parse_args()

    load_env(args.db)
    bot, bot_token = init(args.env, args.local)
    bot.run(bot_token)
