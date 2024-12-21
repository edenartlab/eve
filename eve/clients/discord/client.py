import argparse
import os
import re
import discord
from discord.ext import commands
from dotenv import load_dotenv
from ably import AblyRealtime
import aiohttp

from eve.clients import common
from eve.agent import Agent
from eve.llm import UpdateType
from eve.user import User
from eve.eden_utils import prepare_result
from eve.models import ClientType


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


class Eden2Cog(commands.Cog):
    def __init__(
        self,
        bot: commands.bot,
        agent: Agent,
        db: str = "STAGE",
    ) -> None:
        self.bot = bot
        self.agent = agent
        self.db = db
        self.tools = agent.get_tools(db=self.db)
        self.known_users = {}
        self.known_threads = {}
        self.channel_name = common.get_ably_channel_name(agent.name, ClientType.DISCORD)

        # Setup will be done in on_ready
        self.ably_client = None
        self.channel = None
        # Track message IDs
        self.pending_messages = {}

    async def setup_ably(self):
        """Initialize Ably client and subscribe to updates"""
        self.ably_client = AblyRealtime(os.getenv("ABLY_SUBSCRIBER_KEY"))
        self.channel = self.ably_client.channels.get(self.channel_name)

        async def async_callback(message):
            print(f"Received update in Discord client: {message.data}")

            data = message.data
            if not isinstance(data, dict) or "type" not in data:
                print("Invalid message format:", data)
                return

            update_type = data["type"]
            update_config = data.get("update_config", {})
            discord_channel_id = update_config.get("discord_channel_id")
            message_id = update_config.get("message_id")

            if not discord_channel_id:
                print("No discord_channel_id in update_config:", data)
                return

            channel = self.bot.get_channel(int(discord_channel_id))
            if not channel:
                print(f"Could not find channel with id {discord_channel_id}")
                return

            print(f"Processing update type: {update_type} for channel: {channel.name}")

            try:
                # Get the original message if message_id is provided
                reference = None
                if message_id:
                    try:
                        original_message = await channel.fetch_message(int(message_id))
                        reference = original_message.to_reference()
                    except:
                        print(f"Could not fetch original message {message_id}")

                if update_type == UpdateType.START_PROMPT:
                    pass

                elif update_type == UpdateType.ERROR:
                    error_msg = data.get("error", "Unknown error occurred")
                    await channel.send(f"Error: {error_msg}", reference=reference)

                elif update_type == UpdateType.ASSISTANT_MESSAGE:
                    content = data.get("content")
                    if content:
                        await channel.send(content, reference=reference)

                elif update_type == UpdateType.TOOL_COMPLETE:
                    result = data.get("result", {})
                    result["result"] = prepare_result(result["result"], db=self.db)
                    url = result["result"][0]["output"][0]["url"]
                    await channel.send(url, reference=reference)
                else:
                    print(f"Unknown update type: {update_type}")

            except Exception as e:
                print(f"Error processing update: {e}")

        await self.channel.subscribe(async_callback)
        print(f"Subscribed to Ably channel: {self.channel_name}")

    @commands.Cog.listener()
    async def on_ready(self):
        """Called when the bot is ready and connected"""
        await self.setup_ably()
        print("Bot is ready and Ably is configured")

    def __del__(self):
        """Cleanup when the cog is destroyed"""
        if hasattr(self, "ably_client") and self.ably_client:
            self.ably_client.close()

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        if message.author.id == self.bot.user.id:
            return

        dm = message.channel.type == discord.ChannelType.private
        if dm:
            thread_key = f"discord-dm-{message.author.name}-{message.author.id}"
            if message.author.id not in common.DISCORD_DM_WHITELIST:
                return
        else:
            thread_key = f"discord-{message.guild.id}-{message.channel.id}"

        # Lookup thread
        if thread_key not in self.known_threads:
            self.known_threads[thread_key] = self.agent.request_thread(
                key=thread_key,
                db=self.db,
            )
        thread = self.known_threads[thread_key]

        # Lookup user
        if message.author.id not in self.known_users:
            self.known_users[message.author.id] = User.from_discord(
                message.author.id, message.author.name, db=self.db
            )
        user = self.known_users[message.author.id]

        if common.user_over_rate_limits(user):
            await reply(
                message,
                "I'm sorry, you've hit your rate limit. Please try again a bit later!",
            )
            return

        content = replace_mentions_with_usernames(message.content, message.mentions)

        if self.bot.user.display_name.lower() in content.lower():
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

        ctx = await self.bot.get_context(message)
        await ctx.channel.trigger_typing()

        # Make API request
        api_url = os.getenv("EDEN_API_URL")

        async with aiohttp.ClientSession() as session:
            request_data = {
                "user_id": str(user.id),
                "agent_id": str(self.agent.id),
                "thread_id": str(thread.id),
                "user_message": {
                    "content": content,
                    "name": message.author.name,
                    "attachments": [
                        attachment.url for attachment in message.attachments
                    ],
                },
                "update_config": {
                    "sub_channel_name": self.channel_name,
                    "discord_channel_id": str(message.channel.id),
                    "message_id": str(message.id),  # Add message ID to update_config
                },
            }

            print(f"Sending request: {request_data}")

            async with session.post(
                f"{api_url}/chat",
                json=request_data,
            ) as response:
                if response.status != 200:
                    await reply(
                        message, "Sorry, something went wrong processing your request."
                    )
                    return

    @commands.Cog.listener()
    async def on_member_join(self, member):
        print(f"{member} has joined the guild id: {member.guild.id}")


async def reply(message, content):
    content_chunks = [content[i : i + 1980] for i in range(0, len(content), 1980)]
    for c, chunk in enumerate(content_chunks):
        await message.reply(chunk) if c == 0 else await message.channel.send(chunk)


async def send(message, content):
    content_chunks = [content[i : i + 1980] for i in range(0, len(content), 1980)]
    for chunk in content_chunks:
        await message.channel.send(chunk)


class DiscordBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        self.set_intents(intents)
        commands.Bot.__init__(
            self,
            command_prefix="!",
            intents=intents,
        )

    def set_intents(self, intents: discord.Intents) -> None:
        intents.message_content = True
        intents.messages = True
        intents.presences = True
        intents.members = True

    async def on_ready(self) -> None:
        # logger.info("Running bot...")
        pass

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        await self.process_commands(message)


def start(
    env: str,
    db: str = "STAGE",
) -> None:
    # logger.info("Launching Discord bot...")
    load_dotenv(env)

    agent_name = os.getenv("EDEN_AGENT_USERNAME")
    agent = Agent.load(agent_name, db=db)

    # logger.info(f"Using agent: {agent.name}")
    bot_token = os.getenv("CLIENT_DISCORD_TOKEN")
    bot = DiscordBot()
    bot.add_cog(Eden2Cog(bot, agent, db=db))
    bot.run(bot_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiscordBot")
    parser.add_argument("--agent", help="Agent username")
    parser.add_argument("--db", help="Database to use", default="STAGE")
    parser.add_argument(
        "--env", help="Path to a different .env file not in agent directory"
    )
    args = parser.parse_args()
    start(args.env, args.agent, args.db)
