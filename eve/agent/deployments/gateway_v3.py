import asyncio
import logging
import os
import socket
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Workaround: Import stdlib/packages before local files can shadow them
# This prevents discord.py and email.py in this directory from shadowing imports
_original_path = sys.path.copy()
_deployment_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _deployment_dir]

import discord

sys.path = _original_path
del _original_path, _deployment_dir

import aiohttp
import modal
from fastapi import FastAPI

from eve.agent.agent import Agent
from eve.agent.deployments.discord_gateway import (
    convert_discord_mentions_to_usernames,
    get_or_create_discord_message,
    upload_discord_media_to_s3,
)
from eve.agent.deployments.typing_manager import (
    DiscordTypingManager,
    TelegramTypingManager,
)
from eve.agent.session.context import add_user_to_session
from eve.agent.session.models import (
    Channel,
    ChatMessage,
    ClientType,
    Deployment,
    Session,
    SessionUpdateConfig,
)
from eve.user import DiscordGuildAccess, User, increment_message_count

# Override the imported db with uppercase version for Ably channel consistency
db = os.getenv("DB", "STAGE").upper()

logger = logging.getLogger(__name__)
root_dir = Path(__file__).parent.parent.parent.parent


def construct_agent_chat_url(agent_username: str) -> str:
    """
    Construct the Eden agent chat URL based on the environment and agent username.

    Args:
        agent_username: The username of the agent (not the ID)

    Returns:
        Properly formatted Eden agent chat URL
    """
    root_url = "app.eden.art" if db == "PROD" else "staging.app.eden.art"
    return f"https://{root_url}/chat/{agent_username}"


async def fetch_discord_channel_name(channel_id: str, bot_token: str) -> Optional[str]:
    """
    Fetch the channel name from Discord API.

    Args:
        channel_id: Discord channel ID
        bot_token: Bot token for authentication

    Returns:
        Channel name or None if not found
    """
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bot {bot_token}"}
            url = f"https://discord.com/api/v10/channels/{channel_id}"
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    channel_data = await response.json()
                    return channel_data.get("name")
    except Exception:
        pass
    return None


async def fetch_discord_guild_name(guild_id: str, bot_token: str) -> Optional[str]:
    """
    Fetch the guild (server) name from Discord API.

    Args:
        guild_id: Discord guild ID
        bot_token: Bot token for authentication

    Returns:
        Guild name or None if not found
    """
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bot {bot_token}"}
            url = f"https://discord.com/api/v10/guilds/{guild_id}"
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    guild_data = await response.json()
                    return guild_data.get("name")
    except Exception:
        pass
    return None


# ============================================================================
# DISCORD PRIVATE WORKSPACE PATTERN - Processing Functions
# ============================================================================


async def find_all_following_deployments(channel_id: str) -> List[Deployment]:
    """
    Find all Discord deployments that follow this channel (read or write access).

    Supports both legacy (channel_allowlist) and v3 (channel_configs) formats.

    Args:
        channel_id: Discord channel ID

    Returns:
        List of Deployment objects
    """
    all_deployments = list(
        Deployment.find(
            {
                "platform": ClientType.DISCORD_V3.value,
                "valid": {"$ne": False},
            }
        )
    )

    following = []
    for deployment in all_deployments:
        if not deployment.config or not deployment.config.discord:
            continue

        # V3 webhook-based: check channel_configs
        if deployment.config.discord.channel_configs:
            config_channel_ids = [
                ch.channel_id for ch in deployment.config.discord.channel_configs
            ]
            if channel_id in config_channel_ids:
                following.append(deployment)
                continue

        # Legacy token-based: check channel_allowlist and read_access_channels
        channel_allowlist = deployment.config.discord.channel_allowlist or []
        allowed_ids = [str(item.id) for item in channel_allowlist if item]

        read_access = deployment.config.discord.read_access_channels or []
        read_ids = [str(item.id) for item in read_access if item]

        all_following_ids = set(allowed_ids + read_ids)

        if channel_id in all_following_ids:
            following.append(deployment)

    return following


def deployment_can_write_to_channel(deployment: Deployment, channel_id: str) -> bool:
    """
    Check if a deployment has write access to a channel.

    Supports both legacy (channel_allowlist) and v3 (channel_configs) formats.

    Args:
        deployment: The deployment to check
        channel_id: Discord channel ID

    Returns:
        True if the channel has write access
    """
    if not deployment.config or not deployment.config.discord:
        return False

    # V3 webhook-based: check channel_configs for read_write access
    if deployment.config.discord.channel_configs:
        for ch in deployment.config.discord.channel_configs:
            if ch.channel_id == channel_id:
                return ch.access == "read_write"
        return False

    # Legacy token-based: check channel_allowlist
    channel_allowlist = deployment.config.discord.channel_allowlist or []
    allowed_ids = [str(item.id) for item in channel_allowlist if item]

    return channel_id in allowed_ids


# V3 gateway doesn't support backfilling (no bot token, uses webhooks only)


async def process_discord_message_for_agent(
    message: discord.Message,
    deployment: Deployment,
    should_prompt: bool = False,
    trace_id: str = None,
    parent_channel_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a Discord message for a single agent.

    This creates/updates the agent's private session and optionally prompts the LLM.

    Args:
        message: Discord.py Message object
        deployment: The agent's deployment
        should_prompt: Whether to prompt the agent to respond
        trace_id: Trace ID for logging
        parent_channel_id: Parent channel ID if this is a thread

    Returns:
        Status dict with session_id, message_id, etc.
    """
    channel_id = str(message.channel.id)
    guild_id = str(message.guild.id) if message.guild else None

    trace_id = trace_id or f"discord-{deployment.id}-{message.id}"

    # Add process identifier to help detect multiple gateway instances
    process_id = os.getpid()
    hostname = socket.gethostname()
    logger.info(
        f"[{trace_id}] process_discord_message_for_agent called from "
        f"host={hostname}, pid={process_id}, deployment={deployment.id}, should_prompt={should_prompt}"
    )

    try:
        # Load agent
        agent = Agent.from_mongo(deployment.agent)
        if not agent:
            logger.error(f"[{trace_id}] Agent not found for deployment {deployment.id}")
            return {"status": "error", "message": "Agent not found"}

        # Extract message data
        message_id = str(message.id)
        author_id = str(message.author.id)
        author_username = message.author.name
        content = message.content or ""

        # Extract attachments
        media_urls = []
        if message.attachments:
            for attachment in message.attachments:
                media_urls.append(attachment.url)

        timestamp = message.created_at

        # Convert Discord mentions to readable @username format
        if message.mentions:
            mentions_data = [
                {"id": str(u.id), "username": u.name} for u in message.mentions
            ]
            content = convert_discord_mentions_to_usernames(
                content=content,
                mentions_data=mentions_data,
                bot_discord_id=deployment.secrets.discord.application_id,
                bot_name=agent.name,
            )

        # Convert role mentions to readable @role_name format
        if message.role_mentions:
            for role in message.role_mentions:
                role_mention_pattern = f"<@&{role.id}>"
                content = content.replace(role_mention_pattern, f"@{role.name}")

        # Skip empty messages
        if not content and not media_urls:
            logger.info(f"[{trace_id}] Skipping empty message")
            return {"status": "skipped", "message": "Empty message"}

        # Upload media to S3
        if media_urls:
            media_urls = await upload_discord_media_to_s3(media_urls)

        # Create/get user
        sender = User.from_discord(
            author_id,
            author_username,
            discord_avatar=message.author.avatar.key if message.author.avatar else None,
        )

        # Build session key
        session_key = f"discord-{agent.id}-{guild_id or 'dm'}-{channel_id}"

        # Try to load existing session
        session = Session.find_one({"session_key": session_key})

        if session:
            logger.info(f"[{trace_id}] Found existing session {session.id}")

            # Reactivate if deleted
            if session.deleted:
                session.deleted = False
                session.status = "active"
                session.save()
                logger.info(f"[{trace_id}] Reactivated deleted session {session.id}")
        else:
            # Create new session
            logger.info(f"[{trace_id}] Creating new session for {session_key}")

            # Fetch channel and guild names
            channel_name = None
            guild_name = None

            if guild_id:
                channel_name = message.channel.name
                guild_name = message.guild.name if message.guild else None

                # Build title
                if isinstance(message.channel, discord.Thread):
                    parent_name = (
                        message.channel.parent.name
                        if message.channel.parent
                        else "Unknown"
                    )
                    title = f"{guild_name}: {parent_name}: {channel_name}"
                else:
                    title = f"{guild_name}: {channel_name}"
            else:
                # DM
                title = author_username

            # Determine owner: user for DM, agent.owner for guilds
            owner_id = sender.id if not guild_id else agent.owner

            # For threads, inherit from parent session
            users = []
            agents = [agent.id]
            settings = None
            parent_session_id = None

            if parent_channel_id and guild_id:
                parent_session_key = (
                    f"discord-{agent.id}-{guild_id}-{parent_channel_id}"
                )
                parent_session = Session.find_one({"session_key": parent_session_key})
                if parent_session:
                    users = parent_session.users or []
                    agents = parent_session.agents or [agent.id]
                    settings = parent_session.settings
                    parent_session_id = parent_session.id
                    logger.info(
                        f"[{trace_id}] Inherited from parent session {parent_session_id}"
                    )

            # Create session
            session = Session(
                session_key=session_key,
                owner=owner_id,
                users=users,
                agents=agents,
                settings=settings,
                session_type="passive",
                platform="discord",  # Use string for backward compatibility
                title=title,
                channel=Channel(type="discord", key=channel_id),
                parent_session=parent_session_id,
                discord_channel_id=channel_id,
                discord_guild_id=guild_id,
            )
            session.save()
            logger.info(f"[{trace_id}] Created new session {session.id}")
            logger.info(
                f"[{trace_id}] No backfill for V3 deployments (webhook-based, no bot token)"
            )

        # Check if ChatMessage already exists
        existing_chat = ChatMessage.find_one(
            {
                "channel.key": message_id,
                "channel.type": "discord",
            }
        )

        if existing_chat:
            # Add session to existing message
            if session.id not in existing_chat.session:
                ChatMessage.get_collection().update_one(
                    {"_id": existing_chat.id},
                    {"$addToSet": {"session": session.id}},
                )
                logger.info(
                    f"[{trace_id}] Added session to existing ChatMessage {existing_chat.id}"
                )
            chat_message = existing_chat
        else:
            # Build Discord message URL
            if guild_id:
                discord_url = (
                    f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
                )
            else:
                discord_url = (
                    f"https://discord.com/channels/@me/{channel_id}/{message_id}"
                )

            # Create new ChatMessage directly (add_chat_message signature changed)
            chat_message = ChatMessage(
                createdAt=timestamp,
                session=[session.id],
                channel=Channel(type="discord", key=message_id, url=discord_url),
                role="user",
                content=content,
                sender=sender.id,
                attachments=media_urls if media_urls else None,
            )
            chat_message.save()
            logger.info(f"[{trace_id}] Created new ChatMessage {chat_message.id}")

        # Track in DiscordMessage collection
        # Convert discord.py message to dict format expected by helper
        message_dict = {
            "id": message_id,
            "author": {
                "id": author_id,
                "username": author_username,
            },
            "content": message.content or "",
            "timestamp": timestamp.isoformat(),
            "attachments": [{"url": a.url} for a in message.attachments],
        }
        get_or_create_discord_message(
            message_dict, session_id=session.id, eve_message_id=chat_message.id
        )

        # Increment message count
        increment_message_count(sender.id)

        # Add user to session
        add_user_to_session(session, sender.id)

        # Prompt if needed
        if should_prompt:
            try:
                from eve.agent.session.orchestrator import orchestrate_deployment

                logger.info(f"[{trace_id}] Prompting agent for session {session.id}")

                async for update in orchestrate_deployment(
                    session=session,
                    agent=agent,
                    user_id=sender.id,
                    update_config=SessionUpdateConfig(
                        discord_channel_id=channel_id,
                        discord_message_id=message_id,
                        discord_guild_id=guild_id,
                    ),
                ):
                    # Stream updates (logged by orchestrator)
                    pass

                logger.info(f"[{trace_id}] Agent prompted successfully")
            except Exception as e:
                logger.error(f"[{trace_id}] Failed to prompt agent: {e}", exc_info=True)
                return {"status": "error", "message": f"Failed to prompt agent: {e}"}

        return {
            "status": "success",
            "session_id": str(session.id),
            "message_id": str(chat_message.id),
            "prompted": should_prompt,
        }

    except Exception as e:
        logger.error(f"[{trace_id}] Error processing message: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def parse_mentioned_deployments(
    message: discord.Message,
    following_deployments: List[Deployment],
) -> List[Deployment]:
    """
    Parse which deployments were mentioned in this message.

    Supports both:
    - Legacy: bot user @mentions
    - V3: role @mentions (webhook-based deployments)

    Args:
        message: Discord.py Message object
        following_deployments: List of deployments following this channel

    Returns:
        List of mentioned Deployment objects
    """
    mentioned = []

    # Check user mentions (bot @mentions) - legacy token-based
    mentioned_user_ids = {str(u.id) for u in message.mentions if u.bot}

    # Check role mentions - webhook-based v3
    mentioned_role_ids = {str(r.id) for r in message.role_mentions}

    for deployment in following_deployments:
        # V3 webhook-based: check role mentions
        if (
            deployment.config
            and deployment.config.discord
            and deployment.config.discord.role_id
        ):
            role_id = deployment.config.discord.role_id
            if role_id in mentioned_role_ids:
                mentioned.append(deployment)
                continue

        # Legacy token-based: check bot user mentions
        if deployment.secrets and deployment.secrets.discord:
            app_id = deployment.secrets.discord.application_id
            if app_id and app_id in mentioned_user_ids:
                mentioned.append(deployment)
                continue

    return mentioned


# ============================================================================
# PY-CORD BOT
# ============================================================================


app_name = (
    f"discord-gateway-v3-{db}-{os.getenv('GATEWAY_ID')}"
    if os.getenv("GATEWAY_ID")
    else f"discord-gateway-v3-{db}"
)

# Create Modal app
modal_app = modal.App(
    name=app_name,
    secrets=[
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-gateway-v3-{db}", environment_name="main"),
    ],
)

# Set up image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libmagic1",
        "ffmpeg",
        "wget",
        "libnss3",
        "libnspr4",
        "libatk1.0-0",
        "libatk-bridge2.0-0",
        "libcups2",
        "libatspi2.0-0",
        "libxcomposite1",
    )
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .env({"DB": db})
    .env({"LOCAL_API_URL": os.getenv("LOCAL_API_URL") or ""})
    .add_local_python_source("eve", ignore=[])
)


# Bot instance
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
intents.reactions = True
intents.dm_messages = True
intents.members = True

bot = discord.Bot(intents=intents)

# Slash commands
eden_group = discord.SlashCommandGroup("eden", "Eden bot commands")
access_group = eden_group.create_subgroup(
    "access", "Manage server access roles for Eden"
)


def can_manage_guild(member: discord.Member) -> bool:
    permissions = member.guild_permissions
    return permissions.administrator or permissions.manage_guild


def format_role_mentions(role_ids: List[str]) -> str:
    return ", ".join(f"<@&{role_id}>" for role_id in role_ids)


@access_group.command(name="add-role", description="Allow a role to deploy Eden agents")
async def access_add_role(ctx: discord.ApplicationContext, role: discord.Role):
    if not ctx.guild or not isinstance(ctx.author, discord.Member):
        await ctx.respond("This command must be used in a server.", ephemeral=True)
        return

    if not can_manage_guild(ctx.author):
        await ctx.respond(
            "You need Manage Server permissions to update access roles.",
            ephemeral=True,
        )
        return

    if role.id == ctx.guild.id:
        await ctx.respond("The @everyone role cannot be added.", ephemeral=True)
        return

    try:
        access_doc = DiscordGuildAccess.get_for_guild(str(ctx.guild.id))
        allowed_role_ids = access_doc.allowed_role_ids if access_doc else []
        if str(role.id) in allowed_role_ids:
            await ctx.respond(f"{role.mention} is already allowed.", ephemeral=True)
            return

        updated_roles = allowed_role_ids + [str(role.id)]
        DiscordGuildAccess.set_roles(
            guild_id=str(ctx.guild.id),
            role_ids=updated_roles,
            updated_by_discord_id=str(ctx.author.id),
        )

        await ctx.respond(
            f"Added {role.mention}. Allowed roles: {format_role_mentions(updated_roles)}",
            ephemeral=True,
        )
    except Exception:
        logger.exception("Failed to update access roles")
        await ctx.respond(
            "Failed to update access roles. Please try again later.",
            ephemeral=True,
        )


@access_group.command(
    name="remove-role", description="Remove a role from Eden deploy access"
)
async def access_remove_role(ctx: discord.ApplicationContext, role: discord.Role):
    if not ctx.guild or not isinstance(ctx.author, discord.Member):
        await ctx.respond("This command must be used in a server.", ephemeral=True)
        return

    if not can_manage_guild(ctx.author):
        await ctx.respond(
            "You need Manage Server permissions to update access roles.",
            ephemeral=True,
        )
        return

    try:
        access_doc = DiscordGuildAccess.get_for_guild(str(ctx.guild.id))
        allowed_role_ids = access_doc.allowed_role_ids if access_doc else []
        if str(role.id) not in allowed_role_ids:
            await ctx.respond(f"{role.mention} isn't allowed.", ephemeral=True)
            return

        updated_roles = [
            role_id for role_id in allowed_role_ids if role_id != str(role.id)
        ]
        DiscordGuildAccess.set_roles(
            guild_id=str(ctx.guild.id),
            role_ids=updated_roles,
            updated_by_discord_id=str(ctx.author.id),
        )

        if updated_roles:
            await ctx.respond(
                f"Removed {role.mention}. Allowed roles: {format_role_mentions(updated_roles)}",
                ephemeral=True,
            )
        else:
            await ctx.respond(
                f"Removed {role.mention}. No roles are allowed yet.",
                ephemeral=True,
            )
    except Exception:
        logger.exception("Failed to update access roles")
        await ctx.respond(
            "Failed to update access roles. Please try again later.",
            ephemeral=True,
        )


@access_group.command(
    name="list", description="List roles allowed to deploy Eden agents"
)
async def access_list(ctx: discord.ApplicationContext):
    if not ctx.guild:
        await ctx.respond("This command must be used in a server.", ephemeral=True)
        return

    try:
        access_doc = DiscordGuildAccess.get_for_guild(str(ctx.guild.id))
        allowed_role_ids = access_doc.allowed_role_ids if access_doc else []
        if not allowed_role_ids:
            await ctx.respond(
                "No roles are allowed yet. Use `/eden access add-role @role`.",
                ephemeral=True,
            )
            return

        await ctx.respond(
            f"Allowed roles: {format_role_mentions(allowed_role_ids)}",
            ephemeral=True,
        )
    except Exception:
        logger.exception("Failed to fetch access roles")
        await ctx.respond(
            "Failed to fetch access roles. Please try again later.",
            ephemeral=True,
        )


bot.add_application_command(eden_group)

# Typing managers
typing_managers: Dict[str, DiscordTypingManager] = {}
telegram_typing_manager: Optional[TelegramTypingManager] = None


@bot.event
async def on_ready():
    """Called when the bot is ready."""
    logger.info("=" * 80)
    logger.info("BOT IS READY!")
    logger.info(f"Logged in as: {bot.user} (ID: {bot.user.id})")
    logger.info(f"Guilds: {len(bot.guilds)}")
    logger.info("=" * 80)

    # Note: Typing indicators disabled for V3 (webhook-based, no typing support)


@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages."""
    logger.info(
        f"📨 Message received from {message.author.name} in channel {message.channel.id}"
    )

    # Skip bot messages
    if message.author.bot:
        logger.info(f"Skipping bot message from {message.author.name}")
        return

    channel_id = str(message.channel.id)
    guild_id = str(message.guild.id) if message.guild else None

    logger.info(
        f"Processing message: guild={guild_id}, channel={channel_id}, content={message.content[:50]}"
    )

    # Determine parent channel for threads
    parent_channel_id = None
    if isinstance(message.channel, discord.Thread):
        parent_channel_id = str(message.channel.parent_id)

    # Determine effective channel (for allowlist checking)
    effective_channel_id = parent_channel_id if parent_channel_id else channel_id

    logger.info(
        f"Message received: channel={channel_id}, guild={guild_id}, "
        f"author={message.author.name}, parent={parent_channel_id}"
    )

    # Find all deployments following this channel
    following_deployments = await find_all_following_deployments(effective_channel_id)

    if not following_deployments:
        logger.info(f"No deployments following channel {effective_channel_id}")
        return

    logger.info(f"Found {len(following_deployments)} deployments following channel")

    # Parse which deployments were mentioned
    mentioned_deployments = parse_mentioned_deployments(message, following_deployments)
    mentioned_ids = {d.id for d in mentioned_deployments}

    # Process for each following deployment
    tasks = []
    for deployment in following_deployments:
        # Determine if this deployment should respond
        has_write_access = deployment_can_write_to_channel(
            deployment, effective_channel_id
        )
        was_mentioned = deployment.id in mentioned_ids
        should_prompt = was_mentioned and has_write_access

        # For DMs, check if DMs are enabled
        if not guild_id:
            if not (
                deployment.config
                and deployment.config.discord
                and deployment.config.discord.enable_discord_dm
            ):
                # DMs disabled - send redirect message
                try:
                    agent = Agent.from_mongo(deployment.agent)
                    if agent:
                        eden_url = construct_agent_chat_url(agent.username)
                        await message.channel.send(
                            f"DMs are disabled for this bot. Please chat with me at {eden_url}"
                        )
                except Exception as e:
                    logger.error(f"Failed to send DM redirect: {e}")
                continue
            else:
                # DMs enabled - always prompt
                should_prompt = True

        # Process the message for this deployment
        task = process_discord_message_for_agent(
            message=message,
            deployment=deployment,
            should_prompt=should_prompt,
            parent_channel_id=parent_channel_id,
        )
        tasks.append(task)

    # Run all processing in parallel
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing message for deployment: {result}")
            else:
                logger.info(f"Processing result: {result}")


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    """Handle reaction add events."""
    # Skip bot reactions
    if payload.user_id == bot.user.id:
        return

    message_id = str(payload.message_id)
    user_id = str(payload.user_id)

    # Format emoji
    if payload.emoji.id:
        # Custom emoji
        emoji_str = f"<:{payload.emoji.name}:{payload.emoji.id}>"
    else:
        # Unicode emoji
        emoji_str = payload.emoji.name

    logger.info(
        f"Reaction added: message={message_id}, user={user_id}, emoji={emoji_str}"
    )

    try:
        # Find ChatMessage
        chat_msg = ChatMessage.find_one(
            {
                "channel.key": message_id,
                "channel.type": "discord",
            }
        )

        if not chat_msg:
            logger.info(f"ChatMessage not found for discord message {message_id}")
            return

        # Get or create User
        try:
            guild = bot.get_guild(payload.guild_id) if payload.guild_id else None
            if guild:
                member = await guild.fetch_member(payload.user_id)
                user = User.from_discord(
                    user_id,
                    member.name,
                    discord_avatar=member.avatar.key if member.avatar else None,
                )
            else:
                user = User.from_discord(user_id, str(payload.user_id))
        except Exception as e:
            logger.error(f"Failed to fetch user: {e}")
            user = User.from_discord(user_id, str(payload.user_id))

        # Add reaction
        chat_msg.react(user.id, emoji_str)
        chat_msg.save()

        logger.info(f"Added reaction to ChatMessage {chat_msg.id}")

    except Exception as e:
        logger.error(f"Error handling reaction add: {e}", exc_info=True)


@bot.event
async def on_raw_reaction_remove(payload: discord.RawReactionActionEvent):
    """Handle reaction remove events."""
    # Skip bot reactions
    if payload.user_id == bot.user.id:
        return

    message_id = str(payload.message_id)
    user_id = str(payload.user_id)

    # Format emoji
    if payload.emoji.id:
        # Custom emoji
        emoji_str = f"<:{payload.emoji.name}:{payload.emoji.id}>"
    else:
        # Unicode emoji
        emoji_str = payload.emoji.name

    logger.info(
        f"Reaction removed: message={message_id}, user={user_id}, emoji={emoji_str}"
    )

    try:
        # Find ChatMessage
        chat_msg = ChatMessage.find_one(
            {
                "channel.key": message_id,
                "channel.type": "discord",
            }
        )

        if not chat_msg:
            logger.info(f"ChatMessage not found for discord message {message_id}")
            return

        # Get user
        user = User.from_discord(user_id, str(payload.user_id))

        # Remove reaction
        if chat_msg.reactions:
            original_count = len(chat_msg.reactions)
            chat_msg.reactions = [
                r
                for r in chat_msg.reactions
                if not (r.get("user") == user.id and r.get("emoji") == emoji_str)
            ]
            if len(chat_msg.reactions) < original_count:
                chat_msg.save()
                logger.info(f"Removed reaction from ChatMessage {chat_msg.id}")

    except Exception as e:
        logger.error(f"Error handling reaction remove: {e}", exc_info=True)


# Typing indicators not supported for V3 (webhook-based deployments)
# Webhooks cannot show typing status in Discord


# ============================================================================
# FASTAPI (for webhooks)
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Start bot in background thread to avoid event loop conflicts with Modal
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        logger.error("DISCORD_BOT_TOKEN not set!")
        raise ValueError("DISCORD_BOT_TOKEN is required")

    logger.info(f"Starting Discord bot with token: {bot_token[:20]}...")

    def run_bot_in_thread():
        """Run bot in a separate thread with its own event loop."""
        try:
            logger.info("Starting bot in separate thread...")
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            logger.info("Calling bot.run()...")
            bot.run(bot_token)
            logger.info("bot.run() completed (bot disconnected)")
        except Exception as e:
            logger.error(f"Bot failed in thread: {e}", exc_info=True)

    bot_thread = threading.Thread(
        target=run_bot_in_thread, daemon=True, name="discord-bot"
    )
    bot_thread.start()

    # Wait for bot to connect
    logger.info("Waiting for bot to connect...")
    await asyncio.sleep(5)
    logger.info("Bot connection wait complete, proceeding with app startup")

    yield

    # Cleanup
    logger.info("Shutting down Discord bot...")
    await bot.close()


fastapi_app = FastAPI(lifespan=lifespan)


# ============================================================================
# MODAL DEPLOYMENT
# ============================================================================


@modal_app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("eve-secrets", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-{db}", environment_name="main"),
        modal.Secret.from_name("eve-secrets-gateway", environment_name="main"),
        modal.Secret.from_name(f"eve-secrets-gateway-v3-{db}", environment_name="main"),
    ],
    min_containers=1,
    timeout=86400,  # 24 hours
    scaledown_window=600,
)
@modal.asgi_app()
def gateway_app():
    """Modal ASGI app entrypoint."""
    return fastapi_app


# Alias for modal serve
app = modal_app
