#!/usr/bin/env python3
"""Backfill guild_id for Discord V3 channel configs.

Uses DISCORD_BOT_TOKEN to resolve channel -> guild_id when missing.
Defaults to dry-run.

Run with:
  DB=PROD DISCORD_BOT_TOKEN=... PYTHONPATH=/Users/jmill/projects-old/eden/eve \\
  rye run python /Users/jmill/projects-old/eden/eve/scripts/backfill_discord_v3_channel_guilds.py
"""

import argparse
import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from eve.mongo import get_collection

DISCORD_API_BASE = "https://discord.com/api/v10"


async def fetch_guild_id(
    session: aiohttp.ClientSession, token: str, channel_id: str
) -> Optional[str]:
    url = f"{DISCORD_API_BASE}/channels/{channel_id}"
    headers = {"Authorization": f"Bot {token}"}
    async with session.get(url, headers=headers) as response:
        if response.status != 200:
            text = await response.text()
            logger.warning(
                f"Failed to fetch channel {channel_id} (status {response.status}): {text}"
            )
            return None
        payload = await response.json()
        return payload.get("guild_id")


async def fetch_guild_name(
    session: aiohttp.ClientSession, token: str, guild_id: str
) -> Optional[str]:
    url = f"{DISCORD_API_BASE}/guilds/{guild_id}"
    headers = {"Authorization": f"Bot {token}"}
    async with session.get(url, headers=headers) as response:
        if response.status != 200:
            text = await response.text()
            logger.warning(
                f"Failed to fetch guild {guild_id} (status {response.status}): {text}"
            )
            return None
        payload = await response.json()
        return payload.get("name")


def normalize_channel(
    channel: Dict[str, Any], guild_id: Optional[str]
) -> Dict[str, Any]:
    updated = dict(channel)
    if guild_id:
        updated["guild_id"] = guild_id
    return updated


async def process_deployments(dry_run: bool, limit: Optional[int], force_fetch: bool):
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is required for this script")

    collection = get_collection("deployments2")
    query = {
        "platform": "discord_v3",
        "config.discord.channel_configs": {"$exists": True, "$ne": []},
    }

    cursor = collection.find(query)
    if limit is not None:
        cursor = cursor.limit(limit)

    total = 0
    updated_count = 0

    async with aiohttp.ClientSession() as session:
        for doc in cursor:
            total += 1
            deployment_id = doc.get("_id")
            discord_config = (doc.get("config") or {}).get("discord") or {}
            config_guild_id = discord_config.get("guild_id")
            config_guild_name = discord_config.get("guild_name")
            config_role_id = discord_config.get("role_id")
            config_role_name = discord_config.get("role_name")
            channel_configs = discord_config.get("channel_configs") or []
            existing_guilds = discord_config.get("guilds") or []
            guilds_by_id = {g.get("guild_id"): g for g in existing_guilds if g}

            updated_channels: List[Dict[str, Any]] = []
            changed = False

            for channel in channel_configs:
                channel_id = channel.get("channel_id")
                existing_guild_id = channel.get("guild_id")

                if existing_guild_id and not force_fetch:
                    updated_channels.append(channel)
                    continue

                resolved_guild_id = None
                if channel_id:
                    resolved_guild_id = await fetch_guild_id(
                        session, bot_token, str(channel_id)
                    )
                if not resolved_guild_id and existing_guild_id:
                    resolved_guild_id = existing_guild_id

                if not resolved_guild_id:
                    logger.warning(
                        f"Could not resolve guild for channel {channel_id} in deployment {deployment_id}"
                    )
                if resolved_guild_id and resolved_guild_id != existing_guild_id:
                    updated_channels.append(
                        normalize_channel(channel, resolved_guild_id)
                    )
                    changed = True
                else:
                    updated_channels.append(channel)

            guild_ids = {
                ch.get("guild_id") for ch in updated_channels if ch.get("guild_id")
            }

            updated_guilds = []
            guilds_changed = False
            for guild_id in sorted(guild_ids):
                existing = guilds_by_id.get(guild_id) or {}
                guild_name = existing.get("guild_name")
                role_id = existing.get("role_id")
                role_name = existing.get("role_name")

                if (
                    not guild_name
                    and config_guild_id
                    and guild_id == config_guild_id
                    and config_guild_name
                ):
                    guild_name = config_guild_name
                if not role_id and config_guild_id and guild_id == config_guild_id:
                    role_id = config_role_id
                    role_name = role_name or config_role_name

                if not guild_name:
                    guild_name = await fetch_guild_name(session, bot_token, guild_id)

                updated_guild = {
                    "guild_id": guild_id,
                    "guild_name": guild_name,
                    "role_id": role_id,
                    "role_name": role_name,
                }
                updated_guilds.append(updated_guild)

                if existing != updated_guild:
                    guilds_changed = True

            if not changed and not guilds_changed:
                logger.info(f"No changes for deployment {deployment_id}")
                continue

            updated_count += 1
            logger.info(
                f"Deployment {deployment_id}: would update {sum(1 for c in updated_channels if c.get('guild_id'))} channel configs and {len(updated_guilds)} guild configs"
            )

            if not dry_run:
                collection.update_one(
                    {"_id": deployment_id},
                    {
                        "$set": {
                            "config.discord.channel_configs": updated_channels,
                            "config.discord.guilds": updated_guilds,
                        }
                    },
                )

    logger.info(
        f"Done. Reviewed {total} deployments, updated {updated_count}. Dry run: {dry_run}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill guild_id for Discord V3 channel configs",
    )
    parser.add_argument(
        "--dry-run",
        type=lambda value: value.lower() == "true",
        default=True,
        help="Run in dry-run mode (default: true). Use --dry-run=false to apply changes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of deployments processed",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Always fetch guild_id from Discord API even if config.guild_id exists",
    )

    args = parser.parse_args()

    asyncio.run(
        process_deployments(
            dry_run=args.dry_run, limit=args.limit, force_fetch=args.force_fetch
        )
    )


if __name__ == "__main__":
    main()
