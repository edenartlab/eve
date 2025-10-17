import os
import json

import aiohttp
from ably import AblyRest
from fastapi import Request
from loguru import logger

from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig
from eve.agent.session.models import UpdateType
from eve.utils import prepare_result
from eve.api.helpers import get_eden_creation_url

db = os.getenv("DB", "STAGE").upper()


class DiscordClient(PlatformClient):
    TOOLS = [
        "discord_post",
        "discord_search",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Discord token and add Discord tools"""
        try:
            # Validate bot token and get application info
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://discord.com/api/v10/users/@me",
                    headers={"Authorization": f"Bot {secrets.discord.token}"},
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Invalid token: {await response.text()}")
                    bot_info = await response.json()

                    # Get application ID and construct OAuth URL
                    application_id = bot_info.get("id")
                    if application_id:
                        # Initialize config if it doesn't exist
                        if config is None:
                            from eve.agent.session.models import (
                                DeploymentSettingsDiscord,
                            )

                            config = DeploymentConfig(
                                discord=DeploymentSettingsDiscord()
                            )
                        elif config.discord is None:
                            from eve.agent.session.models import (
                                DeploymentSettingsDiscord,
                            )

                            config.discord = DeploymentSettingsDiscord()

                        # Set the OAuth client ID and URL
                        config.discord.oauth_client_id = application_id
                        config.discord.oauth_url = f"https://discord.com/oauth2/authorize?client_id={application_id}&permissions=274877958144&integration_type=0&scope=bot"

                        # Also store application_id in secrets if needed
                        if secrets.discord and not secrets.discord.application_id:
                            secrets.discord.application_id = application_id

        except Exception as e:
            raise APIError(f"Invalid Discord token: {str(e)}", status_code=400)

        try:
            # Add Discord tools to agent
            self.add_tools()
        except Exception as e:
            raise APIError(f"Failed to add Discord tools: {str(e)}", status_code=400)

        return secrets, config

    async def postdeploy(self) -> None:
        """Notify Discord gateway service via Ably"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-v2-{db}")

            await channel.publish(
                "command",
                {"command": "start", "deployment_id": str(self.deployment.id)},
            )
        except Exception as e:
            raise Exception(f"Failed to notify gateway service: {e}")

    async def stop(self) -> None:
        """Stop Discord client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        try:
            ably_client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = ably_client.channels.get(f"discord-gateway-v2-{db}")

            await channel.publish(
                "command",
                {"command": "stop", "deployment_id": str(self.deployment.id)},
            )

            # Remove Discord tools
            self.remove_tools()

        except Exception as e:
            logger.error(f"Failed to notify gateway service: {e}")

    async def interact(self, request: Request) -> None:
        """Handle session interactions for Discord"""
        try:
            from eve.api.api_requests import DeploymentInteractRequest

            # Parse the interaction request
            data = await request.json()
            interact_request = DeploymentInteractRequest(**data)

            # Forward the session request to the sessions API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{os.getenv('EDEN_API_URL')}/sessions/prompt",
                    json=interact_request.interaction.model_dump(),
                    headers={
                        "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                        "Content-Type": "application/json",
                        "X-Client-Platform": "discord",
                        "X-Client-Deployment-Id": str(self.deployment.id),
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Failed to process session interaction: {error_text}"
                        )

                    logger.info(
                        f"Successfully handled Discord session interaction for deployment {self.deployment.id}"
                    )

        except Exception as e:
            logger.error(f"Error handling Discord interaction: {str(e)}", exc_info=True)
            raise

    def _validate_emission_context(self, emission) -> tuple:
        """
        Validate and extract emission context.
        Returns: (is_dm, channel_id, message_id, discord_user_id)
        """
        channel_id = emission.update_config.discord_channel_id
        message_id = emission.update_config.discord_message_id
        discord_user_id = getattr(emission.update_config, 'discord_user_id', None)

        is_dm = discord_user_id is not None and channel_id is None

        # Validate required IDs based on message type
        if is_dm and not discord_user_id:
            logger.error("Missing discord_user_id for DM")
            return None, None, None, None

        if not is_dm and not channel_id:
            logger.error("Missing discord_channel_id in update_config")
            return None, None, None, None

        return is_dm, channel_id, message_id, discord_user_id

    def _build_message_payload(
        self,
        emission,
        is_dm: bool,
        channel_id: str,
        message_id: str
    ) -> dict:
        """Build the message payload based on emission type"""
        payload = {}

        # Only add message_reference for non-DM channel messages
        if message_id and not is_dm and channel_id:
            payload["message_reference"] = {
                "message_id": message_id,
                "channel_id": channel_id,
                "fail_if_not_exists": False,
            }

        update_type = emission.type

        if update_type == UpdateType.ASSISTANT_MESSAGE:
            content = emission.content
            if content:
                payload["content"] = content

        elif update_type == UpdateType.TOOL_COMPLETE:
            result = emission.result
            if not result:
                logger.debug("No tool result to post")
                return None

            # Process result to extract media URLs
            processed_result = prepare_result(json.loads(result))

            if (
                processed_result.get("result")
                and len(processed_result["result"]) > 0
                and "output" in processed_result["result"][0]
            ):
                outputs = processed_result["result"][0]["output"]

                # Extract URLs from outputs
                urls = []
                for output in outputs[:4]:  # Discord supports up to 4 embeds
                    if isinstance(output, dict) and "url" in output:
                        urls.append(output["url"])

                if urls:
                    # Prepare message content with URLs
                    content = "\n".join(urls)
                    payload["content"] = content

                    # Get creation ID from the first output for Eden link
                    creation_id = None
                    if isinstance(outputs, list) and len(outputs) > 0:
                        creation_id = str(outputs[0].get("creation"))

                    # Add components for Eden link if creation_id exists
                    if creation_id:
                        eden_url = get_eden_creation_url(creation_id)
                        payload["components"] = [
                            {
                                "type": 1,  # Action Row
                                "components": [
                                    {
                                        "type": 2,  # Button
                                        "style": 5,  # Link
                                        "label": "View on Eden",
                                        "url": eden_url,
                                    }
                                ],
                            }
                        ]
                else:
                    logger.warning(
                        "No valid URLs found in tool result for Discord"
                    )
            else:
                logger.warning(
                    "Unexpected tool result structure for Discord emission"
                )

        elif update_type == UpdateType.ERROR:
            error_msg = emission.error or "Unknown error occurred"
            payload["content"] = f"Error: {error_msg}"

        else:
            logger.debug(f"update_type: {update_type}")
            logger.debug(f"Ignoring emission type: {update_type}")
            return None

        return payload

    def _chunk_message(self, content: str, max_length: int = 2000) -> list[str]:
        """
        Split a message into chunks that fit within Discord's character limit.
        Tries to split at newlines to keep formatting intact.
        """
        if len(content) <= max_length:
            return [content]

        chunks = []
        remaining = content

        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break

            # Try to find a good break point (newline) within the limit
            chunk = remaining[:max_length]

            # Look for the last newline in this chunk
            last_newline = chunk.rfind('\n')

            if last_newline > 0 and last_newline > max_length * 0.5:
                # Use newline as break point if it's not too early
                split_point = last_newline + 1
            else:
                # Otherwise, try to break at a space
                last_space = chunk.rfind(' ')
                if last_space > 0 and last_space > max_length * 0.5:
                    split_point = last_space + 1
                else:
                    # Hard break at max_length
                    split_point = max_length

            chunks.append(remaining[:split_point])
            remaining = remaining[split_point:]

        return chunks

    async def _send_dm_message(
        self,
        discord_user_id: str,
        payload: dict,
        headers: dict
    ) -> None:
        """Send a message via Discord DM, chunking if necessary"""
        async with aiohttp.ClientSession() as session:
            # Create DM channel
            dm_channel_response = await session.post(
                "https://discord.com/api/v10/users/@me/channels",
                headers=headers,
                json={"recipient_id": discord_user_id},
            )
            if dm_channel_response.status != 200:
                error_text = await dm_channel_response.text()
                logger.error(f"Failed to create DM channel: {error_text}")
                raise Exception(f"Failed to create DM channel: {error_text}")

            dm_channel_data = await dm_channel_response.json()
            dm_channel_id = dm_channel_data["id"]

            # Send message to DM channel, chunking if necessary
            url = f"https://discord.com/api/v10/channels/{dm_channel_id}/messages"

            content = payload.get("content", "")
            chunks = self._chunk_message(content)

            # Send each chunk
            for i, chunk in enumerate(chunks):
                chunk_payload = payload.copy()
                chunk_payload["content"] = chunk

                # Only include message_reference and components in the first chunk
                if i > 0:
                    chunk_payload.pop("message_reference", None)
                    chunk_payload.pop("components", None)

                async with session.post(url, headers=headers, json=chunk_payload) as response:
                    if response.status == 200:
                        logger.info(f"Successfully sent Discord DM chunk {i+1}/{len(chunks)} to user {discord_user_id}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to send Discord message chunk {i+1}: {error_text}")
                        raise Exception(f"Failed to send Discord message: {error_text}")

    async def _send_channel_message(
        self,
        channel_id: str,
        payload: dict,
        headers: dict
    ) -> None:
        """Send a message to a Discord channel, chunking if necessary"""
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"

        content = payload.get("content", "")
        chunks = self._chunk_message(content)

        async with aiohttp.ClientSession() as session:
            # Send each chunk
            for i, chunk in enumerate(chunks):
                chunk_payload = payload.copy()
                chunk_payload["content"] = chunk

                # Only include message_reference and components in the first chunk
                if i > 0:
                    chunk_payload.pop("message_reference", None)
                    chunk_payload.pop("components", None)

                async with session.post(url, headers=headers, json=chunk_payload) as response:
                    if response.status == 200:
                        logger.info(f"Successfully sent Discord message chunk {i+1}/{len(chunks)} to channel {channel_id}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to send Discord message chunk {i+1}: {error_text}")
                        raise Exception(f"Failed to send Discord message: {error_text}")

    async def handle_emission(self, emission) -> None:
        """Handle an emission from the platform client"""
        try:
            if not self.deployment:
                raise ValueError("Deployment is required for handle_emission")

            # Validate and extract context
            is_dm, channel_id, message_id, discord_user_id = self._validate_emission_context(emission)
            if is_dm is None:  # Validation failed
                return

            # Build message payload
            payload = self._build_message_payload(emission, is_dm, channel_id, message_id)
            if not payload or not payload.get("content"):
                logger.debug("No content to send")
                return

            # Send message
            headers = {
                "Authorization": f"Bot {self.deployment.secrets.discord.token}",
                "Content-Type": "application/json",
            }

            if is_dm:
                await self._send_dm_message(discord_user_id, payload, headers)
            else:
                await self._send_channel_message(channel_id, payload, headers)

        except Exception as e:
            logger.error("Error handling Discord emission: {error}", error=str(e), exc_info=True)
            raise
