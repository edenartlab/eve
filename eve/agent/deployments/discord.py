import os
import json
import logging

import aiohttp
from ably import AblyRest
from fastapi import Request

from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig
from eve.agent.llm import UpdateType
from eve.eden_utils import prepare_result
from eve.api.helpers import get_eden_creation_url

logger = logging.getLogger(__name__)
db = os.getenv("DB", "STAGE").upper()


class DiscordClient(PlatformClient):
    TOOLS = {
        "discord_post": {},
    }

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Discord token and add Discord tools"""
        try:
            # Validate bot token
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://discord.com/api/v10/users/@me",
                    headers={"Authorization": f"Bot {secrets.discord.token}"},
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Invalid token: {await response.text()}")
                    bot_info = await response.json()
                    print(f"Verified Discord bot: {bot_info['username']}")

            # Add Discord tools to agent
            self.add_tools()

            return secrets, config
        except Exception as e:
            raise APIError(f"Invalid Discord token: {str(e)}", status_code=400)

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
            print(f"Sent start command for deployment {self.deployment.id} via Ably")
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
            print(f"Sent stop command for deployment {self.deployment.id} via Ably")

            # Remove Discord tools
            self.remove_tools()

        except Exception as e:
            print(f"Failed to notify gateway service: {e}")

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

    async def handle_emission(self, emission) -> None:
        """Handle an emission from the platform client"""
        try:
            if not self.deployment:
                raise ValueError("Deployment is required for handle_emission")

            # Extract context from update_config
            channel_id = emission.update_config.discord_channel_id
            message_id = emission.update_config.discord_message_id

            if not channel_id:
                logger.error("Missing discord_channel_id in update_config")
                return

            update_type = emission.type

            # Initialize Discord REST client
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bot {self.deployment.secrets.discord.token}",
                    "Content-Type": "application/json",
                }

                payload = {}
                if message_id:
                    payload["message_reference"] = {
                        "message_id": message_id,
                        "channel_id": channel_id,
                        "fail_if_not_exists": False,
                    }

                if update_type == UpdateType.ASSISTANT_MESSAGE:
                    content = emission.content
                    if content:
                        payload["content"] = content

                elif update_type == UpdateType.TOOL_COMPLETE:
                    result = emission.result
                    if not result:
                        logger.debug("No tool result to post")
                        return

                    # ***debug*** Log raw result
                    logger.info(f"***debug*** Raw result before processing: {result}")
                    logger.info(f"***debug*** Result type: {type(result)}")

                    # Process result to extract media URLs
                    processed_result = prepare_result(json.loads(result))
                    logger.info(f"***debug*** Processed result: {processed_result}")

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
                    logger.debug(f"Ignoring emission type: {update_type}")
                    return

                # Send the message if we have content
                if payload.get("content"):
                    async with session.post(
                        f"https://discord.com/api/v10/channels/{channel_id}/messages",
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status == 200:
                            logger.info(
                                f"Successfully sent Discord message to channel {channel_id}"
                            )
                        else:
                            error_text = await response.text()
                            logger.error(
                                f"Failed to send Discord message: {error_text}"
                            )
                            raise Exception(
                                f"Failed to send Discord message: {error_text}"
                            )

        except Exception as e:
            logger.error(f"Error handling Discord emission: {str(e)}", exc_info=True)
            raise
