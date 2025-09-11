import os
import json
import logging
from datetime import datetime, timedelta

import aiohttp
from fastapi import Request

from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import DeploymentSecrets, DeploymentConfig
from eve.agent.llm import UpdateType
from eve.utils import prepare_result

logger = logging.getLogger(__name__)


class InstagramClient(PlatformClient):
    TOOLS = [
        "instagram_post",
    ]

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Validate Instagram access token and add Instagram tools"""
        try:
            # Validate access token by fetching user info
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://graph.instagram.com/me",
                    params={
                        "fields": "id,username,account_type",
                        "access_token": secrets.instagram.access_token
                    },
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise Exception(f"Invalid Instagram access token: {error_data.get('error', {}).get('message', 'Unknown error')}")
                    
                    user_info = await response.json()
                    print(f"Verified Instagram account: {user_info.get('username')} (ID: {user_info.get('id')})")
                    
                    # Update secrets with user info
                    if not secrets.instagram.user_id:
                        secrets.instagram.user_id = user_info.get("id")
                    if not secrets.instagram.username:
                        secrets.instagram.username = user_info.get("username")

        except Exception as e:
            raise APIError(f"Invalid Instagram access token: {str(e)}", status_code=400)

        try:
            # Add Instagram tools to agent
            self.add_tools()
        except Exception as e:
            raise APIError(f"Failed to add Instagram tools: {str(e)}", status_code=400)

        return secrets, config

    async def postdeploy(self) -> None:
        """Post-deployment setup for Instagram"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        logger.info(f"Instagram deployment {self.deployment.id} is now active")

    async def stop(self) -> None:
        """Stop Instagram client"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        try:
            # Remove Instagram tools
            self.remove_tools()
            logger.info(f"Stopped Instagram deployment {self.deployment.id}")

        except Exception as e:
            logger.error(f"Failed to stop Instagram deployment: {e}")

    async def interact(self, request: Request) -> None:
        """Handle session interactions for Instagram"""
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
                        "X-Client-Platform": "instagram",
                        "X-Client-Deployment-Id": str(self.deployment.id),
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Failed to process session interaction: {error_text}"
                        )

                    logger.info(
                        f"Successfully handled Instagram session interaction for deployment {self.deployment.id}"
                    )

        except Exception as e:
            logger.error(f"Error handling Instagram interaction: {str(e)}", exc_info=True)
            raise

    async def handle_emission(self, emission) -> None:
        """Handle an emission from the platform client"""
        try:
            if not self.deployment:
                raise ValueError("Deployment is required for handle_emission")

            update_type = emission.type

            # For Instagram, we don't need to handle emissions like Discord
            # since posts are published directly through the API
            if update_type == UpdateType.TOOL_COMPLETE:
                logger.info("Instagram tool completed successfully")
            elif update_type == UpdateType.ERROR:
                error_msg = emission.error or "Unknown error occurred"
                logger.error(f"Instagram tool error: {error_msg}")
            else:
                logger.debug(f"Ignoring emission type: {update_type}")

        except Exception as e:
            logger.error(f"Error handling Instagram emission: {str(e)}", exc_info=True)
            raise

    async def refresh_token_if_needed(self) -> None:
        """Refresh Instagram access token if it's close to expiration"""
        if not self.deployment or not self.deployment.secrets.instagram:
            return

        secrets = self.deployment.secrets.instagram
        
        # Check if token expires within 7 days
        if secrets.expires_at and secrets.expires_at <= datetime.now() + timedelta(days=7):
            logger.warning(f"Instagram access token for deployment {self.deployment.id} is expiring soon")
            # Note: Long-lived tokens need to be refreshed via the Graph API
            # This would require implementing token refresh logic