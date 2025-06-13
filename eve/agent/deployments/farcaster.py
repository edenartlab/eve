import os
import aiohttp

from eve.api.errors import APIError
from eve.agent.deployments import (
    PlatformClient,
    DeploymentSecrets,
    DeploymentConfig,
    DeploymentSettingsFarcaster,
)


class FarcasterClient(PlatformClient):
    TOOLS = {}  # No tools for Farcaster yet

    async def predeploy(
        self, secrets: DeploymentSecrets, config: DeploymentConfig
    ) -> tuple[DeploymentSecrets, DeploymentConfig]:
        """Verify Farcaster credentials"""
        try:
            from farcaster import Warpcast

            client = Warpcast(mnemonic=secrets.farcaster.mnemonic)

            # Test the credentials by getting user info
            user_info = client.get_me()
            print(f"Verified Farcaster credentials for user: {user_info}")
        except Exception as e:
            raise APIError(f"Invalid Farcaster credentials: {str(e)}", status_code=400)

        # Generate webhook secret if not provided
        if not secrets.farcaster.neynar_webhook_secret:
            import secrets as python_secrets

            webhook_secret = python_secrets.token_urlsafe(32)
            secrets.farcaster.neynar_webhook_secret = webhook_secret

        return secrets, config

    async def postdeploy(self) -> None:
        """Register webhook with Neynar"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        webhook_url = f"{os.getenv('EDEN_API_URL')}/updates/platform/farcaster"

        async with aiohttp.ClientSession() as session:
            # Get Neynar API key from environment
            neynar_api_key = os.getenv("NEYNAR_API_KEY")
            if not neynar_api_key:
                raise Exception("NEYNAR_API_KEY not found in environment")

            headers = {
                "x-api-key": f"{neynar_api_key}",
                "Content-Type": "application/json",
            }

            # Get user info for webhook registration
            from farcaster import Warpcast

            client = Warpcast(mnemonic=self.deployment.secrets.farcaster.mnemonic)
            user_info = client.get_me()

            webhook_data = {
                "name": f"eden-{self.deployment.id}",
                "url": webhook_url,
                "subscription": {"cast.created": {"mentioned_fids": [user_info.fid]}},
            }

            async with session.post(
                "https://api.neynar.com/v2/farcaster/webhook",
                headers=headers,
                json=webhook_data,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to register Neynar webhook: {error_text}")

                webhook_response = await response.json()
                webhook_id = webhook_response.get("webhook", {}).get("webhook_id")
                webhook_secret = (
                    webhook_response.get("webhook", {})
                    .get("secrets", [{}])[0]
                    .get("value")
                )

                if not webhook_id:
                    raise Exception("No webhook_id in response")

                # Update the webhook secret in deployment secrets
                self.deployment.secrets.farcaster.neynar_webhook_secret = webhook_secret

                # Store webhook ID in deployment for later cleanup
                if not self.deployment.config:
                    self.deployment.config = DeploymentConfig()
                if not self.deployment.config.farcaster:
                    self.deployment.config.farcaster = DeploymentSettingsFarcaster()

                self.deployment.config.farcaster.webhook_id = webhook_id
                self.deployment.save()

                print(
                    f"Registered Neynar webhook {webhook_id} for deployment {self.deployment.id}"
                )

    async def stop(self) -> None:
        """Stop Farcaster client by unregistering webhook from Neynar"""
        if not self.deployment:
            raise ValueError("Deployment is required for stop")

        if self.deployment.config and self.deployment.config.farcaster:
            webhook_id = getattr(self.deployment.config.farcaster, "webhook_id", None)
            if webhook_id:
                try:
                    async with aiohttp.ClientSession() as session:
                        neynar_api_key = os.getenv("NEYNAR_API_KEY")
                        headers = {
                            "x-api-key": f"{neynar_api_key}",
                            "Content-Type": "application/json",
                        }

                        webhook_data = {"webhook_id": webhook_id}

                        async with session.delete(
                            "https://api.neynar.com/v2/farcaster/webhook",
                            headers=headers,
                            json=webhook_data,
                        ) as response:
                            if response.status == 200:
                                print(
                                    f"Successfully unregistered Neynar webhook {webhook_id}"
                                )
                                # Clear the webhook_id from config after successful deletion
                                self.deployment.config.farcaster.webhook_id = None
                                self.deployment.save()
                            else:
                                error_text = await response.text()
                                print(f"Failed to unregister webhook: {error_text}")

                except Exception as e:
                    print(f"Error unregistering Neynar webhook: {e}")
