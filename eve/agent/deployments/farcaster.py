import json
import os
import aiohttp
import logging
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse

from eve.agent.session.models import (
    ChatMessageRequestInput,
    Session,
    SessionUpdateConfig,
    Deployment,
    DeploymentSecrets,
    DeploymentConfig,
    DeploymentSettingsFarcaster,
)
from eve.api.api_requests import SessionCreationArgs
from eve.api.errors import APIError
from eve.agent.deployments import PlatformClient
from eve.agent.llm import UpdateType
from eve.eden_utils import prepare_result
from eve.user import User
import eve.mongo

if TYPE_CHECKING:
    from eve.api.api_requests import DeploymentEmissionRequest, PromptSessionRequest

logger = logging.getLogger(__name__)


class FarcasterClient(PlatformClient):
    TOOLS = [
        "farcaster_cast",
        "farcaster_search",
        "farcaster_mentions",
    ]

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

        try:
            # Add Farcaster tools to agent
            self.add_tools()
        except Exception as e:
            raise APIError(f"Failed to add Farcaster tools: {str(e)}", status_code=400)

        return secrets, config

    async def postdeploy(self) -> None:
        """Register webhook with Neynar"""
        if not self.deployment:
            raise ValueError("Deployment is required for postdeploy")

        webhook_url = (
            f"{os.getenv('EDEN_API_URL')}/v2/deployments/farcaster/neynar-webhook"
        )

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

        try:
            # Remove Farcaster tools
            self.remove_tools()
        except Exception as e:
            print(f"Failed to remove Farcaster tools: {e}")

    async def interact(self, request: Request) -> None:
        """Interact with the Farcaster client"""
        raise NotImplementedError(
            "Interact() with the Farcaster client is not supported"
        )

    async def handle_neynar_webhook(self, request: Request) -> None:
        """Interact with the Farcaster client"""
        import hmac
        import hashlib

        from eve.api.api_requests import PromptSessionRequest

        # Verify Neynar webhook signature
        body = await request.body()
        signature = request.headers.get("X-Neynar-Signature")
        if not signature:
            return JSONResponse(
                status_code=401, content={"error": "Missing signature header"}
            )

        # Find deployment by webhook secret - we'll store this in the deployment
        # For now, let's extract the webhook secret from headers or find another way
        webhook_data = await request.json()
        cast_data = webhook_data.get("data", {})

        if not cast_data or "hash" not in cast_data:
            return JSONResponse(status_code=400, content={"error": "Invalid cast data"})

        # For now, we'll need to find the deployment differently
        # We could use the cast author or have Neynar include a custom field
        # Let's assume we can match by the webhook signature for now
        # TODO: can neynar pass us a deployment id?
        deployment = None
        for d in Deployment.find({"platform": "farcaster"}):
            if (
                d.secrets
                and d.secrets.farcaster
                and d.secrets.farcaster.neynar_webhook_secret
            ):
                # Verify signature
                computed_signature = hmac.new(
                    d.secrets.farcaster.neynar_webhook_secret.encode(),
                    body,
                    hashlib.sha512,
                ).hexdigest()
                if hmac.compare_digest(computed_signature, signature):
                    deployment = d
                    break

        if not deployment:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid signature or deployment not found"},
            )

        if not deployment.config.farcaster.auto_reply:
            return JSONResponse(status_code=200, content={"ok": True})

        # Create chat request similar to Telegram
        cast_hash = cast_data["hash"]
        author = cast_data["author"]
        author_username = author["username"]
        author_fid = author["fid"]

        # Get or create user
        user = User.from_farcaster(author_fid, author_username)

        session_key = f"farcaster-{cast_hash}"

        # attempt to get session by session_key
        try:
            session = Session.load(session_key=session_key)
        except Exception as e:
            if isinstance(e, eve.mongo.MongoDocumentNotFound):
                session = None
            else:
                raise e

        prompt_session_request = PromptSessionRequest(
            user_id=str(user.id),
            actor_agent_id=str(deployment.agent),
            message=ChatMessageRequestInput(
                content=cast_data.get("text", ""),
                sender_name=author_username,
            ),
            update_config=SessionUpdateConfig(
                deployment_id=str(deployment.id),
                update_endpoint=f"{os.getenv('EDEN_API_URL')}/v2/deployments/emission",
                farcaster_hash=cast_hash,
                farcaster_author_fid=author_fid,
            ),
        )

        # create session if it doesn't exist
        if session:
            prompt_session_request.session_id = str(session.id)
        else:
            prompt_session_request.creation_args = SessionCreationArgs(
                owner_id=str(user.id),
                agents=[str(deployment.agent)],
                title=f"Farcaster cast {cast_hash}",
                session_key=session_key,
                platform="farcaster",
            )

        # Make async HTTP POST to /chat
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('EDEN_API_URL')}/sessions/prompt",
                json=prompt_session_request.model_dump(),
                headers={
                    "Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": f"Failed to process chat request: {error_text}"
                        },
                    )

        return JSONResponse(status_code=200, content={"ok": True})

    async def handle_emission(self, emission: "DeploymentEmissionRequest") -> None:
        """Handle an emission from the platform client"""
        try:
            if not self.deployment:
                raise ValueError("Deployment is required for handle_emission")

            # Extract context from update_config
            cast_hash = emission.update_config.farcaster_hash
            author_fid = emission.update_config.farcaster_author_fid

            if not cast_hash or not author_fid:
                logger.error(
                    "Missing farcaster_hash or farcaster_author_fid in update_config"
                )
                return

            # Initialize Farcaster client
            from farcaster import Warpcast

            client = Warpcast(mnemonic=self.deployment.secrets.farcaster.mnemonic)

            update_type = emission.type

            if update_type == UpdateType.ASSISTANT_MESSAGE:
                content = emission.content
                if content:
                    try:
                        client.post_cast(
                            text=content,
                            parent={"hash": cast_hash, "fid": author_fid},
                        )
                        logger.info(
                            f"Posted assistant message cast in reply to {cast_hash}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to post cast: {str(e)}")
                        raise

            elif update_type == UpdateType.TOOL_COMPLETE:
                result = emission.result
                if not result:
                    logger.debug("No tool result to post")
                    return

                # Process result to extract media URLs
                processed_result = prepare_result(json.loads(result))

                if (
                    processed_result.get("result")
                    and len(processed_result["result"]) > 0
                    and "output" in processed_result["result"][0]
                ):
                    outputs = processed_result["result"][0]["output"]

                    # Extract URLs from outputs (up to 4 for Farcaster limit)
                    urls = []
                    for output in outputs[:4]:
                        if isinstance(output, dict) and "url" in output:
                            urls.append(output["url"])

                    if urls:
                        try:
                            # Post cast with media embeds
                            client.post_cast(
                                text="",  # Empty text, just media
                                embeds=urls,
                                parent={"hash": cast_hash, "fid": author_fid},
                            )
                            logger.info(
                                f"Posted tool result cast with {len(urls)} embeds in reply to {cast_hash}"
                            )
                        except Exception as e:
                            logger.error(f"Failed to post cast with embeds: {str(e)}")
                            raise
                    else:
                        logger.warning(
                            "No valid URLs found in tool result for Farcaster embeds"
                        )
                else:
                    logger.warning(
                        "Unexpected tool result structure for Farcaster emission"
                    )

            elif update_type == UpdateType.ERROR:
                error_msg = emission.error or "Unknown error occurred"
                try:
                    client.post_cast(
                        text=f"Error: {error_msg}",
                        parent={"hash": cast_hash, "fid": author_fid},
                    )
                    logger.info(f"Posted error message cast in reply to {cast_hash}")
                except Exception as e:
                    logger.error(f"Failed to post error cast: {str(e)}")
                    # Don't re-raise for error posts to avoid infinite loops

            else:
                logger.debug(f"Ignoring emission type: {update_type}")

        except Exception as e:
            logger.error(f"Error handling Farcaster emission: {str(e)}", exc_info=True)
            raise
