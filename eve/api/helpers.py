import os
import subprocess
import time
import traceback
from typing import Optional

import aiohttp
import modal
from ably import AblyRest
from loguru import logger

from eve.agent import Agent
from eve.agent.deployments import PlatformClient
from eve.agent.session.models import ClientType, Deployment
from eve.api.api_requests import UpdateConfig

db = os.getenv("DB", "STAGE").upper()
busy_state_dict = modal.Dict.from_name(
    f"busy-state-store-{db.lower()}", create_if_missing=True
)


async def get_update_channel(
    update_config: UpdateConfig, ably_client: AblyRest
) -> Optional[AblyRest]:
    return ably_client.channels.get(str(update_config.sub_channel_name))


async def emit_update(
    update_config: Optional[UpdateConfig], data: dict, session_id: Optional[str] = None
):
    """
    Emit updates through configured channels (HTTP, Ably) and also broadcast to SSE connections.

    Args:
        update_config: Configuration for update channels
        data: The update data to emit
        session_id: Optional session ID for SSE broadcasting
    """
    if not update_config and not session_id:
        return

    from eve.agent.session.debug_logger import SessionDebugger

    # Emit through configured channels (HTTP or Ably)
    if update_config:
        debugger = SessionDebugger(session_id)
        if update_config.update_endpoint:
            debugger.log(
                "HTTP emit",
                {"endpoint": update_config.update_endpoint.split("/")[-1]},
                emoji="update",
            )
            await emit_http_update(update_config, data)
        elif update_config.sub_channel_name:
            try:
                debugger.log(
                    "Ably emit",
                    {"channel": update_config.sub_channel_name},
                    emoji="update",
                )
                client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
                channel = client.channels.get(update_config.sub_channel_name)
                await channel.publish(update_config.sub_channel_name, data)
            except Exception as e:
                debugger.log_error("Ably failed", e)
                logger.error(f"Failed to publish to Ably: {str(e)}")

    # Also broadcast to SSE connections if session_id is provided
    if session_id:
        try:
            from eve.api.sse_manager import sse_manager

            connection_count = sse_manager.get_connection_count(session_id)

            if connection_count > 0:
                debugger = SessionDebugger(session_id)
                debugger.log(
                    "SSE broadcast",
                    {"connections": connection_count, "type": data.get("type", "?")},
                    emoji="broadcast",
                )
                await sse_manager.broadcast(session_id, data)
        except Exception as e:
            debugger = SessionDebugger(session_id)
            debugger.log_error("SSE failed", e)
            logger.error(f"Failed to broadcast to SSE connections: {str(e)}")


async def emit_http_update(update_config: UpdateConfig, data: dict):
    from eve.utils.data_utils import serialize_json

    async with aiohttp.ClientSession() as session:
        try:
            # Serialize ObjectIds and other non-JSON types
            serialized_data = serialize_json(data)

            async with session.post(
                update_config.update_endpoint,
                json=serialized_data,
                headers={"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"},
            ) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to send update to endpoint: {await response.text()}"
                    )
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Error sending update to endpoint: {str(e)}")


def get_eden_creation_url(creation_id: str):
    root_url = "app.eden.art" if db == "PROD" else "staging.app.eden.art"
    return f"https://{root_url}/creations/{creation_id}"


def get_ably_client() -> Optional[AblyRest]:
    """Initializes and returns an AblyRest client."""
    api_key = os.getenv("ABLY_PUBLISHER_KEY")
    if not api_key:
        logger.error("ABLY_PUBLISHER_KEY not found in environment.")
        return None
    try:
        # Use AblyRest for publishing
        return AblyRest(api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Ably client: {e}")
        return None


async def publish_busy_state(key: str, is_busy: bool, context: dict):
    """Publishes the busy state to the appropriate Ably channel."""
    ably = get_ably_client()
    if not ably:
        logger.warning(f"Ably client not available, cannot publish state for {key}")
        return

    try:
        # Extract platform/IDs from key (e.g., "deployment_id.platform")
        if "." not in key:
            logger.error(f"Invalid key format for publishing busy state: {key}")
            return

        deployment_id, platform = key.split(".", 1)
        # Construct the channel name the gateway/client subscribes to
        channel_name = f"busy-state-{platform}-{deployment_id}"
        channel = ably.channels.get(channel_name)

        payload = {"is_busy": is_busy}
        platform_context_key = None
        platform_context_value = None

        # Add relevant context for the platform listener
        if platform == "discord":
            channel_id = context.get("discord_channel_id")
            if channel_id:
                payload["channel_id"] = str(channel_id)  # Ensure string
                platform_context_key = "channel_id"
                platform_context_value = str(channel_id)
            else:
                # If no channel_id, the gateway listener might ignore the stop signal
                logger.warning(
                    f"Missing discord_channel_id in context for key {key} - stop signal might be ignored by gateway."
                )
                # Send anyway, maybe the gateway has logic for deployment-level stop?

        elif platform == "telegram":
            chat_id = context.get("telegram_chat_id")
            thread_id = context.get("telegram_thread_id")
            if chat_id:
                payload["chat_id"] = str(chat_id)  # Ensure string
                platform_context_key = "chat_id"
                platform_context_value = str(chat_id)
                if thread_id:
                    payload["thread_id"] = str(thread_id)  # Ensure string
                    # Use chat_id_thread_id for more specific logging if needed
                    platform_context_value = f"{chat_id}_{thread_id}"

            else:
                logger.warning(f"Missing telegram_chat_id in context for key {key}")

        # Add other platforms if needed

        log_context = (
            f" for {platform} context {platform_context_key}={platform_context_value}"
            if platform_context_key
            else ""
        )
        logger.info(
            f"Publishing to Ably channel '{channel_name}': {payload}{log_context}"
        )
        await channel.publish("update", payload)

    except Exception as e:
        logger.error(
            f"Failed to publish busy state to Ably for key {key}: {e}", exc_info=True
        )


async def update_busy_state(update_config, request_id: str, is_busy: bool):
    """Update busy state using the new typing coordinator."""
    logger.info(
        f"[Helpers] update_busy_state called - request: {request_id}, busy: {is_busy}"
    )

    from eve.api.typing_coordinator import update_busy_state as coord_update_busy_state

    # Convert to dict if it's a Pydantic model
    config_dict = None
    if hasattr(update_config, "model_dump"):
        config_dict = update_config.model_dump(exclude_unset=True)
    elif isinstance(update_config, dict):
        config_dict = update_config

    logger.info(f"[Helpers] config_dict: {config_dict}")
    return await coord_update_busy_state(config_dict, request_id, is_busy)


async def update_busy_state_old(update_config, request_id: str, is_busy: bool):
    """Updates the busy state in modal.Dict and publishes to Ably."""
    if not update_config:
        logger.warning("Cannot update busy state: update_config is missing.")
        return

    # Handle Pydantic model or dict
    if hasattr(update_config, "model_dump"):
        config_dict = update_config.model_dump(exclude_unset=True)

    elif isinstance(update_config, dict):
        config_dict = update_config
    else:
        logger.warning(
            f"Cannot update busy state: Invalid update_config type {type(update_config)}."
        )
        return

    deployment_id = config_dict.get("deployment_id")
    platform = None
    context = {}  # Store channel/chat IDs for this specific update

    # Determine platform and context from the update_config
    discord_channel_id = config_dict.get("discord_channel_id")
    telegram_chat_id = config_dict.get("telegram_chat_id")

    if discord_channel_id:
        platform = "discord"
        context["discord_channel_id"] = discord_channel_id
    elif telegram_chat_id:
        platform = "telegram"
        context["telegram_chat_id"] = telegram_chat_id
        if config_dict.get("telegram_thread_id"):
            context["telegram_thread_id"] = config_dict["telegram_thread_id"]
    # Add other platforms based on their unique identifiers in update_config

    if not deployment_id or not platform:
        logger.warning(
            f"Cannot determine deployment_id or platform for busy state. Config: {config_dict}, Request ID: {request_id}"
        )
        return

    key = f"{deployment_id}.{platform}"
    logger.info(
        f"Updating busy state for key '{key}', request_id '{request_id}', is_busy={is_busy}, context={context}"
    )

    try:
        # Get current state
        current_state = busy_state_dict.get(key, {})
        if not isinstance(current_state, dict) or not all(
            k in current_state for k in ["requests", "timestamps", "context_map"]
        ):
            logger.warning(
                f"Invalid state found for key {key}, reinitializing. State: {current_state}"
            )
            current_state = {"requests": [], "timestamps": {}, "context_map": {}}

        requests = current_state.get("requests", [])
        timestamps = current_state.get("timestamps", {})
        context_map = current_state.get("context_map", {})

        # Make copies to modify, ensure correct types
        requests = list(requests)
        timestamps = dict(timestamps)
        context_map = dict(context_map)

        was_overall_busy = len(requests) > 0

        # --- Update state based on is_busy ---
        if is_busy:
            if request_id not in requests:
                requests.append(request_id)
            timestamps[request_id] = time.time()
            # Store context associated with this specific request_id
            context_map[request_id] = context
        else:
            # Request finished, remove it
            if request_id in requests:
                requests.remove(request_id)
            timestamps.pop(request_id, None)
            # Remove context for this finished request
            context_map.pop(request_id, None)

        # --- Persist updated state ---
        new_state = {
            "requests": requests,
            "timestamps": timestamps,
            "context_map": context_map,
        }
        busy_state_dict.put(key, new_state)
        # Log the updated state for debugging
        logger.debug(f"Persisted state for key '{key}': {new_state}")

        # --- Publish state change to Ably ---
        now_is_overall_busy = len(requests) > 0

        # Determine if Ably publish is needed based on the *specific context* of this update
        publish_needed = False
        if is_busy:
            # Publish 'start typing' for this context.
            publish_needed = True
            await publish_busy_state(key, True, context)
        else:
            # Publish 'stop typing' for this context.
            publish_needed = True
            await publish_busy_state(key, False, context)
            # Check if this was the *last* request for this key. If so, we could potentially
            # publish an overall stop, but relying on individual context stops is safer.

        if publish_needed:
            logger.info(
                f"State transition for key '{key}': was_overall_busy={was_overall_busy}, now_is_overall_busy={now_is_overall_busy}. Published is_busy={is_busy} for context {context}."
            )
        else:
            logger.info(
                f"No state change detected requiring Ably publish for key '{key}', context {context}."
            )

    except Exception as e:
        logger.error(
            f"Error updating busy state for key {key}, request_id {request_id}: {e}",
            exc_info=True,
        )


def get_platform_client(
    agent: Agent, platform: ClientType, deployment: Optional[Deployment] = None
) -> PlatformClient:
    """Helper function to get the appropriate platform client"""
    from eve.agent.deployments.captions import CaptionsClient
    from eve.agent.deployments.discord import DiscordClient
    from eve.agent.deployments.discord_v3 import DiscordV3Client
    from eve.agent.deployments.email import EmailClient
    from eve.agent.deployments.farcaster import FarcasterClient
    from eve.agent.deployments.gmail import GmailClient
    from eve.agent.deployments.google_calendar import GoogleCalendarClient
    from eve.agent.deployments.instagram import InstagramClient
    from eve.agent.deployments.printify import PrintifyClient
    from eve.agent.deployments.shopify import ShopifyClient
    from eve.agent.deployments.telegram import TelegramClient
    from eve.agent.deployments.tiktok import TiktokClient
    from eve.agent.deployments.twitter import TwitterClient

    if platform == ClientType.DISCORD:
        return DiscordClient(agent=agent, deployment=deployment)
    elif platform == ClientType.DISCORD_V3:
        return DiscordV3Client(agent=agent, deployment=deployment)
    elif platform == ClientType.TELEGRAM:
        return TelegramClient(agent=agent, deployment=deployment)
    elif platform == ClientType.FARCASTER:
        return FarcasterClient(agent=agent, deployment=deployment)
    elif platform == ClientType.TWITTER:
        return TwitterClient(agent=agent, deployment=deployment)
    elif platform == ClientType.SHOPIFY:
        return ShopifyClient(agent=agent, deployment=deployment)
    elif platform == ClientType.PRINTIFY:
        return PrintifyClient(agent=agent, deployment=deployment)
    elif platform == ClientType.CAPTIONS:
        return CaptionsClient(agent=agent, deployment=deployment)
    elif platform == ClientType.TIKTOK:
        return TiktokClient(agent=agent, deployment=deployment)
    elif platform == ClientType.INSTAGRAM:
        return InstagramClient(agent=agent, deployment=deployment)
    elif platform == ClientType.EMAIL:
        return EmailClient(agent=agent, deployment=deployment)
    elif platform == ClientType.GMAIL:
        return GmailClient(agent=agent, deployment=deployment)
    elif platform == ClientType.GOOGLE_CALENDAR:
        return GoogleCalendarClient(agent=agent, deployment=deployment)
    else:
        raise ValueError(f"Unknown platform: {platform}")


def authenticate_modal_key() -> bool:
    token_id = os.getenv("MODAL_DEPLOYER_TOKEN_ID")
    token_secret = os.getenv("MODAL_DEPLOYER_TOKEN_SECRET")
    subprocess.run(
        [
            "modal",
            "token",
            "set",
            "--token-id",
            token_id,
            "--token-secret",
            token_secret,
        ],
        capture_output=True,
        text=True,
    )


def check_environment_exists(env_name: str) -> bool:
    result = subprocess.run(
        ["modal", "environment", "list"], capture_output=True, text=True
    )
    return f"│ {env_name} " in result.stdout


def create_environment(env_name: str):
    subprocess.run(["modal", "environment", "create", env_name])


def pre_modal_setup():
    if not authenticate_modal_key():
        logger.warning("Skipping Modal environment setup due to missing credentials")
        return
    from eve import DEPLOYMENT_ENV_NAME

    TRIGGER_ENV_NAME = "triggers"
    if not check_environment_exists(DEPLOYMENT_ENV_NAME):
        create_environment(DEPLOYMENT_ENV_NAME)
    if not check_environment_exists(TRIGGER_ENV_NAME):
        create_environment(TRIGGER_ENV_NAME)
