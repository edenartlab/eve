import os
from typing import Optional
import aiohttp
import traceback
import logging
from ably import AblyRest

from eve.agent.session.models import SessionUpdateConfig

logger = logging.getLogger(__name__)


async def emit_update(update_config: Optional[SessionUpdateConfig], data: dict):
    if not update_config:
        return

    if update_config.update_endpoint:
        await emit_http_update(update_config, data)
    elif update_config.sub_channel_name:
        try:
            client = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
            channel = client.channels.get(update_config.sub_channel_name)
            await channel.publish(update_config.sub_channel_name, data)
        except Exception as e:
            logger.error(f"Failed to publish to Ably: {str(e)}")


async def emit_http_update(update_config: SessionUpdateConfig, data: dict):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{os.getenv('EDEN_API_URL')}/v2/deployments/emission",
                json={
                    "update_config": update_config.model_dump(),
                    "data": data,
                },
                headers={"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"},
            ) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to send update to endpoint: {await response.text()}"
                    )
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Error sending update to endpoint: {str(e)}")
