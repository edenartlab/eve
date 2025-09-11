"""
Typing state coordinator for managing typing indicators across platforms.
Ensures proper sequencing and prevents race conditions.
"""

import asyncio
import logging
import os
from typing import Dict, Optional
from ably import AblyRest, AblyRealtime
import time

logger = logging.getLogger(__name__)


class TypingCoordinator:
    """
    Coordinates typing state updates between API handlers and gateway.
    Ensures proper sequencing and prevents race conditions.
    """
    
    def __init__(self, db: str = None):
        self.db = db or os.getenv("DB", "STAGE").upper()
        self.ably_publisher = AblyRest(os.getenv("ABLY_PUBLISHER_KEY"))
        self.active_requests: Dict[str, float] = {}  # request_id -> timestamp
        
    async def update_busy_state(
        self,
        update_config: Optional[Dict],
        request_id: str,
        is_busy: bool
    ) -> bool:
        """
        Update busy state for a deployment with proper sequencing.
        Returns True if update was sent, False if skipped.
        """
        if not update_config:
            return False
        
        deployment_id = update_config.get("deployment_id")
        if not deployment_id:
            return False
        
        # Track request state
        current_time = time.time()
        
        logger.info(
            f"[TypingCoord] update_busy_state called - "
            f"deployment: {deployment_id}, request: {request_id}, busy: {is_busy}"
        )
        
        if is_busy:
            # Starting - record request
            self.active_requests[request_id] = current_time
            logger.info(f"[TypingCoord] Starting typing for request {request_id}")
        else:
            # Stopping - verify request exists
            if request_id not in self.active_requests:
                logger.warning(
                    f"[TypingCoord] Ignoring stop for unknown request {request_id}"
                )
                return False
            
            # Remove from active requests
            start_time = self.active_requests.pop(request_id)
            duration = current_time - start_time
            logger.info(
                f"[TypingCoord] Stopping typing for request {request_id} "
                f"(duration: {duration:.1f}s)"
            )
        
        # Determine platform and channel
        platform = self._get_platform(update_config)
        if not platform:
            logger.warning(f"[TypingCoord] Could not determine platform from config")
            return False
        
        # Prepare message based on platform
        message = self._prepare_message(platform, update_config, request_id, is_busy)
        if not message:
            return False
        
        # Send via Ably with request tracking
        channel_name = self._get_channel_name(platform, deployment_id)
        try:
            channel = self.ably_publisher.channels.get(channel_name)
            await channel.publish("typing", message)
            
            logger.info(
                f"[TypingCoord] Sent {platform} typing update - "
                f"request: {request_id}, busy: {is_busy}, "
                f"channel: {channel_name}"
            )
            return True
            
        except Exception as e:
            logger.error(
                f"[TypingCoord] Failed to send typing update - "
                f"request: {request_id}, error: {e}",
                exc_info=True
            )
            # Clean up request on error
            if is_busy and request_id in self.active_requests:
                del self.active_requests[request_id]
            return False
    
    def _get_platform(self, update_config: Dict) -> Optional[str]:
        """Determine platform from update config"""
        if "discord_channel_id" in update_config:
            return "discord"
        elif "telegram_chat_id" in update_config:
            return "telegram"
        elif "farcaster_hash" in update_config:
            return "farcaster"
        elif "twitter_tweet_to_reply_id" in update_config:
            return "twitter"
        return None
    
    def _prepare_message(
        self,
        platform: str,
        update_config: Dict,
        request_id: str,
        is_busy: bool
    ) -> Optional[Dict]:
        """Prepare platform-specific typing message"""
        
        message = {
            "request_id": request_id,
            "is_busy": is_busy,
            "timestamp": time.time()
        }
        
        if platform == "discord":
            channel_id = update_config.get("discord_channel_id")
            if not channel_id:
                return None
            message["channel_id"] = channel_id
            
        elif platform == "telegram":
            chat_id = update_config.get("telegram_chat_id")
            if not chat_id:
                return None
            message["deployment_id"] = update_config.get("deployment_id")
            message["chat_id"] = chat_id
            message["thread_id"] = update_config.get("telegram_thread_id")
            
        else:
            # Other platforms don't support typing
            return None
        
        return message
    
    def _get_channel_name(self, platform: str, deployment_id: str) -> str:
        """Get Ably channel name for platform"""
        if platform == "discord":
            return f"busy-state-discord-v2-{deployment_id}"
        elif platform == "telegram":
            return f"busy-state-telegram-{self.db}"
        else:
            return f"busy-state-{platform}-{deployment_id}"
    
    async def cleanup_stale_requests(self, timeout: float = 300):
        """Clean up stale active requests older than timeout"""
        current_time = time.time()
        stale_requests = [
            req_id for req_id, start_time in self.active_requests.items()
            if current_time - start_time > timeout
        ]
        
        for request_id in stale_requests:
            logger.warning(
                f"[TypingCoord] Removing stale request {request_id} "
                f"(age: {current_time - self.active_requests[request_id]:.1f}s)"
            )
            del self.active_requests[request_id]
        
        return len(stale_requests)
    
    def get_status(self) -> Dict:
        """Get coordinator status for debugging"""
        current_time = time.time()
        return {
            "active_requests": len(self.active_requests),
            "requests": [
                {
                    "request_id": req_id,
                    "duration": round(current_time - start_time, 1)
                }
                for req_id, start_time in self.active_requests.items()
            ]
        }


# Global instance
_typing_coordinator: Optional[TypingCoordinator] = None


def get_typing_coordinator() -> TypingCoordinator:
    """Get or create the global typing coordinator"""
    global _typing_coordinator
    if _typing_coordinator is None:
        _typing_coordinator = TypingCoordinator()
    return _typing_coordinator


async def update_busy_state(
    update_config: Optional[Dict],
    request_id: str,
    is_busy: bool
) -> bool:
    """
    Convenience function to update busy state.
    Used by API handlers.
    """
    coordinator = get_typing_coordinator()
    return await coordinator.update_busy_state(update_config, request_id, is_busy)