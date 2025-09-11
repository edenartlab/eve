"""
Improved typing management system for Discord and Telegram gateways.
Handles typing indicators with proper state tracking and cleanup.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TypingState(Enum):
    IDLE = "idle"
    TYPING = "typing"
    STOPPING = "stopping"


@dataclass
class TypingSession:
    """Tracks a single typing session for a channel/chat"""
    channel_id: str
    request_id: str
    state: TypingState
    task: Optional[asyncio.Task] = None
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    @property
    def duration(self) -> float:
        return time.time() - self.start_time
    
    @property
    def is_stale(self, timeout: float = 300) -> bool:
        """Check if session is stale (no updates for timeout seconds)"""
        return time.time() - self.last_update > timeout


class ImprovedTypingManager:
    """
    Manages typing indicators with proper state tracking and cleanup.
    Designed for single gateway instance with low concurrency.
    """
    
    def __init__(self, platform: str = "discord"):
        self.platform = platform
        self.sessions: Dict[str, TypingSession] = {}
        self.request_to_channel: Dict[str, str] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 30  # Check for stale sessions every 30s
        self.stale_timeout = 120  # Consider stale after 2 minutes
        self.typing_interval = 8 if platform == "discord" else 5
        
    def start_cleanup_loop(self):
        """Start background cleanup loop"""
        if not self.cleanup_task or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(f"[{self.platform}] Started typing cleanup loop")
    
    async def _cleanup_loop(self):
        """Periodically clean up stale typing sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_sessions()
            except asyncio.CancelledError:
                logger.info(f"[{self.platform}] Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"[{self.platform}] Cleanup loop error: {e}", exc_info=True)
    
    async def _cleanup_stale_sessions(self):
        """Remove stale typing sessions"""
        stale_channels = []
        for channel_id, session in self.sessions.items():
            if session.is_stale(self.stale_timeout):
                stale_channels.append(channel_id)
                logger.warning(
                    f"[{self.platform}] Stale session detected - "
                    f"channel: {channel_id}, request: {session.request_id}, "
                    f"duration: {session.duration:.1f}s"
                )
        
        for channel_id in stale_channels:
            await self.stop_typing(channel_id, reason="stale")
    
    async def start_typing(self, channel_id: str, request_id: str) -> bool:
        """
        Start typing indicator for a channel.
        Returns True if started, False if already typing.
        """
        # Check if already typing in this channel
        if channel_id in self.sessions:
            session = self.sessions[channel_id]
            
            # If same request, just update timestamp
            if session.request_id == request_id:
                session.last_update = time.time()
                logger.debug(
                    f"[{self.platform}] Updated typing keepalive - "
                    f"channel: {channel_id}, request: {request_id}"
                )
                return False
            
            # Different request - stop old, start new
            logger.info(
                f"[{self.platform}] Replacing typing session - "
                f"channel: {channel_id}, old_request: {session.request_id}, "
                f"new_request: {request_id}"
            )
            await self.stop_typing(channel_id, reason="replaced")
        
        # Create new typing session
        session = TypingSession(
            channel_id=channel_id,
            request_id=request_id,
            state=TypingState.TYPING
        )
        
        # Start typing loop
        session.task = asyncio.create_task(
            self._typing_loop(channel_id, request_id)
        )
        
        self.sessions[channel_id] = session
        self.request_to_channel[request_id] = channel_id
        
        logger.info(
            f"[{self.platform}] Started typing - "
            f"channel: {channel_id}, request: {request_id}"
        )
        
        # Ensure cleanup loop is running
        self.start_cleanup_loop()
        return True
    
    async def stop_typing(self, channel_id: str, reason: str = "completed", request_id: Optional[str] = None) -> bool:
        """
        Stop typing indicator for a channel.
        Returns True if stopped, False if not typing.
        """
        session = self.sessions.get(channel_id)
        if not session:
            logger.debug(
                f"[{self.platform}] No typing session to stop - "
                f"channel: {channel_id}, reason: {reason}"
            )
            return False
        
        # If request_id provided, verify it matches
        if request_id and session.request_id != request_id:
            logger.warning(
                f"[{self.platform}] Request mismatch on stop - "
                f"channel: {channel_id}, session_request: {session.request_id}, "
                f"stop_request: {request_id}, reason: {reason}"
            )
            # Don't stop if request doesn't match (unless it's a cleanup/stale stop)
            if reason not in ["stale", "cleanup", "replaced"]:
                return False
        
        # Update state
        session.state = TypingState.STOPPING
        
        # Cancel typing task
        if session.task and not session.task.done():
            session.task.cancel()
            try:
                await asyncio.wait_for(session.task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.error(
                    f"[{self.platform}] Error cancelling typing task - "
                    f"channel: {channel_id}, error: {e}"
                )
        
        # Remove from tracking
        del self.sessions[channel_id]
        if session.request_id in self.request_to_channel:
            del self.request_to_channel[session.request_id]
        
        logger.info(
            f"[{self.platform}] Stopped typing - "
            f"channel: {channel_id}, request: {session.request_id}, "
            f"reason: {reason}, duration: {session.duration:.1f}s"
        )
        return True
    
    async def stop_typing_by_request(self, request_id: str, reason: str = "completed") -> bool:
        """Stop typing for a specific request ID"""
        channel_id = self.request_to_channel.get(request_id)
        if not channel_id:
            logger.debug(
                f"[{self.platform}] No channel found for request: {request_id}"
            )
            return False
        
        return await self.stop_typing(channel_id, reason=reason, request_id=request_id)
    
    async def _typing_loop(self, channel_id: str, request_id: str):
        """Send typing indicators at regular intervals"""
        try:
            while True:
                # Verify session still exists and matches
                session = self.sessions.get(channel_id)
                if not session or session.request_id != request_id:
                    logger.info(
                        f"[{self.platform}] Typing loop exit - session gone or changed - "
                        f"channel: {channel_id}, request: {request_id}"
                    )
                    break
                
                # Update timestamp
                session.last_update = time.time()
                
                # Send typing indicator (implement platform-specific method)
                await self._send_typing_indicator(channel_id)
                
                # Wait for next interval
                await asyncio.sleep(self.typing_interval)
                
        except asyncio.CancelledError:
            logger.debug(
                f"[{self.platform}] Typing loop cancelled - "
                f"channel: {channel_id}, request: {request_id}"
            )
        except Exception as e:
            logger.error(
                f"[{self.platform}] Typing loop error - "
                f"channel: {channel_id}, request: {request_id}, error: {e}",
                exc_info=True
            )
        finally:
            # Ensure cleanup on exit
            if channel_id in self.sessions:
                session = self.sessions[channel_id]
                if session.request_id == request_id:
                    await self.stop_typing(channel_id, reason="loop_exit")
    
    async def _send_typing_indicator(self, channel_id: str):
        """Platform-specific typing indicator sending (to be overridden)"""
        raise NotImplementedError("Subclasses must implement _send_typing_indicator")
    
    async def cleanup(self):
        """Clean up all typing sessions and tasks"""
        logger.info(f"[{self.platform}] Cleaning up typing manager")
        
        # Cancel cleanup loop
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await asyncio.wait_for(self.cleanup_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Stop all active typing sessions
        channels = list(self.sessions.keys())
        for channel_id in channels:
            await self.stop_typing(channel_id, reason="cleanup")
        
        self.sessions.clear()
        self.request_to_channel.clear()
        logger.info(f"[{self.platform}] Typing manager cleanup complete")
    
    def get_status(self) -> Dict:
        """Get current typing manager status for debugging"""
        return {
            "platform": self.platform,
            "active_sessions": len(self.sessions),
            "sessions": [
                {
                    "channel_id": channel_id,
                    "request_id": session.request_id,
                    "state": session.state.value,
                    "duration": round(session.duration, 1),
                    "last_update": round(time.time() - session.last_update, 1),
                }
                for channel_id, session in self.sessions.items()
            ]
        }


class DiscordTypingManager(ImprovedTypingManager):
    """Discord-specific typing manager"""
    
    def __init__(self, client):
        super().__init__(platform="discord")
        self.client = client
        self.token = client.token
    
    async def _send_typing_indicator(self, channel_id: str):
        """Send typing indicator to Discord channel"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bot {self.token}",
                    "Content-Type": "application/json",
                    "User-Agent": "EveDiscordClient (https://github.com/edenartlab/eve, 1.0)",
                }
                url = f"https://discord.com/api/v10/channels/{channel_id}/typing"
                
                async with session.post(url, headers=headers) as response:
                    if response.status == 204:
                        logger.debug(f"[discord] Sent typing to channel {channel_id}")
                    elif response.status == 429:
                        retry_after = (await response.json()).get("retry_after", 1.0)
                        logger.warning(
                            f"[discord] Rate limited on channel {channel_id}, "
                            f"retry after {retry_after}s"
                        )
                        await asyncio.sleep(retry_after)
                    elif response.status in [401, 403, 404]:
                        logger.error(
                            f"[discord] Permission/existence error for channel {channel_id}: "
                            f"{response.status}"
                        )
                        # Stop typing for this channel
                        await self.stop_typing(channel_id, reason="error")
                    else:
                        text = await response.text()
                        logger.warning(
                            f"[discord] Failed to send typing to {channel_id}: "
                            f"{response.status} - {text}"
                        )
        except Exception as e:
            logger.error(
                f"[discord] Error sending typing to {channel_id}: {e}",
                exc_info=True
            )


class TelegramTypingManager(ImprovedTypingManager):
    """Telegram-specific typing manager"""
    
    def __init__(self):
        super().__init__(platform="telegram")
        self.tokens: Dict[str, str] = {}  # deployment_id -> token
    
    def register_deployment(self, deployment_id: str, token: str):
        """Register a Telegram bot token"""
        self.tokens[deployment_id] = token
        logger.info(f"[telegram] Registered deployment {deployment_id}")
    
    def unregister_deployment(self, deployment_id: str):
        """Unregister a Telegram deployment"""
        if deployment_id in self.tokens:
            del self.tokens[deployment_id]
            logger.info(f"[telegram] Unregistered deployment {deployment_id}")
    
    async def start_typing_with_deployment(
        self, deployment_id: str, chat_id: str, request_id: str, thread_id: Optional[int] = None
    ):
        """Start typing with deployment context"""
        if deployment_id not in self.tokens:
            logger.warning(f"[telegram] No token for deployment {deployment_id}")
            return False
        
        # Store deployment_id in channel_id for lookup
        channel_key = f"{deployment_id}:{chat_id}:{thread_id or 'main'}"
        return await self.start_typing(channel_key, request_id)
    
    async def _send_typing_indicator(self, channel_key: str):
        """Send typing indicator to Telegram chat"""
        import aiohttp
        
        try:
            # Parse channel key
            parts = channel_key.split(":")
            deployment_id = parts[0]
            chat_id = parts[1]
            thread_id = None if parts[2] == "main" else int(parts[2])
            
            token = self.tokens.get(deployment_id)
            if not token:
                logger.warning(f"[telegram] No token for deployment {deployment_id}")
                return
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.telegram.org/bot{token}/sendChatAction"
                payload = {"chat_id": chat_id, "action": "typing"}
                if thread_id:
                    payload["message_thread_id"] = thread_id
                
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.warning(
                            f"[telegram] Failed to send typing to {chat_id}: {text}"
                        )
        except Exception as e:
            logger.error(
                f"[telegram] Error sending typing to {channel_key}: {e}",
                exc_info=True
            )