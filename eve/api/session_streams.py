import asyncio
import json
import logging
import time
from typing import Dict, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StreamClient:
    """Represents a connected SSE client"""
    client_id: str
    queue: asyncio.Queue
    connected_at: float
    user_id: Optional[str] = None
    
    def __hash__(self):
        return hash(self.client_id)
    
    def __eq__(self, other):
        if isinstance(other, StreamClient):
            return self.client_id == other.client_id
        return False


class SessionStreamManager:
    """Manages SSE connections for session streaming"""
    
    def __init__(self):
        # session_id -> Set[StreamClient]
        self._sessions: Dict[str, Set[StreamClient]] = defaultdict(set)
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        # Track active sessions to avoid memory leaks
        self._active_sessions: Set[str] = set()
        
    async def add_client(self, session_id: str, client_id: str, user_id: Optional[str] = None) -> StreamClient:
        """Add a new client to a session stream"""
        async with self._lock:
            client = StreamClient(
                client_id=client_id,
                queue=asyncio.Queue(maxsize=1000),  # Buffer up to 1000 events
                connected_at=time.time(),
                user_id=user_id
            )
            self._sessions[session_id].add(client)
            self._active_sessions.add(session_id)
            logger.info(f"Added client {client_id} to session {session_id}. Total clients: {len(self._sessions[session_id])}")
            return client
    
    async def remove_client(self, session_id: str, client: StreamClient):
        """Remove a client from a session stream"""
        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].discard(client)
                logger.info(f"Removed client {client.client_id} from session {session_id}. Remaining clients: {len(self._sessions[session_id])}")
                
                # Clean up empty sessions
                if not self._sessions[session_id]:
                    del self._sessions[session_id]
                    self._active_sessions.discard(session_id)
                    logger.info(f"No more clients for session {session_id}, cleaned up")
    
    async def broadcast(self, session_id: str, event_type: str, data: dict):
        """Broadcast an event to all clients of a session"""
        if session_id not in self._sessions:
            return
        
        event = {
            "event": event_type,
            "data": data,
            "timestamp": time.time()
        }
        event_str = f"data: {json.dumps(event)}\n\n"
        
        # Get clients snapshot to avoid holding lock during queue operations
        async with self._lock:
            clients = list(self._sessions.get(session_id, []))
        
        # Send to all clients
        disconnected = []
        for client in clients:
            try:
                # Non-blocking put with timeout
                await asyncio.wait_for(
                    client.queue.put(event_str),
                    timeout=0.1  # 100ms timeout for slow clients
                )
            except asyncio.TimeoutError:
                logger.warning(f"Client {client.client_id} queue full, dropping event")
                # Could implement backpressure strategies here
            except Exception as e:
                logger.error(f"Error sending to client {client.client_id}: {e}")
                disconnected.append(client)
        
        # Clean up disconnected clients
        for client in disconnected:
            await self.remove_client(session_id, client)
    
    async def get_active_sessions(self) -> Set[str]:
        """Get set of sessions with active clients"""
        async with self._lock:
            return self._active_sessions.copy()
    
    async def get_client_count(self, session_id: str) -> int:
        """Get number of clients connected to a session"""
        async with self._lock:
            return len(self._sessions.get(session_id, []))
    
    async def cleanup_stale_clients(self, max_age_seconds: int = 3600):
        """Remove clients that have been connected for too long"""
        current_time = time.time()
        async with self._lock:
            for session_id, clients in list(self._sessions.items()):
                stale_clients = [
                    client for client in clients 
                    if current_time - client.connected_at > max_age_seconds
                ]
                for client in stale_clients:
                    await self.remove_client(session_id, client)


# Global instance
session_stream_manager = SessionStreamManager()