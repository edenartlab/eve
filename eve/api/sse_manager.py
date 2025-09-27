import asyncio
import logging
from typing import Dict, Set, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SSEConnection:
    """Represents an SSE connection for a session"""
    session_id: str
    queue: asyncio.Queue
    connected_at: datetime
    client_id: str  # Unique identifier for this connection


class SSEConnectionManager:
    """
    Manages Server-Sent Events connections for session streaming.
    Allows multiple clients to connect and stream updates for the same session.
    """
    
    _instance: Optional['SSEConnectionManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._connections: Dict[str, List[SSEConnection]] = {}  # session_id -> list of connections
            self._connection_count = 0
            self._lock = asyncio.Lock()
            self._initialized = True
            logger.info("SSEConnectionManager initialized")
    
    async def connect(self, session_id: str, client_id: str) -> SSEConnection:
        """
        Register a new SSE connection for a session.
        Returns the connection object with its queue.
        """
        async with self._lock:
            connection = SSEConnection(
                session_id=session_id,
                queue=asyncio.Queue(),
                connected_at=datetime.now(),
                client_id=client_id
            )
            
            if session_id not in self._connections:
                self._connections[session_id] = []
            
            self._connections[session_id].append(connection)
            self._connection_count += 1
            
            logger.info(
                f"SSE connection established for session {session_id} "
                f"(client: {client_id}, total connections: {self._connection_count})"
            )
            
            return connection
    
    async def disconnect(self, session_id: str, connection: SSEConnection):
        """Remove a connection when client disconnects"""
        async with self._lock:
            if session_id in self._connections:
                try:
                    self._connections[session_id].remove(connection)
                    
                    # Clean up empty session entries
                    if not self._connections[session_id]:
                        del self._connections[session_id]
                    
                    self._connection_count -= 1
                    
                    logger.info(
                        f"SSE connection closed for session {session_id} "
                        f"(client: {connection.client_id}, remaining connections: {self._connection_count})"
                    )
                except ValueError:
                    # Connection not in list, already removed
                    pass
    
    async def broadcast(self, session_id: str, data: dict):
        """
        Broadcast an update to all connections watching a session.
        Non-blocking - adds to queue for each connection.
        """
        if session_id not in self._connections:
            return
        
        # Format as SSE message using dumps_json to handle ObjectIds
        from eve.utils import dumps_json
        message = dumps_json(data)
        
        # Send to all connections for this session
        disconnected = []
        for connection in self._connections.get(session_id, []).copy():
            try:
                # Non-blocking put with timeout
                await asyncio.wait_for(
                    connection.queue.put(message),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Queue full for client {connection.client_id} on session {session_id}"
                )
            except Exception as e:
                logger.error(f"Error broadcasting to client {connection.client_id}: {e}")
                disconnected.append(connection)
        
        # Clean up any failed connections
        for conn in disconnected:
            await self.disconnect(session_id, conn)
    
    def get_connection_count(self, session_id: Optional[str] = None) -> int:
        """Get count of active connections, optionally for a specific session"""
        if session_id:
            return len(self._connections.get(session_id, []))
        return self._connection_count
    
    def get_active_sessions(self) -> Set[str]:
        """Get set of session IDs with active connections"""
        return set(self._connections.keys())
    
    async def close_all(self):
        """Close all connections (for shutdown)"""
        async with self._lock:
            for session_id in list(self._connections.keys()):
                for connection in self._connections[session_id]:
                    # Put a shutdown message in the queue
                    try:
                        await connection.queue.put('data: {"event": "shutdown"}\n\n')
                    except:
                        pass
            self._connections.clear()
            self._connection_count = 0
            logger.info("All SSE connections closed")


# Global singleton instance
sse_manager = SSEConnectionManager()