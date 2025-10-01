"""
Debug logging utility for session operations.
Can be toggled on/off via environment variable or runtime configuration.
"""
import os
import logging
import functools
from typing import Any, Optional
import json
from datetime import datetime
import sys

# Environment variable to control debug logging
DEBUG_SESSION = os.getenv("DEBUG_SESSION", "false").lower() == "true"

# Create a dedicated logger for session debugging
session_debug_logger = logging.getLogger("eve.session.debug")
session_debug_logger.setLevel(logging.DEBUG if DEBUG_SESSION else logging.WARNING)
session_debug_logger.propagate = False  # Prevent duplicate logging

# Only add handler if debug is enabled and no handlers exist
if DEBUG_SESSION and not session_debug_logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    # Simple format without duplicate info
    debug_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(debug_formatter)
    session_debug_logger.addHandler(console_handler)


class SessionDebugger:
    """Context-aware session debugger that can be toggled on/off"""
    
    # Colored circle indicators for different log types
    EMOJI = {
        'start': '🟢',      # Green circle for starting
        'end': '🔴',        # Red circle for ending
        'info': '🔵',       # Blue circle for info
        'warning': '🟡',    # Yellow circle for warning
        'error': '🔴',      # Red circle for error
        'debug': '⚪',      # White circle for debug
        'update': '🟣',     # Purple circle for updates
        'broadcast': '🟠',  # Orange circle for broadcasting
        'success': '🟢',    # Green circle for success
        'actor': '🔵',      # Blue circle for actors
        'message': '⚪',    # White circle for messages
        'llm': '🟣',        # Purple circle for LLM
        'tool': '🟠',       # Orange circle for tools
        'connection': '🟢', # Green circle for connections
        'data': '⚫',       # Black circle for data
    }
    
    def __init__(self, session_id: Optional[str] = None, enabled: Optional[bool] = None):
        self.session_id = session_id
        self.enabled = enabled if enabled is not None else DEBUG_SESSION
        self._indent_level = 0
    
    def _format_message(self, message: str, data: Optional[Any] = None, emoji: str = 'debug') -> str:
        """Format a debug message with session context and emoji"""
        indent = "  " * self._indent_level
        session_info = f"[{self.session_id[:8]}]" if self.session_id else "[NoSession]"
        emoji_icon = self.EMOJI.get(emoji, self.EMOJI['debug'])
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        formatted = f"{indent}{emoji_icon} {timestamp} {session_info} {message}"
        
        if data is not None:
            try:
                if isinstance(data, dict):
                    # Compact formatting for common fields
                    if len(data) <= 3 and all(isinstance(v, (str, int, bool, type(None))) for v in data.values()):
                        # Inline simple dictionaries
                        data_str = ' | ' + ', '.join(f"{k}={v}" for k, v in data.items() if v is not None)
                        formatted += data_str
                    else:
                        # Multi-line for complex data
                        data_str = json.dumps(data, indent=2, default=str)
                        data_lines = data_str.split('\n')
                        data_formatted = '\n'.join(f"{indent}    {line}" for line in data_lines)
                        formatted += f"\n{data_formatted}"
                else:
                    formatted += f" | {data}"
            except Exception:
                formatted += f" | {str(data)}"
        
        return formatted
    
    def log(self, message: str, data: Optional[Any] = None, level: str = "debug", emoji: Optional[str] = None):
        """Log a debug message if enabled"""
        if not self.enabled:
            return
        
        # Auto-select emoji based on level if not specified
        if emoji is None:
            emoji = level if level in self.EMOJI else 'debug'
        
        formatted = self._format_message(message, data, emoji)
        
        # Always use debug level for actual logging to avoid duplicate timestamps
        session_debug_logger.debug(formatted)
    
    def start_section(self, section_name: str):
        """Start a new debug section with increased indentation"""
        if not self.enabled:
            return
        
        self.log(f"Starting: {section_name}", emoji='start')
        self._indent_level += 1
    
    def end_section(self, section_name: str):
        """End a debug section with decreased indentation"""
        if not self.enabled:
            return
        
        self._indent_level = max(0, self._indent_level - 1)
        self.log(f"Completed: {section_name}", emoji='end')
    
    def log_error(self, message: str, error: Exception):
        """Log an error with exception details"""
        if not self.enabled:
            return
        
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": None  # Could add traceback if needed
        }
        self.log(message, error_data, emoji='error')
    
    def log_update(self, update_type: str, update_data: Optional[dict] = None):
        """Log a session update event"""
        if not self.enabled:
            return
        
        self.log(f"Update: {update_type}", update_data, emoji='update')
    
    def log_sse_broadcast(self, session_id: str, data: dict, connection_count: int = 0):
        """Log SSE broadcast event"""
        if not self.enabled:
            return
        
        broadcast_info = {
            "session_id": session_id[:8] if session_id else "unknown",
            "connections": connection_count,
            "data_type": data.get("type", "unknown"),
            "data_size": len(json.dumps(data, default=str))
        }
        self.log(f"SSE Broadcast", broadcast_info, emoji='broadcast')


def debug_session_method(func):
    """Decorator to add debug logging to session methods"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Try to extract session_id from arguments
        session_id = None
        if args and hasattr(args[0], 'session'):
            session = args[0].session
            session_id = str(session.id) if hasattr(session, 'id') else None
        elif 'context' in kwargs and hasattr(kwargs['context'], 'session'):
            session = kwargs['context'].session
            session_id = str(session.id) if hasattr(session, 'id') else None
        
        debugger = SessionDebugger(session_id)
        
        if debugger.enabled:
            func_name = func.__name__
            debugger.log(f"🎯 Calling {func_name}", {
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })
        
        try:
            result = await func(*args, **kwargs)
            if debugger.enabled:
                debugger.log(f"✅ {func.__name__} completed successfully")
            return result
        except Exception as e:
            if debugger.enabled:
                debugger.log_error(f"{func.__name__} failed", e)
            raise
    
    return wrapper


# Global debug toggle functions
def enable_session_debug():
    """Enable session debug logging at runtime"""
    global DEBUG_SESSION
    DEBUG_SESSION = True
    session_debug_logger.setLevel(logging.DEBUG)
    print("🔍 Session debug logging ENABLED")


def disable_session_debug():
    """Disable session debug logging at runtime"""
    global DEBUG_SESSION
    DEBUG_SESSION = False
    session_debug_logger.setLevel(logging.WARNING)
    print("🔕 Session debug logging DISABLED")


def is_debug_enabled() -> bool:
    """Check if debug logging is currently enabled"""
    return DEBUG_SESSION