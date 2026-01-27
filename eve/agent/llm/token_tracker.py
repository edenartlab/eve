"""
Token Tracker for LLM Context Analysis

This module provides a simple way to track token usage across different
components of the LLM input context. Results are logged to CSV files
on a Modal Volume for later analysis.

IMPORTANT: All methods in this module are designed to fail gracefully
and will never raise exceptions that could crash the main code loop.

Output file:
- token_usage_{DB}.csv: Token breakdown by category with content (for analysis and debugging)

The session_run_id is also stored in the database in ChatMessage.observability.session_run_id
for the assistant message output. Files are suffixed with DB env (STAGE/PROD).

Usage:
    from eve.agent.llm.token_tracker import render_template_with_token_tracking, token_tracker

    # Register a new LLM call - returns a tracker handle
    tracker = token_tracker.register_call(
        agent_id=agent.id,
        agent_name=agent.username,
        session_id=session.id,
        user_id=user.id,
        session_run_id=context.session_run_id,
    )

    # Track system message components via template rendering
    content = render_template_with_token_tracking(
        system_template,
        session_run_id=context.session_run_id,
        prefix="system",
        name=actor.name,
        memory=memory,
        persona=actor.persona,
        # ... all template variables are tracked as system/{key}
    )

    # Track tools and messages separately
    tracker.track_context(
        agent=actor,
        session=session,
        tools=tools,
        messages=messages,
    )

    # Finish tracking
    tracker.finish(full_prompt=final_prompt_string)
"""

import csv
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Modal Volume mount path for token tracking logs
TOKEN_TRACKER_MODAL_DIR = "/data/token-tracker"
TOKEN_TRACKER_LOCAL_DIR = Path(__file__).parent.parent.parent / "data" / "token-tracker"

# DB environment suffix for separating stage/prod data
db = os.getenv("DB", "STAGE").upper()
TOKEN_USAGE_FILE = f"token_usage_{db}.csv"

# Auto-expire tracking data after this many seconds (prevents memory leaks)
TRACKING_EXPIRY_SECONDS = 600  # 10 minutes


def _safe_str(value: Any) -> Optional[str]:
    """Safely convert value to string, returns None on failure."""
    try:
        if value is None:
            return None
        return str(value)
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4.5 characters per token)."""
    try:
        if not text:
            return 0
        return int(len(text) / 4.5)
    except Exception:
        return 0


def render_template_with_token_tracking(
    template,
    session_run_id: Optional[str] = None,
    prefix: str = "system",
    **kwargs
) -> str:
    """
    Render a Jinja template while tracking each component's tokens.

    This function wraps template.render() and automatically tracks each
    kwarg as a separate token category. This allows granular token analysis
    without polluting the calling code with tracking logic.

    Args:
        template: Jinja2 Template object
        session_run_id: The session run ID for token tracking (optional)
        prefix: Category prefix for tracking (e.g., "system" or "agent_session")
        **kwargs: All arguments to pass to template.render()

    Returns:
        The rendered template string

    Example:
        content = render_template_with_token_tracking(
            system_template,
            session_run_id=context.session_run_id,
            prefix="system",
            name=actor.name,
            memory=memory,
            persona=actor.persona,
        )
        # Tracks: system/name, system/memory, system/persona, etc.
    """
    import json

    # Track each component if session_run_id is provided
    if session_run_id:
        try:
            for key, value in kwargs.items():
                if value is not None:
                    # Convert to string for tracking
                    if isinstance(value, str):
                        content = value
                    elif isinstance(value, (list, dict)):
                        content = json.dumps(value, default=str)
                    else:
                        content = str(value)

                    if content:  # Only track non-empty content
                        token_tracker._add_chunk(session_run_id, f"{prefix}/{key}", content)
        except Exception:
            # Never let tracking errors affect the render
            pass

    # Always render the template
    return template.render(**kwargs)


def normalize_for_comparison(text: str) -> str:
    """Normalize text for fuzzy comparison by collapsing whitespace."""
    try:
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.strip().lower())
    except Exception:
        return ""


def extract_untracked_segments(
    full_prompt: str,
    tracked_chunks: List[str],
    min_segment_length: int = 50
) -> List[Tuple[str, int]]:
    """
    Find segments in the full prompt that weren't tracked.

    Args:
        full_prompt: The complete prompt sent to the LLM
        tracked_chunks: List of chunk contents that were tracked
        min_segment_length: Minimum character length for an untracked segment

    Returns:
        List of (segment_text, token_count) tuples for untracked content
    """
    try:
        if not full_prompt:
            return []

        norm_prompt = normalize_for_comparison(full_prompt)
        covered = [False] * len(norm_prompt)

        for chunk in tracked_chunks:
            if not chunk:
                continue
            norm_chunk = normalize_for_comparison(chunk)
            if not norm_chunk:
                continue

            start = 0
            while True:
                idx = norm_prompt.find(norm_chunk, start)
                if idx == -1:
                    break
                for i in range(idx, min(idx + len(norm_chunk), len(covered))):
                    covered[i] = True
                start = idx + 1

        untracked = []
        current_start = None

        for i, is_covered in enumerate(covered):
            if not is_covered and current_start is None:
                current_start = i
            elif is_covered and current_start is not None:
                segment = norm_prompt[current_start:i].strip()
                if len(segment) >= min_segment_length:
                    untracked.append((segment, estimate_tokens(segment)))
                current_start = None

        if current_start is not None:
            segment = norm_prompt[current_start:].strip()
            if len(segment) >= min_segment_length:
                untracked.append((segment, estimate_tokens(segment)))

        return untracked

    except Exception:
        return []


class TrackerHandle:
    """
    A handle for tracking a single LLM call.

    This handle is returned by TokenTracker.register_call() and should be
    used for all subsequent tracking operations for that specific call.

    All methods are safe and will never raise exceptions.
    """

    def __init__(
        self,
        tracker: "TokenTracker",
        session_run_id: str,
        agent_id: Optional[str],
        session_id: Optional[str],
    ):
        self._tracker = tracker
        self._session_run_id = session_run_id
        self._agent_id = agent_id
        self._session_id = session_id
        self._finished = False

    def _verify_identity(self, agent_id: Any = None, session_id: Any = None) -> bool:
        """Verify that provided IDs match this handle's IDs."""
        try:
            if agent_id is not None:
                if _safe_str(agent_id) != self._agent_id:
                    return False
            if session_id is not None:
                if _safe_str(session_id) != self._session_id:
                    return False
            return True
        except Exception:
            return True  # On error, allow operation to proceed

    def add_chunk(self, category: str, text: str) -> int:
        """
        Add a prompt chunk with its category.

        Args:
            category: Hierarchical category name (e.g., "system/persona")
            text: The actual text content

        Returns:
            Estimated token count for this chunk (0 on error)
        """
        try:
            if self._finished:
                return 0
            return self._tracker._add_chunk(self._session_run_id, category, text)
        except Exception:
            return 0

    def set_model(self, model: str) -> None:
        """Update the model name for this call."""
        try:
            if self._finished:
                return
            self._tracker._set_model(self._session_run_id, model)
        except Exception:
            pass

    def track_context(
        self,
        agent: Any = None,
        session: Any = None,
        user: Any = None,
        tools: Dict = None,
        messages: List = None,
        trigger_context: Dict = None,
        prefix: str = "system",
        **extras,
    ) -> None:
        """
        Extract and track token usage from context objects.

        Tracks:
        - tool_schemas/*: Each tool's schema (sent as separate API param)
        - messages: Combined conversation history (excluding tool results)
        - messages/tool_results: Tool call results from conversation history

        Note: System message components (memory, persona, etc.) are tracked
        separately via render_template_with_token_tracking() in build_system_message.

        Args:
            agent: Agent object (for future use)
            session: Session object (for future use)
            user: User object (for future use)
            tools: Dict of Tool objects (extracts schemas)
            messages: List of ChatMessage objects (tracks by role)
            trigger_context: Dict with trigger info (for future use)
            prefix: Category prefix ("system" or "agent_session")
            **extras: Any additional string values to track
        """
        try:
            if self._finished:
                return

            # Tool schemas (sent as separate API parameter, but counts as input tokens)
            if tools:
                for tool_name, tool in tools.items():
                    try:
                        schema = tool.anthropic_schema() if hasattr(tool, "anthropic_schema") else {}
                        schema_str = json.dumps(schema, default=str)
                        self.add_chunk(f"{prefix}/tool_schemas/{tool_name}", schema_str)
                    except Exception:
                        pass

            # Conversation history (skip system message at index 0)
            # Aggregate all messages into a single blob, with tool results separated
            if messages:
                messages_content = []
                tool_results_content = []

                for msg in messages[1:] if len(messages) > 1 else []:
                    role = getattr(msg, 'role', 'unknown')
                    content = getattr(msg, 'content', '') or ''

                    # Check if this is a tool result message
                    if role == 'tool':
                        if content:
                            tool_results_content.append(content)
                    else:
                        # Regular message content
                        if content:
                            messages_content.append(content)

                        # Also extract tool call results from assistant messages
                        tool_calls = getattr(msg, 'tool_calls', None) or []
                        for tc in tool_calls:
                            result = getattr(tc, 'result', None)
                            if result:
                                try:
                                    result_str = json.dumps(result, default=str)
                                    tool_results_content.append(result_str)
                                except Exception:
                                    pass

                # Track combined messages content
                if messages_content:
                    combined_messages = "\n---\n".join(messages_content)
                    self.add_chunk(f"{prefix}/messages", combined_messages)

                # Track combined tool results
                if tool_results_content:
                    combined_tool_results = "\n---\n".join(tool_results_content)
                    self.add_chunk(f"{prefix}/messages/tool_results", combined_tool_results)

            # Any extra string values
            for key, value in extras.items():
                if isinstance(value, str) and value:
                    self.add_chunk(f"{prefix}/{key}", value)

        except Exception as e:
            try:
                logger.debug(f"[TokenTracker] track_context failed: {e}")
            except Exception:
                pass

    def get_summary(self) -> Dict[str, int]:
        """Get a summary of tokens by category."""
        try:
            if self._finished:
                return {}
            return self._tracker._get_summary(self._session_run_id)
        except Exception:
            return {}

    def get_total_tokens(self) -> int:
        """Get total estimated tokens."""
        try:
            if self._finished:
                return 0
            return self._tracker._get_total_tokens(self._session_run_id)
        except Exception:
            return 0

    def finish(self, full_prompt: Optional[str] = None) -> bool:
        """
        Flush this call's data to storage.

        Args:
            full_prompt: The complete prompt string sent to the LLM API.

        Returns:
            True if successfully written, False otherwise
        """
        try:
            if self._finished:
                return False
            self._finished = True
            return self._tracker._finish(self._session_run_id, full_prompt)
        except Exception:
            self._finished = True
            return False

    def cancel(self) -> None:
        """Cancel tracking without writing to log."""
        try:
            self._finished = True
            self._tracker._cancel(self._session_run_id)
        except Exception:
            pass


class _NullHandle(TrackerHandle):
    """A null handle that does nothing - used when tracking is disabled."""

    def __init__(self):
        self._finished = True

    def add_chunk(self, category: str, text: str) -> int:
        return 0

    def set_model(self, model: str) -> None:
        pass

    def track_context(self, **kwargs) -> None:
        pass

    def get_summary(self) -> Dict[str, int]:
        return {}

    def get_total_tokens(self) -> int:
        return 0

    def finish(self, full_prompt: Optional[str] = None) -> bool:
        return False

    def cancel(self) -> None:
        pass


# Singleton null handle
_NULL_HANDLE = _NullHandle()


class TokenTracker:
    """
    Tracks token usage for LLM API calls.

    This class manages multiple concurrent tracking sessions, each identified
    by a unique session_run_id. Thread-safe for use in multi-request environments.

    All methods are wrapped in try/except to ensure they never crash the
    main application code.

    Output file:
    token_usage_{DB}.csv - Token breakdown with columns:
       session_run_id, timestamp, agent_id, agent_name, session_id, user_id,
       model, category, tokens, char_count, content

    Usage:
        tracker = token_tracker.register_call(agent_id=..., session_id=..., ...)
        tracker.track_context(agent=actor, session=session, ...)
        tracker.finish(full_prompt=...)
    """

    def __init__(self):
        self._calls: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._log_dir: Optional[Path] = None  # Set lazily in _ensure_log_dir
        self._initialized = False
        self._flush_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
        self._last_cleanup = time.time()
        # Background thread pool for file I/O - max 2 workers to avoid too many open files
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="token_tracker")

    def _cleanup_expired(self) -> None:
        """Remove expired tracking entries to prevent memory leaks."""
        try:
            now = time.time()
            # Only cleanup every 60 seconds
            if now - self._last_cleanup < 60:
                return
            self._last_cleanup = now

            expired = []
            for run_id, call_data in self._calls.items():
                created_at = call_data.get("_created_at", 0)
                if now - created_at > TRACKING_EXPIRY_SECONDS:
                    expired.append(run_id)

            for run_id in expired:
                try:
                    del self._calls[run_id]
                    logger.debug(f"[TokenTracker] Expired tracking for {run_id}")
                except Exception:
                    pass
        except Exception:
            pass

    def _ensure_log_dir(self) -> bool:
        """Ensure the log directory exists. Returns True if ready to write."""
        if self._initialized:
            return True

        try:
            # On Modal, use the mounted volume path
            if os.environ.get("MODAL_ENVIRONMENT"):
                self._log_dir = Path(TOKEN_TRACKER_MODAL_DIR)
            else:
                # Locally, use eve/data/token-tracker/
                self._log_dir = TOKEN_TRACKER_LOCAL_DIR

            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            return True
        except Exception:
            return False

    def set_flush_function(self, fn: Callable[[Dict[str, Any]], bool]) -> None:
        """Set a custom flush function for persisting tracking data."""
        try:
            self._flush_fn = fn
        except Exception:
            pass

    def register_call(
        self,
        agent_id: Any,
        session_id: Any,
        user_id: Optional[Any] = None,
        session_run_id: Optional[str] = None,
        model: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> TrackerHandle:
        """
        Register a new LLM call to track.

        Args:
            agent_id: The agent making this call
            session_id: The session this call belongs to
            user_id: Optional user who triggered this call
            session_run_id: Unique ID for this prompt session run
            model: Optional model name being used
            agent_name: Optional human-readable agent name (username)

        Returns:
            TrackerHandle for this call (use for all subsequent operations)
        """
        try:
            with self._lock:
                self._cleanup_expired()

                run_id = session_run_id or str(uuid.uuid4())
                agent_id_str = _safe_str(agent_id)
                session_id_str = _safe_str(session_id)

                self._calls[run_id] = {
                    "session_run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent_id": agent_id_str,
                    "agent_name": _safe_str(agent_name),
                    "session_id": session_id_str,
                    "user_id": _safe_str(user_id),
                    "model": model,
                    "chunks": [],
                    "_created_at": time.time(),
                }

                return TrackerHandle(self, run_id, agent_id_str, session_id_str)
        except Exception:
            return _NULL_HANDLE

    def _add_chunk(self, session_run_id: str, category: str, text: str) -> int:
        """Internal: Add a chunk to a specific tracking session."""
        try:
            if not text:
                return 0

            with self._lock:
                if session_run_id not in self._calls:
                    return 0

                tokens = estimate_tokens(text)
                self._calls[session_run_id]["chunks"].append({
                    "category": category,
                    "tokens": tokens,
                    "char_count": len(text),
                    "content": text,
                })
                return tokens
        except Exception:
            return 0

    def _set_model(self, session_run_id: str, model: str) -> None:
        """Internal: Set model for a specific tracking session."""
        try:
            with self._lock:
                if session_run_id in self._calls:
                    self._calls[session_run_id]["model"] = model
        except Exception:
            pass

    def _get_summary(self, session_run_id: str) -> Dict[str, int]:
        """Internal: Get summary for a specific tracking session."""
        try:
            with self._lock:
                if session_run_id not in self._calls:
                    return {}

                summary = {}
                for chunk in self._calls[session_run_id]["chunks"]:
                    category = chunk["category"]
                    summary[category] = summary.get(category, 0) + chunk["tokens"]
                return summary
        except Exception:
            return {}

    def _get_total_tokens(self, session_run_id: str) -> int:
        """Internal: Get total tokens for a specific tracking session."""
        try:
            with self._lock:
                if session_run_id not in self._calls:
                    return 0
                return sum(c["tokens"] for c in self._calls[session_run_id]["chunks"])
        except Exception:
            return 0

    def _find_untracked_content(
        self,
        full_prompt: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compare full prompt with tracked chunks to find untracked content."""
        try:
            tracked_contents = [c.get("content", "") for c in chunks]
            untracked_segments = extract_untracked_segments(
                full_prompt,
                tracked_contents,
                min_segment_length=50
            )

            untracked_chunks = []
            for idx, (segment, tokens) in enumerate(untracked_segments):
                untracked_chunks.append({
                    "category": f"untracked_{idx + 1:02d}",
                    "tokens": tokens,
                    "char_count": len(segment),
                    "content": segment,
                })
            return untracked_chunks
        except Exception:
            return []

    def _flush_to_csv(self, call_data: Dict[str, Any]) -> bool:
        """Default flush function - writes to CSV files."""
        try:
            if not self._ensure_log_dir():
                return False

            usage_file = self._log_dir / TOKEN_USAGE_FILE
            usage_exists = usage_file.exists()

            with open(usage_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not usage_exists:
                    writer.writerow([
                        "session_run_id", "timestamp", "agent_id", "agent_name",
                        "session_id", "user_id", "model", "category", "tokens",
                        "char_count", "content",
                    ])

                for chunk in call_data["chunks"]:
                    writer.writerow([
                        call_data["session_run_id"],
                        call_data["timestamp"],
                        call_data["agent_id"],
                        call_data.get("agent_name", ""),
                        call_data["session_id"],
                        call_data["user_id"],
                        call_data["model"],
                        chunk["category"],
                        chunk["tokens"],
                        chunk["char_count"],
                        chunk.get("content", ""),
                    ])

            return True
        except Exception:
            return False

    def _finish(self, session_run_id: str, full_prompt: Optional[str] = None) -> bool:
        """Internal: Flush a specific tracking session to storage.

        File I/O is performed in a background thread to avoid blocking the main loop.
        """
        try:
            with self._lock:
                if session_run_id not in self._calls:
                    return False

                call_data = self._calls.pop(session_run_id)

            # Process data preparation synchronously (fast, in-memory operations)
            if full_prompt:
                call_data["chunks"].append({
                    "category": "_full_prompt",
                    "tokens": estimate_tokens(full_prompt),
                    "char_count": len(full_prompt),
                    "content": full_prompt,
                })

                tracked_chunks = [
                    c for c in call_data["chunks"]
                    if not c["category"].startswith("_")
                ]
                untracked = self._find_untracked_content(full_prompt, tracked_chunks)
                call_data["chunks"].extend(untracked)

            total_tokens = sum(
                c["tokens"] for c in call_data["chunks"]
                if not c["category"].startswith("_")
            )

            untracked_count = sum(
                1 for c in call_data["chunks"]
                if c["category"].startswith("untracked_")
            )
            untracked_tokens = sum(
                c["tokens"] for c in call_data["chunks"]
                if c["category"].startswith("untracked_")
            )

            log_msg = (
                f"[TokenTracker] LLM call logged | session_run_id={session_run_id} | "
                f"agent={call_data.get('agent_id', 'unknown')} | "
                f"total_tokens={total_tokens}"
            )
            if untracked_count > 0:
                log_msg += f" | untracked_chunks={untracked_count} (~{untracked_tokens} tokens)"

            try:
                logger.info(log_msg)
            except Exception:
                pass

            # Submit file I/O to background thread pool to avoid blocking main loop
            def background_flush():
                try:
                    if self._flush_fn:
                        self._flush_fn(call_data)
                    else:
                        self._flush_to_csv(call_data)
                except Exception as e:
                    try:
                        logger.warning(f"[TokenTracker] Background flush failed: {e}")
                    except Exception:
                        pass

            self._executor.submit(background_flush)
            return True

        except Exception as e:
            try:
                logger.warning(f"[TokenTracker] Failed to write log: {e}")
            except Exception:
                pass
            # Ensure cleanup even on error
            try:
                with self._lock:
                    self._calls.pop(session_run_id, None)
            except Exception:
                pass
            return False

    def _cancel(self, session_run_id: str) -> None:
        """Internal: Cancel tracking for a specific session."""
        try:
            with self._lock:
                self._calls.pop(session_run_id, None)
        except Exception:
            pass

    # =========================================================================
    # Public convenience methods for direct session_run_id access
    # These allow simpler integration without needing to pass handles around
    # =========================================================================

    def track_context(
        self,
        session_run_id: str,
        agent: Any = None,
        session: Any = None,
        user: Any = None,
        tools: Dict = None,
        messages: List = None,
        trigger_context: Dict = None,
        prefix: str = "system",
        **extras,
    ) -> None:
        """
        Track context for a specific session_run_id.

        This is a convenience method that doesn't require passing a handle.
        Use after register_call() with the same session_run_id.

        Note: System message components (memory, persona, etc.) are tracked
        separately via render_template_with_token_tracking() in build_system_message.
        """
        try:
            with self._lock:
                if session_run_id not in self._calls:
                    return

            # Create a temporary handle for tracking
            handle = TrackerHandle(
                self,
                session_run_id,
                self._calls.get(session_run_id, {}).get("agent_id"),
                self._calls.get(session_run_id, {}).get("session_id"),
            )
            handle.track_context(
                agent=agent,
                session=session,
                user=user,
                tools=tools,
                messages=messages,
                trigger_context=trigger_context,
                prefix=prefix,
                **extras,
            )
        except Exception:
            pass

    def set_model(self, session_run_id: str, model: str) -> None:
        """Set model for a specific session_run_id."""
        try:
            self._set_model(session_run_id, model)
        except Exception:
            pass

    def finish(
        self,
        session_run_id: str,
        full_prompt: Optional[str] = None,
    ) -> bool:
        """
        Finish tracking for a specific session_run_id.

        This is a convenience method that doesn't require passing a handle.
        Use after register_call() and track_context() with the same session_run_id.
        """
        try:
            return self._finish(session_run_id, full_prompt)
        except Exception:
            return False

    def cancel(self, session_run_id: str) -> None:
        """Cancel tracking for a specific session_run_id."""
        try:
            self._cancel(session_run_id)
        except Exception:
            pass


# Global singleton instance
token_tracker = TokenTracker()
