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
    from eve.agent.llm.token_tracker import token_tracker

    # Track an LLM request - single entry point
    await token_tracker.track_request(
        model=context.config.model,
        system=system_message,
        messages=context.messages,
        tools=context.tools,
        instrumentation=context.instrumentation,
        metadata=context.metadata,
    )
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from eve.agent.session.models import ChatMessage
    from eve.tool import Tool

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


def normalize_for_comparison(text: str) -> str:
    """Normalize text for fuzzy comparison by collapsing whitespace."""
    try:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.strip().lower())
    except Exception:
        return ""


def generate_content_fingerprint(text: str, max_length: int = 40) -> str:
    """
    Generate a stable, human-readable fingerprint from content.

    This creates consistent identifiers for untracked segments so that
    the same static template content gets the same name across different
    API calls, even if it appears at different positions.

    Args:
        text: The content to fingerprint
        max_length: Maximum length of the fingerprint identifier

    Returns:
        A stable identifier like "agent_spec_name_ver" or "behavior_capabiliti"
    """
    try:
        if not text:
            return "empty"

        # Normalize: collapse whitespace, lowercase
        normalized = re.sub(r"\s+", " ", text.strip().lower())

        # Extract meaningful content by removing common XML/markup patterns
        # This helps get to the "meat" of the content for fingerprinting
        # Remove XML tags but keep their names as they're often descriptive
        tag_names = re.findall(r"</?([a-z_]+)", normalized)

        # Remove XML tags, keeping content
        cleaned = re.sub(r"<[^>]+>", " ", normalized)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Build fingerprint from either tag names or content
        if tag_names and len(tag_names) >= 2:
            # Use first few tag names if XML-heavy
            fingerprint = "_".join(tag_names[:3])
        elif cleaned:
            # Use first N characters of cleaned content
            # Keep only alphanumeric and spaces, then convert to identifier
            alpha_only = re.sub(r"[^a-z0-9\s]", "", cleaned)
            words = alpha_only.split()[:5]  # First 5 words
            fingerprint = "_".join(words)
        else:
            fingerprint = "unknown"

        # Ensure valid identifier: starts with letter, only alphanumeric and underscore
        fingerprint = re.sub(r"[^a-z0-9_]", "_", fingerprint)
        fingerprint = re.sub(r"_+", "_", fingerprint).strip("_")

        # Truncate to max length
        if len(fingerprint) > max_length:
            fingerprint = fingerprint[:max_length].rstrip("_")

        # Ensure non-empty
        if not fingerprint:
            fingerprint = "content"

        return fingerprint

    except Exception:
        return "unknown"


def extract_untracked_segments(
    full_prompt: str, tracked_chunks: List[str], min_segment_length: int = 50
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


def parse_xml_sections(
    text: str,
    terminal_tags: Optional[Set[str]] = None,
    prefix: str = "system",
    max_depth: int = 5,
) -> List[Dict[str, Any]]:
    """
    Parse XML tags from text and create hierarchical token categories.

    This function dynamically extracts XML tags from the input text and creates
    a nested hierarchy of categories for token tracking. It handles nested tags,
    self-closing tags, and text outside of tags.

    Args:
        text: The text to parse (e.g., rendered system message)
        terminal_tags: Tag names that should not be further parsed (their content
                      is treated as a single chunk, including any nested XML)
        prefix: Category prefix (e.g., "system")
        max_depth: Maximum number of levels to descend into the XML hierarchy.
                   Default is 5, meaning paths like "system/AGENT_SPEC/Tools/CreateTool/UseCases"
                   (5 levels after prefix) are allowed, but deeper paths are truncated.
                   Tags at max_depth become terminal (their content is captured without
                   further parsing).

    Returns:
        List of {category: str, tokens: int, char_count: int, content: str}

    Example:
        >>> text = "<AGENT_SPEC><Summary>Hello</Summary><Persona>Be nice</Persona></AGENT_SPEC>"
        >>> sections = parse_xml_sections(text, terminal_tags={"Persona"}, prefix="system")
        >>> # Returns:
        >>> # [
        >>> #   {"category": "system/AGENT_SPEC/Summary", "tokens": 1, "char_count": 5, "content": "Hello"},
        >>> #   {"category": "system/AGENT_SPEC/Persona", "tokens": 2, "char_count": 7, "content": "Be nice"},
        >>> # ]
    """
    try:
        if not text:
            return []

        terminal_tags = terminal_tags or set()
        results: List[Dict[str, Any]] = []

        # Regex patterns for XML tags
        # Opening tag: <TagName attr="value">
        opening_tag_pattern = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)\s*[^>]*(?<!/)>")
        # Closing tag: </TagName>
        closing_tag_pattern = re.compile(r"</([A-Za-z_][A-Za-z0-9_]*)>")
        # Self-closing tag: <TagName attr="value"/>
        self_closing_pattern = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)\s*[^>]*/\s*>")

        # Stack to track current path in the XML hierarchy
        path_stack: List[str] = []
        # Track positions of tag openings for content extraction
        tag_start_positions: List[int] = []
        # Track which tags are terminal (their content should not be further parsed)
        terminal_depth: Optional[int] = None
        # Track whether terminal mode was triggered by max_depth (vs explicit terminal_tags)
        terminal_due_to_depth: bool = False

        # Find all tag positions and types
        events: List[Tuple[int, str, str, int]] = []  # (position, type, tag_name, end_position)

        for match in opening_tag_pattern.finditer(text):
            # Check if this is actually a self-closing tag (avoid double-counting)
            full_match = text[match.start() : match.end() + 10]  # Look ahead for />
            if "/>" not in text[match.start() : match.end() + 2]:
                events.append((match.start(), "open", match.group(1), match.end()))

        for match in closing_tag_pattern.finditer(text):
            events.append((match.start(), "close", match.group(1), match.end()))

        for match in self_closing_pattern.finditer(text):
            events.append((match.start(), "self_close", match.group(1), match.end()))

        # Sort events by position
        events.sort(key=lambda x: x[0])

        # Track text content between tags at each level
        last_end_position = 0
        pending_text_content: Dict[int, str] = {}  # depth -> accumulated text

        for pos, event_type, tag_name, end_pos in events:
            current_depth = len(path_stack)

            # Capture text content before this tag (if not inside a terminal tag)
            if terminal_depth is None and pos > last_end_position:
                text_before = text[last_end_position:pos].strip()
                if text_before and current_depth > 0:
                    # Accumulate text at current depth
                    if current_depth not in pending_text_content:
                        pending_text_content[current_depth] = ""
                    pending_text_content[current_depth] += text_before + " "

            if event_type == "open":
                # Check if we're entering a terminal tag or exceeding max_depth
                # When path_stack has max_depth-1 elements, the next tag would exceed max_depth
                if terminal_depth is None and tag_name in terminal_tags:
                    terminal_depth = current_depth
                    terminal_due_to_depth = False
                    # Store the start position for terminal tag content extraction
                    tag_start_positions.append(end_pos)
                elif terminal_depth is None and len(path_stack) >= max_depth - 1:
                    # Max depth exceeded - content will be attributed to parent tag
                    terminal_depth = current_depth
                    terminal_due_to_depth = True
                    tag_start_positions.append(end_pos)
                elif terminal_depth is None:
                    tag_start_positions.append(end_pos)

                path_stack.append(tag_name)
                last_end_position = end_pos

            elif event_type == "close":
                if path_stack and path_stack[-1] == tag_name:
                    # Build the category path
                    # When terminal due to max_depth, exclude the terminal tag from category
                    if terminal_due_to_depth and terminal_depth is not None and len(path_stack) - 1 == terminal_depth:
                        category = f"{prefix}/{'/'.join(path_stack[:-1])}"
                    else:
                        category = f"{prefix}/{'/'.join(path_stack)}"

                    # Check if we're closing a terminal tag
                    if terminal_depth is not None and len(path_stack) - 1 == terminal_depth:
                        # Extract all content from the terminal tag (including nested XML)
                        start_pos = tag_start_positions.pop() if tag_start_positions else last_end_position
                        content = text[start_pos:pos].strip()
                        if content:
                            results.append(
                                {
                                    "category": category,
                                    "tokens": estimate_tokens(content),
                                    "char_count": len(content),
                                    "content": content,
                                }
                            )
                        terminal_depth = None
                        terminal_due_to_depth = False
                    elif terminal_depth is None:
                        # Normal tag - check for accumulated text content
                        if tag_start_positions:
                            tag_start_positions.pop()

                        depth = len(path_stack)
                        if depth in pending_text_content:
                            content = pending_text_content.pop(depth).strip()
                            if content:
                                results.append(
                                    {
                                        "category": category,
                                        "tokens": estimate_tokens(content),
                                        "char_count": len(content),
                                        "content": content,
                                    }
                                )

                    path_stack.pop()
                    last_end_position = end_pos

            elif event_type == "self_close":
                # Self-closing tags have no content, but we still track them
                if terminal_depth is None:
                    category = f"{prefix}/{'/'.join(path_stack + [tag_name])}"
                    # Extract any attributes as content
                    tag_text = text[pos:end_pos]
                    results.append(
                        {
                            "category": category,
                            "tokens": estimate_tokens(tag_text),
                            "char_count": len(tag_text),
                            "content": tag_text,
                        }
                    )
                last_end_position = end_pos

        # Handle any remaining text after all tags
        if terminal_depth is None and last_end_position < len(text):
            remaining_text = text[last_end_position:].strip()
            if remaining_text:
                category = prefix if not path_stack else f"{prefix}/{'/'.join(path_stack)}"
                results.append(
                    {
                        "category": category,
                        "tokens": estimate_tokens(remaining_text),
                        "char_count": len(remaining_text),
                        "content": remaining_text,
                    }
                )

        return results

    except Exception as e:
        logger.debug(f"[TokenTracker] parse_xml_sections failed: {e}")
        # Fallback: return the entire text as a single chunk
        try:
            return [
                {
                    "category": f"{prefix}/_unparsed",
                    "tokens": estimate_tokens(text),
                    "char_count": len(text),
                    "content": text,
                }
            ]
        except Exception:
            return []


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
        await token_tracker.track_request(
            model=context.config.model,
            system=system_message,
            messages=context.messages,
            tools=context.tools,
            instrumentation=context.instrumentation,
            metadata=context.metadata,
        )
    """

    def __init__(self):
        self._calls: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._log_dir: Optional[Path] = None  # Set lazily in _ensure_log_dir
        self._initialized = False
        self._flush_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
        self._last_cleanup = time.time()
        # Background thread pool for file I/O - max 2 workers to avoid too many open files
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="token_tracker"
        )

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
                # Check if the mount point exists before trying to create subdirs
                if not self._log_dir.parent.exists():
                    logger.warning(f"[TokenTracker] Modal volume mount not available: {self._log_dir.parent}")
                    return False
            else:
                # Locally, use eve/data/token-tracker/
                self._log_dir = TOKEN_TRACKER_LOCAL_DIR

            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"[TokenTracker] Failed to initialize log directory: {e}")
            return False

    def set_flush_function(self, fn: Callable[[Dict[str, Any]], bool]) -> None:
        """Set a custom flush function for persisting tracking data."""
        try:
            self._flush_fn = fn
        except Exception:
            pass

    def _add_chunk(self, session_run_id: str, category: str, text: str) -> int:
        """Internal: Add a chunk to a specific tracking session."""
        try:
            if not text:
                return 0

            with self._lock:
                if session_run_id not in self._calls:
                    return 0

                tokens = estimate_tokens(text)
                self._calls[session_run_id]["chunks"].append(
                    {
                        "category": category,
                        "tokens": tokens,
                        "char_count": len(text),
                        "content": text,
                    }
                )
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
        self, full_prompt: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compare full prompt with tracked chunks to find untracked content.

        Uses content-based fingerprinting to generate stable category names
        that remain consistent across different API calls, even when the
        same static content appears at different positions in the prompt.
        """
        try:
            tracked_contents = [c.get("content", "") for c in chunks]
            untracked_segments = extract_untracked_segments(
                full_prompt, tracked_contents, min_segment_length=50
            )

            untracked_chunks = []
            fingerprint_counts: Dict[str, int] = {}  # Track collisions

            for segment, tokens in untracked_segments:
                # Generate content-based fingerprint for stable naming
                fingerprint = generate_content_fingerprint(segment)

                # Handle collisions by appending a counter
                if fingerprint in fingerprint_counts:
                    fingerprint_counts[fingerprint] += 1
                    category = (
                        f"untracked/{fingerprint}_{fingerprint_counts[fingerprint]}"
                    )
                else:
                    fingerprint_counts[fingerprint] = 1
                    category = f"untracked/{fingerprint}"

                untracked_chunks.append(
                    {
                        "category": category,
                        "tokens": tokens,
                        "char_count": len(segment),
                        "content": segment,
                    }
                )
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
                    writer.writerow(
                        [
                            "session_run_id",
                            "timestamp",
                            "agent_id",
                            "agent_name",
                            "session_id",
                            "user_id",
                            "model",
                            "category",
                            "tokens",
                            "char_count",
                            "content",
                        ]
                    )

                for chunk in call_data["chunks"]:
                    writer.writerow(
                        [
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
                        ]
                    )

            return True
        except Exception:
            return False

    def _finalize_call_data(
        self, session_run_id: str, full_prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            with self._lock:
                if session_run_id not in self._calls:
                    return None
                call_data = self._calls.pop(session_run_id)

            # Process data preparation synchronously (fast, in-memory operations)
            if full_prompt:
                call_data["chunks"].append(
                    {
                        "category": "_full_prompt",
                        "tokens": estimate_tokens(full_prompt),
                        "char_count": len(full_prompt),
                        "content": full_prompt,
                    }
                )

                # Skip untracked detection if template_structure is present.
                # template_structure uses a space placeholder for token counting (not actual
                # content), so untracked detection would flag the real XML as "untracked".
                # Since template_structure already accounts for this overhead with accurate
                # char counts, untracked detection is redundant.
                has_template_structure = any(
                    c.get("category", "").endswith("/template_structure")
                    for c in call_data["chunks"]
                )
                if not has_template_structure:
                    tracked_chunks = [
                        c for c in call_data["chunks"] if not c["category"].startswith("_")
                    ]
                    untracked = self._find_untracked_content(full_prompt, tracked_chunks)
                    call_data["chunks"].extend(untracked)

            summary: Dict[str, int] = {}
            for chunk in call_data["chunks"]:
                category = chunk.get("category") or ""
                if category.startswith("_"):
                    continue
                summary[category] = summary.get(category, 0) + chunk.get("tokens", 0)

            total_tokens = sum(summary.values())
            call_data["_summary"] = summary
            call_data["_total_tokens"] = total_tokens

            untracked_count = sum(
                1
                for c in call_data["chunks"]
                if (c.get("category") or "").startswith("untracked_")
            )
            untracked_tokens = sum(
                c.get("tokens", 0)
                for c in call_data["chunks"]
                if (c.get("category") or "").startswith("untracked_")
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

            return call_data
        except Exception as e:
            try:
                logger.warning(f"[TokenTracker] Failed to finalize call data: {e}")
            except Exception:
                pass
            try:
                with self._lock:
                    self._calls.pop(session_run_id, None)
            except Exception:
                pass
            return None

    def _flush_call_data(self, call_data: Dict[str, Any]) -> None:
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

        try:
            self._executor.submit(background_flush)
        except Exception as e:
            logger.warning(f"[TokenTracker] Failed to submit flush task: {e}")

    def _finish(self, session_run_id: str, full_prompt: Optional[str] = None) -> bool:
        """Internal: Flush a specific tracking session to storage.

        File I/O is performed in a background thread to avoid blocking the main loop.
        """
        call_data = self._finalize_call_data(session_run_id, full_prompt)
        if not call_data:
            return False
        self._flush_call_data(call_data)
        return True

    def _finish_with_summary(
        self, session_run_id: str, full_prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Internal: Finish and return summary data for this tracking session."""
        call_data = self._finalize_call_data(session_run_id, full_prompt)
        if not call_data:
            return None
        self._flush_call_data(call_data)
        return {
            "total_tokens": call_data.get("_total_tokens"),
            "breakdown": call_data.get("_summary"),
        }

    def _cancel(self, session_run_id: str) -> None:
        """Internal: Cancel tracking for a specific session."""
        try:
            with self._lock:
                self._calls.pop(session_run_id, None)
        except Exception:
            pass

    async def track_request(
        self,
        model: str,
        system: str,
        messages: List["ChatMessage"],
        tools: Optional[Union[Dict[str, "Tool"], List["Tool"]]] = None,
        terminal_tags: Optional[List[str]] = None,
        # Optional - will be extracted from instrumentation/metadata if not provided
        session_run_id: Optional[str] = None,
        agent_id: Any = None,
        agent_name: Optional[str] = None,
        session_id: Any = None,
        user_id: Optional[Any] = None,
        # Pass these to let track_request extract metadata automatically
        instrumentation: Any = None,
        metadata: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Track an LLM API request with automatic XML parsing for system messages.

        This is the primary entry point for token tracking. Call this method once,
        right before the actual LLM API call, with the exact data being sent.

        The method:
        1. Extracts metadata from instrumentation/metadata objects if not provided directly
        2. Parses the system message to extract XML hierarchy (system/*)
        3. Tracks each message by role (messages/user, messages/assistant)
        4. Tracks tool results from assistant messages (messages/tool_result/{tool_name})
        5. Tracks each tool schema (tools/{tool_name})
        6. Flushes to storage asynchronously

        Args:
            model: Model name being used
            system: The fully rendered system message string
            messages: List of ChatMessage objects (the conversation)
            tools: Dict or List of Tool objects
            terminal_tags: XML tag names that should not be further parsed
                          (e.g., ["Persona", "SessionContext", "Instructions"])
            session_run_id: Unique ID for this prompt session run (auto-generated if not provided)
            agent_id: The agent making this call (extracted from instrumentation/metadata if not provided)
            agent_name: Human-readable agent name
            session_id: The session this call belongs to
            user_id: User who triggered this call
            instrumentation: PromptSessionInstrumentation object (for extracting metadata)
            metadata: LLMContextMetadata object (for extracting metadata)

        Returns:
            Summary dict with "total_tokens" and "breakdown" (or None on error)

        Example:
            await token_tracker.track_request(
                model="claude-opus-4-20250514",
                system=rendered_system_message,
                messages=context.messages,
                tools=context.tools,
                instrumentation=context.instrumentation,
                metadata=context.metadata,
            )
        """
        try:
            # Default terminal tags if not provided
            if terminal_tags is None:
                terminal_tags = []

            # Extract metadata from instrumentation if available
            if instrumentation:
                if not session_run_id:
                    session_run_id = getattr(instrumentation, "session_run_id", None)
                if not agent_id:
                    agent_id = getattr(instrumentation, "agent_id", None)
                if not agent_name:
                    agent_name = getattr(instrumentation, "agent_name", None)
                if not user_id:
                    user_id = getattr(instrumentation, "user_id", None)
                if not session_id:
                    session_id = getattr(instrumentation, "session_id", None)

            # Extract metadata from metadata object if available
            if metadata:
                if not session_id:
                    session_id = getattr(metadata, "session_id", None)
                # Check trace_metadata for additional info
                trace_metadata = getattr(metadata, "trace_metadata", None)
                if trace_metadata:
                    if not agent_id:
                        agent_id = getattr(trace_metadata, "agent_id", None)
                    if not agent_name:
                        agent_name = getattr(trace_metadata, "agent_name", None)
                    if not user_id:
                        user_id = getattr(trace_metadata, "user_id", None)
                    if not session_id:
                        session_id = getattr(trace_metadata, "session_id", None)

            # Generate session_run_id if still not available
            if not session_run_id:
                session_run_id = str(uuid.uuid4())

            # Register the call
            with self._lock:
                self._cleanup_expired()

                agent_id_str = _safe_str(agent_id)
                session_id_str = _safe_str(session_id)

                self._calls[session_run_id] = {
                    "session_run_id": session_run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent_id": agent_id_str,
                    "agent_name": _safe_str(agent_name),
                    "session_id": session_id_str,
                    "user_id": _safe_str(user_id),
                    "model": model,
                    "chunks": [],
                    "_created_at": time.time(),
                }

            # Convert terminal_tags to a set for efficient lookup
            terminal_tag_set = set(terminal_tags) if terminal_tags else set()

            # 1. Parse system message with XML hierarchy
            if system:
                system_sections = parse_xml_sections(
                    system,
                    terminal_tags=terminal_tag_set,
                    prefix="system",
                )
                for section in system_sections:
                    self._add_chunk(
                        session_run_id,
                        section["category"],
                        section["content"],
                    )

            # 2. Track messages by role
            if messages:
                for msg in messages:
                    try:
                        # Skip system messages - they're tracked via system parameter
                        if msg.role == "system":
                            continue

                        # Determine the category based on role
                        role = msg.role
                        if role == "tool":
                            role = "tool_result"

                        # Serialize message content
                        content = msg.content or ""

                        # Include tool_calls in assistant messages if present
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            tool_calls_str = json.dumps(
                                [
                                    {"name": tc.name, "args": tc.args}
                                    for tc in msg.tool_calls
                                ],
                                default=str,
                            )
                            content = f"{content}\n[Tool Calls: {tool_calls_str}]"

                        if content:
                            self._add_chunk(
                                session_run_id,
                                f"messages/{role}",
                                content,
                            )

                        # Track tool results separately (these consume significant tokens)
                        # Tool results are stored in the tool_calls of assistant messages
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                try:
                                    tool_name = getattr(tc, "tool", None) or getattr(tc, "name", "unknown")

                                    # Build result data matching what gets sent to the API
                                    result_data = {"status": getattr(tc, "status", None)}
                                    if getattr(tc, "error", None):
                                        result_data["error"] = tc.error
                                    if getattr(tc, "result", None) is not None:
                                        result_data["result"] = tc.result

                                    result_content = json.dumps(result_data, default=str)
                                    self._add_chunk(
                                        session_run_id,
                                        f"messages/tool_result/{tool_name}",
                                        result_content,
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass

            # 3. Track tools by name
            if tools:
                tools_iter = tools.items() if isinstance(tools, dict) else [(t.key if hasattr(t, "key") else t.name, t) for t in tools]
                for tool_name, tool in tools_iter:
                    try:
                        # Get the tool schema
                        schema = (
                            tool.anthropic_schema()
                            if hasattr(tool, "anthropic_schema")
                            else {}
                        )
                        schema_str = json.dumps(schema, default=str)
                        self._add_chunk(
                            session_run_id,
                            f"tools/{tool_name}",
                            schema_str,
                        )
                    except Exception:
                        pass

            # 4. Finalize and flush asynchronously
            call_data = self._finalize_call_data(session_run_id, full_prompt=None)
            if call_data:
                self._flush_call_data(call_data)
                return {
                    "total_tokens": call_data.get("_total_tokens"),
                    "breakdown": call_data.get("_summary"),
                }

            return None

        except Exception as e:
            logger.warning(f"[TokenTracker] track_request failed: {e}")
            # Clean up on error
            try:
                with self._lock:
                    self._calls.pop(session_run_id, None)
            except Exception:
                pass
            return None


# Global singleton instance
token_tracker = TokenTracker()
