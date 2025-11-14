from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from eve.agent.session.debug_logger import SessionDebugger, DEBUG_SESSION

try:
    import sentry_sdk
except ImportError:  # pragma: no cover - sentry may not be installed in tests
    sentry_sdk = None  # type: ignore

try:
    from langfuse import Langfuse
    from langfuse.client import StatefulGenerationClient, StatefulSpanClient
except ImportError:  # pragma: no cover - langfuse optional
    Langfuse = None  # type: ignore
    StatefulGenerationClient = None  # type: ignore
    StatefulSpanClient = None  # type: ignore


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_DEBUG_ENABLED = _bool_env("SESSION_OBS_DEBUG", DEBUG_SESSION)
DEFAULT_SENTRY_ENABLED = _bool_env("SESSION_OBS_SENTRY", bool(os.getenv("SENTRY_DSN")))
DEFAULT_LANGFUSE_ENABLED = _bool_env(
    "SESSION_OBS_LANGFUSE",
    bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")),
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_level(level: str) -> str:
    return level.upper() if level else "INFO"


@dataclass
class StageRecord:
    id: str
    name: str
    level: str
    started_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    error: Optional[str] = None

    def close(
        self,
        status: str,
        finished_at: float,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.status = status
        self.duration_ms = round((finished_at - self.started_at) * 1000, 2)
        if metadata:
            self.metadata.update(metadata)
        if error:
            self.error = str(error)


class _StageScope:
    def __init__(
        self, instrumentation: "PromptSessionInstrumentation", record: StageRecord
    ):
        self.instrumentation = instrumentation
        self.record = record

    def __enter__(self) -> "_StageScope":
        return self

    def annotate(self, **metadata: Any) -> None:
        self.instrumentation.update_stage_metadata(self.record.id, metadata)

    def __exit__(self, exc_type, exc, _tb) -> bool:
        status = "error" if exc else "success"
        self.instrumentation._end_stage(
            self.record.id,
            status=status,
            error=str(exc) if exc else None,
        )
        return False


class PromptSessionInstrumentation:
    """Structured observability helper for prompt sessions.

    This class combines three concerns:
        1. Human-readable debug logging (via SessionDebugger).
        2. Structured info/warning/error logs enriched with session metadata.
        3. Optional integrations for Sentry transactions and Langfuse traces/spans.

    Usage pattern:

        inst = PromptSessionInstrumentation(
            session_id=str(session.id),
            session_run_id=context.session_run_id,
            user_id=context.initiating_user_id,
            agent_id=str(actor.id),
        )

        with inst.track_stage("determine_actors", level="info"):
            actors = await determine_actors(...)

        inst.log_io("llm", "outbound", {"model": llm_context.config.model})
    """

    def __init__(
        self,
        *,
        session_id: Optional[str],
        session_run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        trace_name: str = "prompt_session",
        debug_enabled: Optional[bool] = None,
        sentry_enabled: Optional[bool] = None,
        langfuse_enabled: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.session_run_id = session_run_id or str(uuid.uuid4())
        self.user_id = user_id
        self.agent_id = agent_id
        self.trace_name = trace_name
        self.extra_metadata = metadata or {}
        self._start_clock = time.perf_counter()

        debug_flag = (
            debug_enabled if debug_enabled is not None else DEFAULT_DEBUG_ENABLED
        )
        self.debugger = SessionDebugger(session_id, enabled=debug_flag)
        self._structured_logger = None
        self._bind_logger()

        self._profiling: List[StageRecord] = []
        self._stage_stack: List[str] = []
        self._open_stages: Dict[str, StageRecord] = {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, Any] = {}

        self._sentry_enabled = (
            sentry_enabled if sentry_enabled is not None else DEFAULT_SENTRY_ENABLED
        ) and sentry_sdk is not None
        self._sentry_transaction = None

        langfuse_flag = (
            langfuse_enabled
            if langfuse_enabled is not None
            else DEFAULT_LANGFUSE_ENABLED
        )
        self._langfuse_enabled = bool(langfuse_flag and Langfuse is not None)
        self._langfuse_client: Optional[Langfuse] = None
        self._langfuse_trace_started = False
        self._langfuse_default_tags = ["prompt_session"]

    # --------------------------------------------------------------------- #
    # Logging helpers
    # --------------------------------------------------------------------- #
    def log_event(
        self,
        message: str,
        *,
        level: str = "info",
        payload: Optional[Dict[str, Any]] = None,
        emoji: Optional[str] = None,
    ) -> None:
        level_name = _normalize_level(level)
        log = self._structured_logger
        if payload:
            log = log.bind(payload=payload)
        log.log(level_name, message)

        if self.debugger and self.debugger.enabled:
            self.debugger.log(message, payload, level=level_name.lower(), emoji=emoji)

    def log_io(
        self,
        kind: str,
        direction: str,
        payload: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        data = {"kind": kind, "direction": direction}
        if payload:
            data.update(payload)
        self.log_event(
            f"I/O {kind} ({direction})", level=level, payload=data, emoji="data"
        )

    def log_update(
        self, update_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        self.log_event(
            f"Session update: {update_type}",
            level="debug",
            payload=payload or {},
            emoji="update",
        )

    def log_error(
        self, message: str, error: Exception, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        if payload:
            details.update(payload)
        self.log_event(message, level="error", payload=details, emoji="error")
        self.capture_exception(error, context=details)

    # --------------------------------------------------------------------- #
    # Stage profiling
    # --------------------------------------------------------------------- #
    def track_stage(
        self,
        name: str,
        *,
        level: str = "debug",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _StageScope:
        record = self._start_stage(name, level=level, metadata=metadata)
        return _StageScope(self, record)

    def stage_start(
        self,
        name: str,
        *,
        level: str = "debug",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        record = self._start_stage(name, level=level, metadata=metadata)
        return record.id

    def stage_end(
        self,
        stage_id: str,
        *,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self._end_stage(stage_id, status=status, metadata=metadata, error=error)

    def _bind_logger(self) -> None:
        self._structured_logger = logger.bind(
            component="PromptSession",
            session_id=self.session_id,
            session_run_id=self.session_run_id,
            agent_id=self.agent_id,
            user_id=self.user_id,
        )

    def update_context(
        self,
        *,
        session_id: Optional[str] = None,
        session_run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        if session_id is not None:
            self.session_id = session_id
        if session_run_id is not None:
            self.session_run_id = session_run_id
        if agent_id is not None:
            self.agent_id = agent_id
        if user_id is not None:
            self.user_id = user_id
        if self.debugger:
            self.debugger.set_session_context(
                session_id=self.session_id, session_run_id=self.session_run_id
            )
        self._bind_logger()

    def _start_stage(
        self,
        name: str,
        *,
        level: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StageRecord:
        stage_id = str(uuid.uuid4())
        parent_id = self._stage_stack[-1] if self._stage_stack else None
        record = StageRecord(
            id=stage_id,
            name=name,
            level=_normalize_level(level),
            started_at=time.perf_counter(),
            metadata=metadata or {},
            parent_id=parent_id,
        )
        self._profiling.append(record)
        self._open_stages[stage_id] = record
        self._stage_stack.append(stage_id)
        self.log_event(
            f"[stage:start] {name}",
            level=level,
            payload={"metadata": metadata, "parent_stage": parent_id},
            emoji="start",
        )
        self.add_breadcrumb(f"stage:{name}:start", metadata or {})
        return record

    def _end_stage(
        self,
        stage_id: str,
        *,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        record = self._open_stages.pop(stage_id, None)
        if not record:
            return
        finished_at = time.perf_counter()
        record.close(
            status=status, finished_at=finished_at, metadata=metadata, error=error
        )
        if self._stage_stack and self._stage_stack[-1] == stage_id:
            self._stage_stack.pop()
        self.log_event(
            f"[stage:{status}] {record.name}",
            level=record.level,
            payload={
                "metadata": record.metadata,
                "duration_ms": record.duration_ms,
                "error": record.error,
            },
            emoji="end" if status == "success" else "error",
        )
        breadcrumb_data = {"duration_ms": record.duration_ms}
        if metadata:
            breadcrumb_data.update(metadata)
        self.add_breadcrumb(f"stage:{record.name}:{status}", breadcrumb_data)

    def update_stage_metadata(self, stage_id: str, metadata: Dict[str, Any]) -> None:
        record = self._open_stages.get(stage_id)
        if record and metadata:
            record.metadata.update(metadata)

    # --------------------------------------------------------------------- #
    # Metrics helpers
    # --------------------------------------------------------------------- #
    def record_counter(self, name: str, value: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0.0) + value

    def set_gauge(self, name: str, value: Any) -> None:
        self._gauges[name] = value

    def summary(self) -> Dict[str, Any]:
        total_ms = round((time.perf_counter() - self._start_clock) * 1000, 2)
        completed_stages = [
            {
                "name": r.name,
                "duration_ms": r.duration_ms,
                "status": r.status,
                "metadata": r.metadata,
                "error": r.error,
            }
            for r in self._profiling
            if r.duration_ms is not None
        ]
        return {
            "session_id": self.session_id,
            "session_run_id": self.session_run_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "total_duration_ms": total_ms,
            "stages": completed_stages,
            "counters": self._counters,
            "gauges": self._gauges,
            "metadata": self.extra_metadata,
        }

    # --------------------------------------------------------------------- #
    # Sentry helpers
    # --------------------------------------------------------------------- #
    def ensure_sentry_transaction(
        self,
        *,
        name: Optional[str] = None,
        op: str = "session.prompt",
        sampled: Optional[bool] = None,
    ):
        if not self._sentry_enabled:
            return None
        if self._sentry_transaction:
            return self._sentry_transaction
        transaction_name = name or self.trace_name
        transaction = sentry_sdk.start_transaction(
            name=transaction_name, op=op, sampled=sampled
        )
        if transaction:
            transaction.set_tag("session_id", self.session_id)
            transaction.set_tag("session_run_id", self.session_run_id)
            if self.user_id:
                transaction.set_tag("user_id", self.user_id)
            if self.agent_id:
                transaction.set_tag("agent_id", self.agent_id)
            for key, value in self.extra_metadata.items():
                transaction.set_tag(key, value)
            sentry_sdk.Hub.current.scope.span = transaction  # type: ignore[attr-defined]
        self._sentry_transaction = transaction
        return transaction

    def attach_sentry_transaction(self, transaction) -> None:
        if transaction:
            self._sentry_transaction = transaction

    def record_sentry_data(self, key: str, value: Any) -> None:
        if self._sentry_transaction:
            self._sentry_transaction.set_data(key, value)

    def add_breadcrumb(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        level: str = "info",
        category: str = "session",
    ) -> None:
        if not self._sentry_enabled or sentry_sdk is None:
            return
        sentry_sdk.add_breadcrumb(
            message=message,
            level=level,
            category=category,
            data=data or {},
        )

    def capture_exception(
        self, error: Exception, *, context: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self._sentry_enabled or sentry_sdk is None:
            return
        with sentry_sdk.push_scope() as scope:
            for key, value in (context or {}).items():
                scope.set_extra(key, value)
            scope.set_tag("session_run_id", self.session_run_id)
            scope.set_tag("session_id", self.session_id)
            sentry_sdk.capture_exception(error)

    def finish_sentry_transaction(self, status: str = "ok") -> None:
        if self._sentry_transaction:
            try:
                self._sentry_transaction.set_status(status)
                self._sentry_transaction.finish()
            finally:
                self._sentry_transaction = None

    # --------------------------------------------------------------------- #
    # Langfuse helpers
    # --------------------------------------------------------------------- #
    def _ensure_langfuse_client(self) -> Optional[Langfuse]:
        if not self._langfuse_enabled:
            return None
        if self._langfuse_client:
            return self._langfuse_client
        try:
            self._langfuse_client = Langfuse(
                environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT"),
                sdk_integration="eve-session-instrumentation",
            )
        except Exception as exc:  # pragma: no cover - network/credential dependent
            self.log_event(
                "Failed to initialize Langfuse client",
                level="warning",
                payload={"error": str(exc)},
            )
            self._langfuse_enabled = False
            self._langfuse_client = None
        return self._langfuse_client

    def ensure_langfuse_trace(self) -> None:
        client = self._ensure_langfuse_client()
        if not client or self._langfuse_trace_started:
            return
        metadata = {
            **self.extra_metadata,
            "session_run_id": self.session_run_id,
        }
        client.trace(
            id=self.session_run_id,
            name=self.trace_name,
            user_id=self.user_id,
            session_id=self.session_id,
            metadata=metadata,
            tags=self._langfuse_default_tags,
            input=None,
        )
        self._langfuse_trace_started = True

    def create_langfuse_span(
        self,
        name: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        input_payload: Optional[Any] = None,
        output_payload: Optional[Any] = None,
    ) -> Optional[StatefulSpanClient]:
        self.ensure_langfuse_trace()
        client = self._ensure_langfuse_client()
        if not client:
            return None
        return client.span(
            trace_id=self.session_run_id,
            name=name,
            start_time=_now_utc(),
            metadata=self._merge_metadata(metadata),
            input=input_payload,
            output=output_payload,
        )

    def create_langfuse_generation(
        self,
        name: str,
        *,
        model: Optional[str] = None,
        input_payload: Optional[Any] = None,
        output_payload: Optional[Any] = None,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[StatefulGenerationClient]:
        self.ensure_langfuse_trace()
        client = self._ensure_langfuse_client()
        if not client:
            return None
        return client.generation(
            trace_id=self.session_run_id,
            name=name,
            model=model,
            input=input_payload,
            output=output_payload,
            usage_details=usage,
            metadata=self._merge_metadata(metadata),
            start_time=_now_utc(),
            end_time=_now_utc(),
        )

    def finalize_langfuse(self) -> None:
        if self._langfuse_client:
            try:
                summary = self.summary()
                self._langfuse_client.event(
                    trace_id=self.session_run_id,
                    name="prompt_session_summary",
                    start_time=_now_utc(),
                    metadata=summary,
                )
                self._langfuse_client.flush()
            except Exception as exc:  # pragma: no cover
                self.log_event(
                    "Langfuse flush failed",
                    level="warning",
                    payload={"error": str(exc)},
                )

    def _merge_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = {
            "session_id": self.session_id,
            "session_run_id": self.session_run_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
        }
        merged.update(self.extra_metadata)
        if metadata:
            merged.update(metadata)
        return merged

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #
    def finalize(self, *, success: bool = True) -> Dict[str, Any]:
        summary = self.summary()
        status = "ok" if success else "internal_error"
        self.finish_sentry_transaction(status=status)
        if success:
            self.log_event(
                "Prompt session instrumentation finalized",
                level="info",
                payload={"total_duration_ms": summary["total_duration_ms"]},
                emoji="success",
            )
        else:
            self.log_event(
                "Prompt session instrumentation finalized with errors",
                level="warning",
                payload={"total_duration_ms": summary["total_duration_ms"]},
                emoji="warning",
            )
        self.finalize_langfuse()
        return summary


@contextmanager
def langfuse_span(
    inst: PromptSessionInstrumentation,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Convenience context manager for Langfuse spans."""
    span = inst.create_langfuse_span(name, metadata=metadata)
    try:
        yield span
    finally:
        if span:
            span.end(end_time=_now_utc())
