from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from packaging.version import Version

from eve.agent.session.debug_logger import SessionDebugger

try:
    import sentry_sdk
except ImportError:  # pragma: no cover - sentry may not be installed in tests
    sentry_sdk = None  # type: ignore

try:
    from langfuse import Langfuse
    from langfuse.client import (
        StatefulGenerationClient,
        StatefulSpanClient,
        StatefulTraceClient,
    )
except ImportError:  # pragma: no cover - langfuse optional
    Langfuse = None  # type: ignore
    StatefulGenerationClient = None  # type: ignore
    StatefulSpanClient = None  # type: ignore
    StatefulTraceClient = None  # type: ignore


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        trace_input: Optional[Any] = None,
    ):
        self.session_id = session_id
        self.session_run_id = session_run_id or str(uuid.uuid4())
        self.user_id = user_id
        self.agent_id = agent_id
        self.trace_name = trace_name
        self.extra_metadata = metadata or {}
        self.debugger = SessionDebugger(session_id=session_id, enabled=debug_enabled)
        self._langfuse_trace_input = trace_input
        self._langfuse_trace_output: Optional[Any] = None
        self._start_clock = time.perf_counter()

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
        self._langfuse_trace: Optional[StatefulTraceClient] = None
        self._langfuse_supports_prompt = False

    # --------------------------------------------------------------------- #
    # Logging helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _coerce_int(value: Optional[Any]) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Optional[Any]) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def build_langfuse_usage_payload(
        cls,
        *,
        prompt_tokens: Optional[Any] = None,
        completion_tokens: Optional[Any] = None,
        cached_prompt_tokens: Optional[Any] = None,
        cached_completion_tokens: Optional[Any] = None,
        cache_creation_input_tokens: Optional[Any] = None,
        cache_read_input_tokens: Optional[Any] = None,
        prompt_cost: Optional[Any] = None,
        completion_cost: Optional[Any] = None,
        total_cost: Optional[Any] = None,
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Optional[Dict[str, int]],
        Optional[Dict[str, float]],
    ]:
        pt = cls._coerce_int(prompt_tokens)
        ct = cls._coerce_int(completion_tokens)
        total_tokens = None
        if pt is not None or ct is not None:
            total_tokens = (pt or 0) + (ct or 0)

        usage: Dict[str, Any] = {}
        if pt is not None:
            usage["prompt_tokens"] = pt
        if ct is not None:
            usage["completion_tokens"] = ct
        if total_tokens is not None:
            usage["total_tokens"] = total_tokens
        cached_pt = cls._coerce_int(cached_prompt_tokens)
        cached_ct = cls._coerce_int(cached_completion_tokens)
        if cached_pt is not None:
            usage["cached_prompt_tokens"] = cached_pt
        if cached_ct is not None:
            usage["cached_completion_tokens"] = cached_ct
        total_cost_float = cls._coerce_float(total_cost)
        if total_cost_float is not None:
            usage["total_cost"] = total_cost_float

        usage_details: Dict[str, int] = {}
        if pt is not None:
            usage_details["input"] = pt
        if ct is not None:
            usage_details["output"] = ct
        if total_tokens is not None:
            usage_details["total"] = total_tokens
        cache_creation = cls._coerce_int(cache_creation_input_tokens)
        cache_read = cls._coerce_int(cache_read_input_tokens)
        if cache_creation is not None:
            usage_details["cache_creation_input_tokens"] = cache_creation
        if cache_read is not None:
            usage_details["cache_read_input_tokens"] = cache_read

        cost_details: Dict[str, float] = {}
        prompt_cost_float = cls._coerce_float(prompt_cost)
        completion_cost_float = cls._coerce_float(completion_cost)
        if prompt_cost_float is not None:
            cost_details["prompt_cost"] = prompt_cost_float
        if completion_cost_float is not None:
            cost_details["completion_cost"] = completion_cost_float
        if total_cost_float is not None:
            cost_details["total_cost"] = total_cost_float

        return (
            usage or None,
            usage_details or None,
            cost_details or None,
        )

    def log_event(
        self,
        message: str,
        *,
        level: str = "info",
        payload: Optional[Dict[str, Any]] = None,
        emoji: Optional[str] = None,
    ) -> None:
        level_name = _normalize_level(level)
        if self.debugger:
            self.debugger.log(
                message,
                payload,
                level=level_name.lower(),
                emoji=emoji,
            )
        log = self._structured_logger
        if payload:
            log = log.bind(payload=payload)
        log.log(level_name, message)

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
        if self.debugger:
            self.debugger.log(
                f"Update: {update_type}",
                payload or {},
                level="debug",
                emoji="update",
            )
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

    def set_trace_input(self, payload: Any) -> None:
        self._langfuse_trace_input = payload
        if self._langfuse_trace:
            self._langfuse_trace.update(input=payload)

    def set_trace_output(self, payload: Any) -> None:
        self._langfuse_trace_output = payload
        if self._langfuse_trace:
            self._langfuse_trace.update(output=payload)

    def _refresh_langfuse_metadata(self) -> None:
        if self._langfuse_trace:
            self._langfuse_trace.update(metadata=self._merge_metadata({}))

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
        self._bind_logger()
        self._refresh_langfuse_metadata()

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
                "id": r.id,
                "name": r.name,
                "duration_ms": r.duration_ms,
                "status": r.status,
                "metadata": r.metadata,
                "error": r.error,
                "parent_id": r.parent_id,
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
        trace_id = self.session_run_id
        try:
            uuid.UUID(trace_id)
        except (ValueError, TypeError):
            trace_id = str(uuid.uuid4())
        transaction = sentry_sdk.start_transaction(
            name=transaction_name, op=op, sampled=sampled, trace_id=trace_id
        )
        if transaction:
            transaction.set_tag("session_id", self.session_id)
            transaction.set_tag("session_run_id", self.session_run_id)
            if self.user_id:
                transaction.set_tag("user_id", self.user_id)
            if self.agent_id:
                transaction.set_tag("agent_id", self.agent_id)
            environment = os.getenv("ENV", os.getenv("ENVIRONMENT", "local"))
            transaction.set_tag("environment", environment)
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
        metadata = self._merge_metadata({"session_run_id": self.session_run_id})
        self._langfuse_trace = client.trace(
            id=self.session_run_id,
            name=self.trace_name,
            user_id=self.user_id,
            session_id=self.session_id,
            metadata=metadata,
            tags=self._langfuse_default_tags,
            input=self._langfuse_trace_input,
            output=self._langfuse_trace_output,
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
        usage: Optional[Dict[str, Any]] = None,
        usage_details: Optional[Dict[str, int]] = None,
        cost_details: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        prompt: Optional[Any] = None,
    ) -> Optional[StatefulGenerationClient]:
        self.ensure_langfuse_trace()
        client = self._ensure_langfuse_client()
        if not client:
            return None
        log_payload: Dict[str, Any] = {
            "trace_id": self.session_run_id,
            "name": name,
            "model": model,
            "usage": usage,
            "usage_details": usage_details,
            "cost_details": cost_details,
        }
        if metadata:
            log_payload["metadata"] = metadata
        if input_payload is not None:
            log_payload["input_payload"] = input_payload
        if output_payload is not None:
            log_payload["output_payload"] = output_payload
        # logger.debug(f"Langfuse generation payload: {log_payload}")
        return client.generation(
            trace_id=self.session_run_id,
            name=name,
            model=model,
            input=input_payload,
            output=output_payload,
            usage=usage,
            usage_details=usage_details,
            cost_details=cost_details,
            metadata=self._merge_metadata(metadata),
            start_time=start_time or _now_utc(),
            end_time=end_time or _now_utc(),
        )

    def finalize_langfuse(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if not self._langfuse_client:
            return
        summary = summary or self.summary()
        try:
            if not self._langfuse_trace_started:
                self.ensure_langfuse_trace()
            if summary and not self._langfuse_trace_output:
                self.set_trace_output(summary)
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
        self.finalize_langfuse(summary=summary)
        return summary

    def _detect_langfuse_prompt_support(self) -> bool:
        try:
            import langfuse

            version = getattr(langfuse, "__version__", None)
            if version is None:
                return False
            return Version(version) >= Version("2.7.3")
        except Exception:
            return False


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
