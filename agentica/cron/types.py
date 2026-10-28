# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Core type definitions for the cron scheduling system.

Schedule types (at/every/cron), job status, and run status enums.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


# ============== Schedule Types ==============

class ScheduleKind(str, Enum):
    """Kind of schedule."""
    AT = "at"
    EVERY = "every"
    CRON = "cron"


@dataclass
class AtSchedule:
    """One-time schedule at a specific timestamp."""
    kind: Literal["at"] = "at"
    at_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "at_ms": self.at_ms}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtSchedule":
        return cls(at_ms=data.get("at_ms", 0))

    @classmethod
    def from_datetime(cls, dt: datetime) -> "AtSchedule":
        return cls(at_ms=int(dt.timestamp() * 1000))


@dataclass
class EverySchedule:
    """Interval-based schedule."""
    kind: Literal["every"] = "every"
    interval_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "interval_ms": self.interval_ms}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EverySchedule":
        return cls(interval_ms=data.get("interval_ms", 0))

    @classmethod
    def from_seconds(cls, seconds: int) -> "EverySchedule":
        return cls(interval_ms=seconds * 1000)


@dataclass
class CronSchedule:
    """Cron expression schedule.

    Supports both 5-part (minute precision) and 6-part (second precision) formats:
    - 5-part: "min hour day month weekday" (e.g., "30 7 * * *" = every day at 7:30)
    - 6-part: "sec min hour day month weekday" (e.g., "0 30 7 * * *" = every day at 7:30:00)
    """
    kind: Literal["cron"] = "cron"
    expression: str = ""
    timezone: str = "Asia/Shanghai"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "expression": self.expression,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CronSchedule":
        return cls(
            expression=data.get("expression", ""),
            timezone=data.get("timezone", "Asia/Shanghai"),
        )

    @classmethod
    def at_time(cls, hour: int, minute: int = 0, second: int = 0,
                weekday: str = "*", timezone: str = "Asia/Shanghai") -> "CronSchedule":
        """Create a daily schedule at specific time.

        Examples:
            CronSchedule.at_time(7, 30)        # every day at 7:30
            CronSchedule.at_time(9, 0, 0, "1-5")  # weekdays 9:00
        """
        if second == 0:
            expr = f"{minute} {hour} * * {weekday}"
        else:
            expr = f"{second} {minute} {hour} * * {weekday}"
        return cls(expression=expr, timezone=timezone)


# Union type for all schedule types
Schedule = AtSchedule | EverySchedule | CronSchedule


def schedule_from_dict(data: dict[str, Any]) -> Schedule:
    """Create a Schedule from a dictionary."""
    kind = data.get("kind", "at")
    if kind == "at":
        return AtSchedule.from_dict(data)
    elif kind == "every":
        return EverySchedule.from_dict(data)
    elif kind == "cron":
        return CronSchedule.from_dict(data)
    raise ValueError(f"Unknown schedule kind: {kind}")


# ============== Daily Task Protocol ==============

@dataclass
class DailyTaskSpec:
    """Product-layer task spec for scheduled agent work.

    Cron remains the execution backend; this spec gives CLI/Gateway/product
    surfaces a stable shape for user intent, permissions, and runtime limits.
    """

    name: str
    prompt: str
    schedule: Schedule
    user_id: str = "default"
    workspace: str | None = None
    permissions: dict[str, Any] = field(default_factory=dict)
    deliver: str = "local"
    timezone: str = "Asia/Shanghai"
    timeout_seconds: float = 0.0
    max_retries: int = 0
    retry_delay_ms: int = 60000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "schedule": self.schedule.to_dict(),
            "user_id": self.user_id,
            "workspace": self.workspace,
            "permissions": self.permissions,
            "deliver": self.deliver,
            "timezone": self.timezone,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_ms": self.retry_delay_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DailyTaskSpec":
        return cls(
            name=data.get("name", ""),
            prompt=data.get("prompt", ""),
            schedule=schedule_from_dict(data.get("schedule", {"kind": "at"})),
            user_id=data.get("user_id", "default"),
            workspace=data.get("workspace"),
            permissions=data.get("permissions") or {},
            deliver=data.get("deliver", "local"),
            timezone=data.get("timezone", "Asia/Shanghai"),
            timeout_seconds=data.get("timeout_seconds", 0.0),
            max_retries=data.get("max_retries", 0),
            retry_delay_ms=data.get("retry_delay_ms", 60000),
        )


# ============== Job Status ==============

class JobStatus(str, Enum):
    """Status of a scheduled job."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class RunStatus(str, Enum):
    """Status of a single job run."""
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class TaskRun:
    """Structured history record for one scheduled task execution."""

    run_id: str
    task_id: str
    status: RunStatus
    started_at_ms: int
    ended_at_ms: int
    attempt: int = 1
    result: str = ""
    error: str = ""
    error_type: str = ""
    # Token usage of this run when the runner reported it (see
    # ``CronRunOutput``): input_tokens (all prompt tokens, cached included),
    # output_tokens, cache_read_tokens, cache_write_tokens, api_calls,
    # cost_usd. None for runs from runners that return plain text, and for
    # records written before this field existed.
    usage: dict[str, Any] | None = None

    @property
    def duration_ms(self) -> int:
        if not self.started_at_ms or not self.ended_at_ms:
            return 0
        return max(0, self.ended_at_ms - self.started_at_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "duration_ms": self.duration_ms,
            "attempt": self.attempt,
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
            "usage": self.usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRun":
        return cls(
            run_id=data.get("run_id", ""),
            task_id=data.get("task_id", ""),
            status=RunStatus(data.get("status", RunStatus.FAILED.value)),
            started_at_ms=data.get("started_at_ms", 0),
            ended_at_ms=data.get("ended_at_ms", 0),
            attempt=data.get("attempt", 1),
            result=data.get("result", ""),
            error=data.get("error", ""),
            error_type=data.get("error_type", ""),
            usage=data.get("usage"),
        )


@dataclass
class CronRunOutput:
    """What an ``AgentRunner`` may return instead of plain text.

    ``usage`` uses the ``TaskRun.usage`` keys. Runners that return a bare
    ``str`` still work; their runs record no usage.
    """

    content: str
    usage: dict[str, Any] | None = None


# Delivery targets a job may name. Only "local" exists: the result is written
# to the run history and output files and sent nowhere else. Accepting other
# names ("telegram", "origin") used to store them and then deliver nothing.
SUPPORTED_DELIVER_TARGETS = frozenset({"local"})


def validate_deliver(deliver: str) -> str:
    """Return ``deliver`` if it is a supported target, else raise ValueError."""
    if deliver not in SUPPORTED_DELIVER_TARGETS:
        raise ValueError(
            f"deliver={deliver!r} is not supported: results are only kept locally "
            "(run history and output files); delivery to channels is not implemented. "
            "Use deliver='local' or omit it."
        )
    return deliver
