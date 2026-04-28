"""In-process runtime monitoring helpers for API traffic."""

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class RuntimeMonitor:
    """Small in-memory request monitor for lightweight API observability."""

    total_requests: int = 0
    error_requests: int = 0
    total_duration_ms: float = 0.0
    path_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    path_error_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: Lock = field(default_factory=Lock)

    def record(self, *, path: str, status_code: int, duration_ms: float) -> None:
        """Record a request outcome."""
        with self._lock:
            self.total_requests += 1
            self.total_duration_ms += duration_ms
            self.path_counts[path] += 1
            if status_code >= 400:
                self.error_requests += 1
                self.path_error_counts[path] += 1

    def snapshot(self) -> dict[str, object]:
        """Return a serializable monitoring snapshot."""
        with self._lock:
            average_duration_ms = (
                round(self.total_duration_ms / self.total_requests, 2)
                if self.total_requests
                else 0.0
            )
            error_rate = round(self.error_requests / self.total_requests, 4) if self.total_requests else 0.0
            return {
                "total_requests": self.total_requests,
                "error_requests": self.error_requests,
                "error_rate": error_rate,
                "average_duration_ms": average_duration_ms,
                "path_counts": dict(self.path_counts),
                "path_error_counts": dict(self.path_error_counts),
            }


runtime_monitor = RuntimeMonitor()
