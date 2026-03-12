from __future__ import annotations

"""JSONL event logging and replay utilities for observability/repro."""

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class JsonlEventLogger:
    """Append-only JSONL logger for stream events."""

    def __init__(self, path: str | Path):
        """Open or create a JSONL file for event logging."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, kind: str, payload: Any) -> None:
        """Write one event record with serialized payload."""
        line = {
            "kind": kind,
            "payload": _serialize(payload),
        }
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close underlying file handle."""
        self._fh.close()


def replay(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate JSONL events from disk in original order."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _serialize(obj: Any) -> Any:
    """Convert common runtime objects to JSON-safe representations."""
    if dataclasses.is_dataclass(obj):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "min": float(np.min(obj)) if obj.size else None,
            "max": float(np.max(obj)) if obj.size else None,
            "mean": float(np.mean(obj)) if obj.size else None,
        }
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj
