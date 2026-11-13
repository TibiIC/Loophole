"""Record handle commands into a trajectory file (JSONL).

A trajectory is a header line followed by one line per call::

    {"type": "trajectory", "puzzle": "easy", "source": "scripted", "created": ...}
    {"t": 0.0, "tool": "grab", "args": {"handle": 0, "target": "rope", "segment": 20}}
    {"t": 0.41, "tool": "move", "args": {"handle": 0, "dz": 0.1}}

The same format is replayed locally (replay.py), over MCP (replay_mcp.py, used
by the Harbor oracle), and produced from external pose streams (pose_stream.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..sim.env import LoopholeEnv


class Recorder:
    """Wraps an env: forwards calls and appends them (with timestamps) to a trajectory."""

    def __init__(self, env: LoopholeEnv, path: str | Path, source: str = "manual", meta: dict | None = None):
        self.env = env
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.monotonic()
        self._f = open(self.path, "w", buffering=1)
        header = {"type": "trajectory", "puzzle": env.puzzle.name, "puzzle_sha256": env.puzzle.source_sha256, "source": source, "created": time.time()}
        self._f.write(json.dumps({**header, **(meta or {})}) + "\n")

    def act(self, tool: str, args: dict) -> dict:
        result = self.env.act(tool, args)
        self._f.write(json.dumps({"t": round(time.monotonic() - self._t0, 3), "tool": tool, "args": args}) + "\n")
        return result

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_trajectory(path: str | Path, puzzle: str, steps: list[tuple[str, dict]], source: str = "scripted", meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"type": "trajectory", "puzzle": puzzle, "source": source, **(meta or {})}) + "\n")
        for i, (tool, args) in enumerate(steps):
            f.write(json.dumps({"t": float(i), "tool": tool, "args": args}) + "\n")


def load_trajectory(path: str | Path) -> list[tuple[str, dict]]:
    steps = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") in ("trajectory", "header"):
            continue
        steps.append((rec["tool"], rec.get("args", {})))
    return steps
