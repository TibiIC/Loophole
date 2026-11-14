"""Independent episode verification.

The verifier never trusts rewards or results written by the sidecar, let alone
anything the agent says. It takes the expected puzzle/seed from the task
configuration, then *replays every logged action* in a fresh simulator built
from that puzzle and recomputes the score from the replayed physics. The
logged state hashes are used as a consistency check: any divergence (tampered
log, wrong puzzle/config, nondeterminism) yields reward 0.

    python -m loophole.reward.verify --log episode.jsonl --puzzle easy \
        --out /logs/verifier/reward.json --report /logs/verifier/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import effective_config, load_config
from .roadmap import Roadmap
from ..sim.env import PHYSICS_TOOLS, LoopholeEnv
from ..sim.spec import load_puzzle, resolve_puzzle


def _zero(reason: str, **extra) -> dict:
    return {"reward": 0.0, "success": 0, "actions_used": 0, "violations": 0, "log_consistent": 0, "reason": reason, **extra}


def verify_episode(log_path: str | Path, puzzle: str, seed: int = 0, config_path: str | None = None) -> dict:
    log_path = Path(log_path)
    if not log_path.exists():
        return _zero(f"no episode log at {log_path}")
    records = []
    for i, line in enumerate(log_path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            return _zero(f"malformed log line {i}")
    if not records or records[0].get("type") != "header":
        return _zero("log has no header")
    header = records[0]
    pz = load_puzzle(resolve_puzzle(puzzle))
    if header.get("puzzle_sha256") != pz.source_sha256:
        return _zero("episode was played on a different puzzle file", expected=pz.source_sha256, got=header.get("puzzle_sha256"))
    if int(header.get("seed", -1)) != seed:
        return _zero("episode used an unexpected seed")
    cfg = load_config(config_path)
    eff = effective_config(cfg, pz)
    if header.get("config") != json.loads(json.dumps(eff.model_dump())):
        return _zero("episode used a different environment configuration")
    rm = Roadmap.for_puzzle(pz) if (pz.goal.type == "separation" and eff.reward.roadmap) else None
    if header.get("roadmap_sha256") != (rm.sha256 if rm else None):
        return _zero("episode used a different cost-to-go roadmap")

    env = LoopholeEnv(pz, config=cfg, seed=seed)
    try:
        if env.initial_hash != header.get("initial_state_hash"):
            return _zero("initial state mismatch (nondeterministic build?)")
        n_logged = 0
        for rec in records[1:]:
            if rec.get("type") != "action":
                continue
            n_logged += 1
            if rec.get("tool") not in PHYSICS_TOOLS:
                return _zero(f"unknown tool in log: {rec.get('tool')}")
            try:
                env.act(rec["tool"], rec.get("args", {}))
            except Exception as e:  # a logged action must replay cleanly
                return _zero(f"action {n_logged} failed on replay: {e}")
            if env.state_hash() != rec.get("state_hash"):
                return _zero(f"state hash mismatch after action {n_logged}")
            if env.actions_used != rec.get("i"):
                return _zero("action numbering mismatch")
        final = env.finalize()
    finally:
        env.close()
    logged_final = next((r for r in reversed(records) if r.get("type") == "final"), None)
    return {
        "reward": float(final["score"]),
        "success": int(bool(final["success"])),
        "actions_used": int(final["actions_used"]),
        "violations": int(final["violations"]["count"]),
        "log_consistent": 1,
        "sidecar_agrees": int(logged_final is not None and abs(logged_final["score"] - final["score"]) < 1e-9),
        "reason": final["done_reason"],
        "final": final,
    }


def main(argv: list[str] | None = None) -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--puzzle", default=os.environ.get("LOOPHOLE_PUZZLE"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("LOOPHOLE_SEED", "0")))
    ap.add_argument("--config", default=os.environ.get("LOOPHOLE_CONFIG"))
    ap.add_argument("--out", default="/logs/verifier/reward.txt")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    if not a.puzzle:
        ap.error("--puzzle (or LOOPHOLE_PUZZLE) is required")
    try:
        res = verify_episode(a.log, a.puzzle, a.seed, a.config)
    except Exception as e:  # never crash without writing a reward
        res = _zero(f"verifier error: {type(e).__name__}: {e}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.out.endswith(".txt"):  # Harbor reward.txt: one float in [0, 1]
        Path(a.out).write_text(f"{float(res['reward']):.6f}\n")
    else:
        numeric = {k: v for k, v in res.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        Path(a.out).write_text(json.dumps(numeric))
    if a.report:
        Path(a.report).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("reward", "success", "actions_used", "violations", "log_consistent", "reason")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
