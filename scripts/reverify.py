#!/usr/bin/env python3
"""Re-score an eval run's episode logs inside a given sim image (the simulator's exact code).

    .venv/bin/python scripts/reverify.py runs/20261114-080705 --image loophole-sim:eval-20261114

Harbor's verifier is rebuilt from the working tree; if that tree changed during an eval, the
verifier no longer matches the simulator that played the episodes and every log is rejected.
This replays each trial's log (verifier/episode.jsonl) with loophole.reward.verify inside
`--image` (e.g. the image of the eval's own sidecars), writes verifier/reverified.json per
trial, and rewrites summary.md with the re-verified scores (noting the image used).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval import summary  # noqa: E402

CODE = """
import json, sys
from loophole.reward.verify import verify_episode
r = verify_episode('/ep/episode.jsonl', 'alpha-1.0-paper', 0, '/opt/loophole/loophole/configs/alpha_training.yaml')
print(json.dumps(r, default=str))
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    cfg = json.loads((a.run_dir / "config.json").read_text())
    rows = []
    for name in ("oracle", "noop", "stop-early", "opencode"):
        for trial in sorted((a.run_dir / name).glob("*__*")):
            ep = trial / "verifier" / "episode.jsonl"
            if not ep.exists():
                continue
            out = subprocess.run(["docker", "run", "--rm", "--network", "none", "-v", f"{ep.parent.resolve()}:/ep:ro",
                                  "--entrypoint", "python", a.image, "-c", CODE], capture_output=True, text=True)
            rep = json.loads(out.stdout.strip().splitlines()[-1])
            (trial / "verifier" / "reverified.json").write_text(json.dumps({"image": a.image, **rep}, indent=1))
            final = rep.get("final") or {}
            rows.append({"run": name, "trial": trial.name, "reward": rep["reward"], "success": bool(rep.get("success")),
                         "actions": rep.get("actions_used"), "violations": rep.get("violations"), "reason": rep.get("reason"),
                         "log_consistent": rep.get("log_consistent"), "ctg": final.get("cost_to_go") or {}})
            print(json.dumps(rows[-1]), flush=True)
    text = summary(a.run_dir, cfg["model"], rows)
    note = (f"\n> Scores re-verified with `scripts/reverify.py --image {a.image}` (the simulator image the episodes ran on); "
            "see each trial's `verifier/reverified.json`. Harbor's own `reward.txt` is kept unchanged next to it.\n")
    (a.run_dir / "summary.md").write_text(text + note)
    print(text + note)


if __name__ == "__main__":
    main()
