#!/usr/bin/env python3
"""Replay the Rat Trap reference (Tibi's recorded solve, closed-loop from its fixed waypoints) with training shaping
on, and log every action's reward terms: checkpoints reached, the checkpoint potential and its shaping step, the
aperture-alignment terms, the word and its norm, uncertified substeps. The worked examples in docs/rewards.md.

    python scripts/rattrap_shaping_trace.py [OUT_DIR]      # default runs/rewards-examples
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loophole.puzzles.rattrap.agent_tools import Harness  # noqa: E402
from loophole.puzzles.rattrap.reward import RatTrapEpisode  # noqa: E402
from loophole.puzzles.rattrap.spec import load_spec  # noqa: E402
from loophole.reference.follow_waypoints import follow, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "rewards-examples"
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec()
    ep = RatTrapEpisode(spec, training=True)
    hz = Harness(spec, ep, Path(tempfile.mkdtemp()), vision=False, overrides={})
    rec = open(out / "rattrap-reference-shaping.jsonl", "w", buffering=1)
    act = ep.act

    def logged(tool, a):
        r = act(tool, a)
        rec.write(json.dumps({"i": ep.actions, "tool": tool, "k": ep.tracker.k, "phi": ep._rs.geometry["checkpoint_potential"],
                              "shaping": r.get("shaping"), "parts": r.get("shaping_parts"), "alignment": r.get("alignment"),
                              "word": r["word"], "norm": r["norm"], "solved": r["solved"], "violation": r["violation"],
                              "uncertified": r.get("uncertified_substeps")}, default=float) + "\n")
        return r

    ep.act = logged
    phi0 = ep._rs.geometry["checkpoint_potential"]
    follow(load(str(ROOT / "tasks" / "loophole-rattrap" / "solution" / "waypoints.jsonl")), lambda tool, a: hz.call(tool, a)[0])
    summary = {"phi0": phi0, "actions": ep.actions, "solved": ep.solved, "score": ep.score(), "k": ep.tracker.k,
               "shaping_total": ep.shaping_total, "reached_at": ep.tracker.reached_at, "word0": ep.word0, "norm0": ep.norm0}
    (out / "rattrap-reference-shaping.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
