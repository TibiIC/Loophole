#!/usr/bin/env python3
"""Cost-to-go roadmap for Alpha 1.0 (Zhang et al. 2020 geometry), seeded around the oracle path.

    .venv/bin/python scripts/build_roadmap_alpha_paper.py      # ~30-60 min, writes roadmap-alpha-1.0-paper.npz (poses only)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loophole.puzzles import paper_alpha as pa  # noqa: E402
from loophole.puzzles.ompl_alpha import parse_cfg  # noqa: E402
from loophole.reward.roadmap import build_roadmap  # noqa: E402

cfg = parse_cfg(pa.paper_dir() / pa.FILES["cfg"])
box = (np.array(cfg["volume"]["min"]) * pa.UNIT_SCALE, np.array(cfg["volume"]["max"]) * pa.UNIT_SCALE)
# edges checked with <= 2 mm motion of any point of the tube; tube sampled around the published solution
info = build_roadmap("alpha-1.0-paper", seed=1, n_tube=5000, n_uniform=2500, n_goal=300, k=12,
                     sigma_t=0.02, sigma_r=0.1, resolution=0.002, box=box)
out = pa.PUZZLE_OUT / "alpha-1.0-paper-roadmap.json"
out.write_text(json.dumps(info, indent=1))
print(json.dumps(info))
