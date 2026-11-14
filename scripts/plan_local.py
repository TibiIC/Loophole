#!/usr/bin/env python3
"""Local fallback for scripts/modal_plan.py: the same per-seed planning on this machine's cores.

    nice -n 10 .venv/bin/python scripts/plan_local.py --seeds 4 --budget 3600 --workers 4 [--planner rrt-connect]

Writes runs/plan-seeds/seed-<n>.json (same format as the Modal run; `platform.where` = "local") and
runs/plan-seeds/summary.json. Note: the verifier runs on Linux; paths planned here on macOS should be
re-validated in the Linux image before they become the oracle.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.argv += ["--out", str(Path(__file__).resolve().parents[1] / "runs" / "plan-seeds")]
    from plan_reference import main

    main()
