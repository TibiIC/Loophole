"""Hand-written reference solutions for the procedural placeholder puzzles.

Regenerate the stored trajectory files with `python scripts/make_reference.py`.
"""

from __future__ import annotations

SCRIPTS: dict[str, list[tuple[str, dict]]] = {
    # Lift the bight tip until the whole bight is above ring and loop, then carry the ring away.
    "easy": (
        [("grab", {"handle": 0, "target": "rope", "segment": 20})]
        + [("move", {"handle": 0, "dz": 0.1})] * 4
        + [("grab", {"handle": 1, "target": "ring"})]
        + [("move", {"handle": 1, "dx": 0.1})] * 9
    ),
}
