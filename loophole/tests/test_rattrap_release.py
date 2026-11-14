"""Release after a blocked move must not register a spurious violation (docs/HARNESS_NOTES.md).

Fixture: the first 89 actions of a Gemini 4 Carbon run that ended this way: a move blocked with the cord strained
against the frame, then `release`, which registered a 1.006 mm penetration (tolerance 1 mm) for a few substeps
before the cord settled at 0.1 mm. Under the old semantics (halt on block, every substep audited) the fixture
reproduces that violation, so the test really exercises the defect; under the current ones it does not."""

import json
from pathlib import Path

import pytest

from loophole.puzzles.rattrap.reward import RatTrapEpisode
from loophole.puzzles.rattrap.spec import load_spec

FIX = Path(__file__).parent / "fixtures" / "rattrap_release_after_block.jsonl"
OLD = {"rig_block_mode": "halt", "release_settle_audit": False}


def _replay(overrides):
    ep = RatTrapEpisode(load_spec(overrides=overrides))
    blocked_then_released = 0
    prev_blocked = False
    for r in (json.loads(l) for l in FIX.read_text().splitlines() if l.strip()):
        if ep.done:
            break
        out = ep.act(r["tool"], r["args"])
        blocked_then_released += prev_blocked and r["tool"] == "release"
        prev_blocked = bool(out.get("blocked"))
    return ep, blocked_then_released


def test_old_semantics_reproduce_the_spurious_violation():
    ep, n = _replay(OLD)
    assert n >= 1 and ep.violation and ep.sim.max_penetration > float(ep.spec["penetration_tolerance"])


@pytest.mark.parametrize("overrides", [{}, {"rig_block_mode": "retreat", "release_settle_audit": False}, {"rig_block_mode": "retreat"}],
                         ids=["current (halt + settle audit)", "retreat only", "retreat + settle audit"])
def test_block_against_the_frame_then_release_is_not_a_violation(overrides):
    ep, n = _replay(overrides)
    assert n >= 1  # the sequence does block a move against the frame and release right after
    assert not ep.violation and ep.sim.max_penetration <= float(ep.spec["penetration_tolerance"])


def test_a_large_release_transient_still_counts():
    """The release settle is not a blind spot: a transient beyond release_transient_limit still counts."""
    ep, _ = _replay({"rig_block_mode": "halt", "release_transient_limit": 0.0005})
    assert ep.violation
