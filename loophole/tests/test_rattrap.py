"""Rat Trap scaffold: cord and rigid-structure physics (geometry from loophole/puzzles/rattrap/rattrap.yaml)."""

import numpy as np
import pytest

from loophole.puzzles.rattrap.sim import RatTrapSim
from loophole.puzzles.rattrap.spec import RatTrapError, load_spec

SPEC = load_spec()
TOL = SPEC["penetration_tolerance"]


@pytest.fixture
def sim():
    s = RatTrapSim(SPEC)
    yield s
    s.close()


def _watch_stretch(s: RatTrapSim) -> dict:
    """Track the worst inter-node distance excess over every substep."""
    seg = np.array(s.built.rope.seg_lengths)
    worst = {"excess": 0.0}
    inner = s._substep

    def substep():
        inner()
        P = s.node_positions()
        worst["excess"] = max(worst["excess"], float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=1) - seg)))

    s._substep = substep
    s.rig.substep = substep
    return worst


def test_segment_count_is_derived_from_the_tightest_aperture():
    d = SPEC.segment_derivation
    assert d["tightest_aperture"] == min(SPEC.apertures, key=lambda n: SPEC.apertures[n].radius)
    assert d["sagitta_at_segment_length_m"] <= d["clearance_m"] + 1e-12
    assert SPEC.cord_segments == len(RatTrapSim(SPEC).built.rope.body_names)


def test_route_penetrating_a_part_fails_loudly():
    bad = [[-0.05, 0.0, 0.09], [-0.05, 0.0, 0.22]]  # straight up through the lower and upper bars' solid spans
    with pytest.raises(RatTrapError, match=r"cord segment \d+ penetrates (bar_lower|bar_upper)|(bar_lower|bar_upper) penetrates cord segment \d+"):
        RatTrapSim(load_spec(overrides={"cord_route": bad, "ball_node": 1}))


def test_inextensible_under_maximum_speed_pulls(sim):
    worst = _watch_stretch(sim)
    for node in (sim.n_nodes - 1, sim.n_nodes // 3):
        sim.grab_node(0, node)
        for d in ([0.03, 0, 0], [0, 0.03, 0], [0, 0, -0.03], [-0.03, -0.03, 0.03]):
            sim.move(0, d)  # rig_max_linear_step at rig_linear_speed: the maximum the rig allows
        sim.release(0)
    assert worst["excess"] <= 1e-6, f"cord stretched by {worst['excess']} m"


def test_drag_propagation_without_stretching(sim):
    k = 26  # on the strand between the two bars
    sim.grab_node(0, k)
    P0 = sim.node_positions()
    r = sim.move(0, [0, -0.01, 0])
    P1 = sim.node_positions()
    moved = np.linalg.norm(P1 - P0, axis=1)
    assert not r["blocked"] and moved[k] > 0.009
    assert moved[k - 1] > 0.005 and moved[k + 1] > 0.005 and moved[k + 2] > 0.002  # neighbours follow
    assert np.max(np.linalg.norm(np.diff(P1, axis=0), axis=1) - np.array(sim.built.rope.seg_lengths)) <= 1e-6


def test_blocked_move_into_a_rigid_part(sim):
    end = sim.n_nodes - 1
    sim.grab_node(0, end)
    rail_x = SPEC["frame_inner_width"] / 2 + SPEC["rail_thickness"] / 2
    sim.move(0, [rail_x - sim.node_positions()[end][0], 0, 0])
    last = None
    for _ in range(6):
        last = sim.move(0, [0, -0.03, 0])
        if last["blocked"]:
            break
    assert last["blocked"] and last["requested_fraction_executed"] < 1.0
    y = sim.node_positions()[end][1]
    assert y > SPEC["rail_thickness"] / 2 + SPEC["cord_radius"] - TOL  # stopped at the rail's front face
    assert sim.max_penetration < TOL


def test_no_self_pass_through():
    # a strand along x, then down and back under it: the free end ends directly below the strand's middle
    route = [[-0.10, 0.08, 0.25], [0.10, 0.08, 0.25], [0.10, 0.08, 0.20], [0.0, 0.08, 0.20]]
    s = RatTrapSim(load_spec(overrides={"cord_route": route, "ball_node": 3}))
    P = s.node_positions()
    strand = [k for k in range(s.n_nodes) if abs(P[k][2] - 0.25) < 1e-6]
    hold = min(strand, key=lambda k: abs(P[k][0]))
    s.grab_node(1, hold)  # the strand is held still
    end = s.n_nodes - 1
    s.grab_node(0, end)
    last = None
    for _ in range(5):  # fire the end straight up through the held strand at maximum speed
        last = s.move(0, [0, 0, 0.03])
        if last["blocked"]:
            break
    assert last["blocked"]
    assert s.node_positions()[end][2] < 0.25  # never crossed the strand
    assert s.max_penetration < TOL
    s.close()


@pytest.mark.parametrize("row", SPEC["clearance_table"], ids=lambda r: f"{r['aperture']}-{r['object']}")
def test_clearance_table(row):
    """Drive a cord piece / the ball through every declared aperture; compare with the YAML's table."""
    ap = SPEC.apertures[row["aperture"]]
    reach = (ap.depth / 2 + (SPEC["ball_radius"] if row["object"] == "ball" else 2 * SPEC["cord_radius"]) + 0.012)
    start = ap.center - ap.axis * reach
    s = RatTrapSim(SPEC, include_cord=False, ring_free=False, probe={"kind": row["object"], "pos": start.tolist(), "axis": ap.axis.tolist()})
    s.grab_body(0, "probe")
    r = s.move_to(0, (ap.center + ap.axis * reach).tolist())
    progressed = float((s.data.xpos[s.model.body("probe").id] - ap.center) @ ap.axis)
    outcome = "pass" if (not r.get("blocked") and progressed > reach - 0.002) else "block"
    assert outcome == row["expect"], f"{row}: probe got {outcome} (advanced to {progressed:+.4f} m along the axis)"
    assert s.max_penetration < TOL
    s.close()


SETTLE_STEPS = 250  # N: substeps (1 ms) within which every joint velocity must have decayed


def test_settling(sim):
    sim.grab_node(0, sim.n_nodes - 1)
    sim.move(0, [0.0, 0.0, -0.03])
    v0 = sim.max_speed()
    for _ in range(SETTLE_STEPS):
        sim._substep()
    assert sim.max_speed() < 1e-3, f"still moving at {sim.max_speed()} after {SETTLE_STEPS} steps (right after the move: {v0})"


def test_determinism():
    hashes = []
    for _ in range(2):
        s = RatTrapSim(SPEC)
        s.grab_node(0, 26)
        s.move(0, [0, -0.02, 0.0])
        s.grab_node(1, s.n_nodes - 1)
        s.move(1, [0.0, 0.01, -0.02])
        s.release(0)
        hashes.append(s.state_hash())
        s.close()
    assert hashes[0] == hashes[1]
