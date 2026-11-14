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
    """Track the worst inter-node distance excess, and the loop's closure gap, over every substep."""
    seg = np.array(s.built.rope.seg_lengths)
    worst = {"excess": 0.0, "closure": 0.0}
    inner = s._substep

    def substep():
        inner()
        P = s.node_positions()
        worst["excess"] = max(worst["excess"], float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=1) - seg)))
        if SPEC["cord_closed"]:
            worst["closure"] = max(worst["closure"], float(np.linalg.norm(P[-1] - P[0])))

    s._substep = substep
    s.rig.substep = substep
    return worst


def test_segment_count_is_derived_from_the_binding_aperture():
    d = SPEC.segment_derivation
    assert d["binding_aperture"] in SPEC.apertures
    assert d["sagitta_at_segment_length_m"] <= d["clearance_m"] + 1e-12
    assert SPEC.cord_segments == len(RatTrapSim(SPEC).built.rope.body_names)


def test_route_penetrating_a_part_fails_loudly():
    z = SPEC.landmarks["post_lower_z"]
    bad = [[-0.04, 0.0, z - 0.03], [-0.04, 0.0, z + 0.03], [-0.04, -0.03, z + 0.03]]  # up through the lower post's solid part
    with pytest.raises(RatTrapError, match=r"cord segment \d+ penetrates post_lower|post_lower penetrates cord segment \d+"):
        RatTrapSim(load_spec(overrides={"cord_route": bad, "ball_node": 1}))


def test_inextensible_under_maximum_speed_pulls(sim):
    worst = _watch_stretch(sim)
    for node in (_far_node(sim), _strand_node(sim)):
        sim.grab_node(0, node)
        for d in ([0.03, 0, 0], [0, 0.03, 0], [0, 0, -0.03], [-0.03, -0.03, 0.03]):
            sim.move(0, d)  # rig_max_linear_step at rig_linear_speed: the maximum the rig allows
        sim.release(0)
    assert worst["excess"] <= 1e-6, f"cord stretched by {worst['excess']} m"
    assert worst["closure"] <= 2e-4, f"loop closure opened by {worst['closure']} m"


def _strand_node(s: RatTrapSim) -> int:
    """A node on one strand of the loop, out in front of the frame."""
    target = np.array([SPEC.landmarks["dowel_right_x"] + 0.01, -0.08, SPEC["base_thickness"] + 0.0022])
    return int(np.argmin(np.linalg.norm(s.node_positions() - target, axis=1)))


def _far_node(s: RatTrapSim) -> int:
    """The loop's far end (the node farthest out in front)."""
    return int(np.argmin(s.node_positions()[:, 1]))


def test_starts_as_a_closed_loop_around_the_right_dowel(sim):
    assert SPEC["cord_closed"]
    P = sim.node_positions()
    assert np.linalg.norm(P[-1] - P[0]) < 1e-6
    lm = SPEC.landmarks
    # the loop encircles the right dowel: the winding number of its xy projection around the dowel's axis is +-1
    # (taken over the part of the loop at the dowel's height range)
    rel = P[:, :2] - np.array([lm["dowel_right_x"], 0.0])
    ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    assert abs(round((ang[-1] - ang[0]) / (2 * np.pi))) == 1


def test_drag_propagation_without_stretching(sim):
    k = _strand_node(sim)
    sim.grab_node(0, k)
    P0 = sim.node_positions()
    r = sim.move(0, [0.01, 0, 0])
    P1 = sim.node_positions()
    moved = np.linalg.norm(P1 - P0, axis=1)
    assert not r["blocked"] and moved[k] > 0.009
    assert moved[k - 1] > 0.005 and moved[k + 1] > 0.005 and moved[k + 2] > 0.002  # neighbours follow
    assert np.max(np.linalg.norm(np.diff(P1, axis=0), axis=1) - np.array(sim.built.rope.seg_lengths)) <= 1e-6


def test_blocked_move_into_a_rigid_part(sim):
    end = _far_node(sim)
    sim.grab_node(0, end)
    rail_x = SPEC["frame_inner_width"] / 2 + SPEC["rail_thickness"] / 2
    sim.move(0, [rail_x - sim.node_positions()[end][0], 0, 0])
    side = np.sign(sim.node_positions()[end][1])
    last = None
    for _ in range(6):
        last = sim.move(0, [0, -side * 0.03, 0])  # towards and through the rail
        if last["blocked"]:
            break
    assert last["blocked"] and last["requested_fraction_executed"] < 1.0
    y = sim.node_positions()[end][1]
    assert side * y > SPEC["rail_thickness"] / 2 + SPEC["cord_radius"] - TOL  # stopped at the rail's face
    assert sim.max_penetration < TOL


def test_no_self_pass_through():
    # a strand along x, then down and back under it: the free end ends directly below the strand's middle
    route = [[-0.10, 0.08, 0.25], [0.10, 0.08, 0.25], [0.10, 0.08, 0.20], [0.0, 0.08, 0.20]]
    s = RatTrapSim(load_spec(overrides={"cord_route": route, "cord_closed": False, "cord_anchor_start": True}))
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
    size = SPEC["mouse_radius"] if row["object"] == "mouse" else 2 * SPEC["cord_radius"]
    reach = ap.depth / 2 + size + 0.012
    s = None
    for sgn in (1.0, -1.0):  # approach from whichever side the probe can start without touching anything
        axis = sgn * ap.axis
        # overlap is checked before any settling (settling would push an overlapping free probe out of the part)
        spec0 = load_spec(overrides={"settle_after_route_steps": 0})
        s = RatTrapSim(spec0, include_cord=False, probe={"kind": row["object"], "pos": (ap.center - axis * reach).tolist(), "axis": axis.tolist()},
                       check_route=False)
        if not s.penetrations():
            break
        s.close()
        s = None
    if s is None:  # the object cannot even sit on the passage line on either side (a dowel / post runs through it)
        assert row["expect"] == "block", f"{row}: the probe cannot be placed on either side of the aperture"
        return
    s.grab_body(0, "probe")
    s.move_to(0, (ap.center + axis * reach).tolist())
    progressed = float((s.data.xpos[s.model.body("probe").id] - ap.center) @ axis)
    cleared = ap.depth / 2 + size  # the probe has left the aperture (what blocks it afterwards is not this aperture)
    outcome = "pass" if progressed > cleared else "block"
    assert outcome == row["expect"], f"{row}: probe got {outcome} (advanced to {progressed:+.4f} m along the axis; clears at {cleared:.4f})"
    assert s.max_penetration < TOL
    s.close()


SETTLE_STEPS = 250  # N: substeps (1 ms) within which every joint velocity must have decayed


def test_settling(sim):
    sim.grab_node(0, _far_node(sim))
    sim.move(0, [0.0, 0.0, -0.03])
    v0 = sim.max_speed()
    for _ in range(SETTLE_STEPS):
        sim._substep()
    assert sim.max_speed() < 1e-3, f"still moving at {sim.max_speed()} after {SETTLE_STEPS} steps (right after the move: {v0})"


def test_determinism():
    hashes = []
    for _ in range(2):
        s = RatTrapSim(SPEC)
        s.grab_node(0, _strand_node(s))
        s.move(0, [0, -0.02, 0.0])
        s.grab_node(1, s.n_nodes - 1)
        s.move(1, [0.0, 0.01, -0.02])
        s.release(0)
        hashes.append(s.state_hash())
        s.close()
    assert hashes[0] == hashes[1]
