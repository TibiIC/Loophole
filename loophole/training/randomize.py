"""Physics domain randomisation (training only), shared by loophole.gym and the trace exporter."""

from __future__ import annotations

import numpy as np


def pair_geoms(env) -> np.ndarray:
    """Geoms of the fixed and the free pieces (the contact pair)."""
    m = env.model
    bodies = {m.body(b).id for b in env.built.frame_bodies + env.built.object_bodies}
    return np.flatnonzero([m.geom_bodyid[g] in bodies for g in range(m.ngeom)])


def defaults(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = env.model
    return m.geom_condim.copy(), m.geom_friction.copy(), m.geom_solref.copy()


def apply_physics(env, params: dict | None, base: tuple) -> None:
    """Restore `base` contact parameters, then apply {"friction": mu, "contact_timeconst_scale": k}."""
    m = env.model
    condim, friction, solref = base
    m.geom_condim[:], m.geom_friction[:], m.geom_solref[:] = condim, friction, solref
    params = params or {}
    g = pair_geoms(env)
    if "friction" in params:
        m.geom_condim[g] = 3
        m.geom_friction[g, 0] = float(params["friction"])
    if "contact_timeconst_scale" in params:
        m.geom_solref[g, 0] = solref[g, 0] * float(params["contact_timeconst_scale"])
