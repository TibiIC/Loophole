"""Environment configuration (rig limits, audit thresholds, reward weights).

Puzzle-specific geometry lives in the puzzle YAML; everything here is about how
the environment *runs* a puzzle. Defaults are in ``default_config.yaml`` and can
be overridden with a YAML file (``LOOPHOLE_CONFIG``) or keyword overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.yaml")


class RigConfig(BaseModel):
    num_handles: int = Field(2, ge=1, le=2)
    max_linear_step: float = 0.10  # m per move request
    max_angular_step: float = 0.6  # rad per move request
    linear_speed: float = 0.25  # m/s while executing a move
    angular_speed: float = 1.5  # rad/s while executing a move
    settle_steps: int = 100  # substeps after each move/grab/release
    max_wait_steps: int = 4000
    lag_limit: float = 0.015  # m; mocap stops advancing when the grasped body lags this far
    angle_lag_limit: float = 0.35  # rad
    grasp_tolerance: float = 0.02  # m; a grasp point must be this close to the body's surface
    # The rig is "careful": it also stalls when any contact or the rope anchor is strained
    # beyond these levels, which sit below the audit's violation thresholds.
    stall_penetration: float = 0.002
    stall_anchor_drift: float = 0.0025
    workspace_min: list[float] = [-1.5, -1.5, 0.0]
    workspace_max: list[float] = [1.5, 1.5, 1.5]
    weld_solref: list[float] = [0.01, 1.0]
    weld_solimp: list[float] = [0.9, 0.95, 0.001, 0.5, 2.0]


class AuditConfig(BaseModel):
    penetration_threshold: float = 0.004  # m; deeper contact => violation (passing through needs >= ~11 mm)
    anchor_drift_threshold: float = 0.005  # m; rope end leaving its anchor => violation
    max_substep_displacement: float = 0.003  # m per substep (3 m/s at dt=1 ms); faster bodies could skip through thin parts
    max_records: int = 50


class RewardComponentConfig(BaseModel):
    type: str
    weight: float = 1.0
    enabled: bool = True
    # False: a training signal only - its per-step rewards are logged, but it does not
    # contribute to the episode score the Harbor verifier writes.
    in_score: bool = True
    params: dict[str, Any] = {}


class RewardConfig(BaseModel):
    # Cost-to-go roadmap reward (work in progress, off by default): loads the puzzle's
    # roadmap-*.npz and adds the `cost_to_go_shaping` and `progress` components.
    roadmap: bool = False
    auto_roadmap_components: bool = True  # with roadmap: add cost_to_go_shaping + progress
    components: list[RewardComponentConfig] = [
        RewardComponentConfig(type="base", weight=1.0, params={"alpha": 0.5}),
        RewardComponentConfig(type="topology_shaping", weight=0.2, enabled=False, params={"gamma": 1.0}),
    ]


class TopologyConfig(BaseModel):
    enabled: bool = True  # track crossings and expose the summary in get_state
    expose_in_state: bool = True
    primary: str = "rope"  # word fed to goal_distance: "rope" (reduced), "rope_raw" or "target"


class RenderConfig(BaseModel):
    width: int = 480
    height: int = 360


class EnvConfig(BaseModel):
    reset_settle_steps: int = 2000
    rig: RigConfig = RigConfig()
    audit: AuditConfig = AuditConfig()
    reward: RewardConfig = RewardConfig()
    topology: TopologyConfig = TopologyConfig()
    render: RenderConfig = RenderConfig()


def _deep_update(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> EnvConfig:
    data: dict = {}
    if DEFAULT_CONFIG_PATH.exists():
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}
    path = path or os.environ.get("LOOPHOLE_CONFIG")
    if path:
        data = _deep_update(data, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        data = _deep_update(data, overrides)
    return EnvConfig.model_validate(data)


def effective_config(base: EnvConfig, puzzle) -> EnvConfig:
    """Apply a puzzle's own `config` overrides (simulator units) on top of `base`."""
    if not getattr(puzzle, "config", None):
        return base
    return EnvConfig.model_validate(_deep_update(base.model_dump(), puzzle.config))
