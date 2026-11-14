"""Gymnasium API for the Alpha puzzle (training only; evaluation is the Harbor/MCP task).

    from loophole.gym import AlphaEnv, make_vec_env
    env = AlphaEnv(start="perturb")                      # training starts, never the canonical one
    obs, info = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    venv = make_vec_env(8, start="curriculum", curriculum_distance=0.3)

* ``start``: perturb | curriculum | interlocked | flipped | bank (see loophole/training/starts.py);
  ``tie`` = both | canonical | flipped selects how the loops are tied (linking number -1 / +1).
* ``reset(seed, options)`` samples a start (``loophole.training.starts``), resets the simulator,
  applies optional domain randomisation, and grabs tube B (one action, as an agent must).
  ``options`` may override ``start``, ``curriculum_distance``, ``curriculum_source`` and set
  ``log_path`` (per-episode simulator log, the format scripts/export_traces.py reads).
* ``step(action)``: ``action`` in [-1, 1]^6 = (translation, rotation vector) scaled by the rig
  caps (0.1 m, 0.2 rad), applied as one `move` about the grasp point. Returns the per-step
  shaping reward (the config's training components, primarily the C-space tunnel potential),
  ``terminated`` (free, or a violation), ``truncated`` (action budget), and ``info``. At the end
  of an episode ``info["score"]`` is the verifier's score (``episode_score``, same code path).
* ``mode="eval"`` uses the canonical start with no randomisation: the held-out test. Training
  (``mode="train"``, the default) rejects ``start="canonical"``.

Domain randomisation (training only, all off by default): ``orientation`` (observations and
actions are expressed in a random frame per episode; cameras get a random azimuth),
``friction`` (Coulomb mu ~ U(friction_range); the task is frictionless), ``stiffness`` (contact
time constant scaled by U(stiffness_range)), ``obs_noise`` (Gaussian on pose / pixels / depth).
Physics randomisation changes the dynamics, so such episodes are not verifier-replayable; their
logs record the parameters under ``header.training``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import load_config
from .sim.env import EpisodeOver, LoopholeEnv
from .sim.geometry import quat_from_rotvec, quat_mul, quat_to_mat
from .sim.rig import RigError
from .training import randomize
from .training.starts import StartSampler

CONFIG = Path(__file__).parent / "configs" / "alpha_training.yaml"


def _mat_to_euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of geometry.quat_from_euler_xyz (extrinsic X, Y, Z: R = Rz Ry Rx)."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


class AlphaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, puzzle: str = "alpha-1.0-paper", config: str | Path | None = None, mode: str = "train",
                 start: str = "perturb", curriculum_distance: float = 0.5, curriculum_source: str = "oracle",
                 observations: tuple[str, ...] = ("pose",), image_size: tuple[int, int] = (96, 96),
                 randomize_orientation: bool = False, randomize_friction: bool = False, friction_range=(0.0, 0.5),
                 randomize_stiffness: bool = False, stiffness_range=(0.5, 2.0), obs_noise: float = 0.0,
                 sampler_kwargs: dict | None = None, render_mode: str | None = None, tie: str = "both",
                 roadmap: str = "auto"):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        if mode == "train" and start == "canonical":
            raise ValueError("the canonical start is the held-out test: use mode='eval' to evaluate on it")
        self.env = LoopholeEnv(puzzle, config=load_config(config or CONFIG))
        if self.env.checker is None or self.env.roadmap is None:
            raise ValueError("AlphaEnv needs a separation puzzle with reward.roadmap: true")
        # training roadmap (evaluation roadmap + start-bank corridors): cost-to-go from bank starts.
        # "auto" = use it in training mode if it exists; evaluation always keeps the task's roadmap.
        from .puzzles import paper_alpha as pa
        from .reward.roadmap import Roadmap
        from .training.bank import TRAIN_ROADMAP_FILE

        train_rm = pa.PUZZLE_OUT / TRAIN_ROADMAP_FILE.format(name=puzzle)
        use_train = mode == "train" and (roadmap == "train" or (roadmap == "auto" and train_rm.exists()))
        if use_train:
            self.env.roadmap = Roadmap.load(train_rm)
            self.env.reset()
        self.roadmap_kind = "train" if use_train else "eval"
        self.mode, self.start = mode, ("canonical" if mode == "eval" else start)
        self.q0 = self.env.target_pose().copy()  # canonical start (constructed with start_pose=None)
        self.sampler = None if mode == "eval" else StartSampler(
            self.env, self.q0, mode=start, curriculum_distance=curriculum_distance, curriculum_source=curriculum_source,
            chirality=tie, **(sampler_kwargs or {}))
        self.obs_keys = tuple(observations)
        if not set(self.obs_keys) <= {"pose", "rgb", "depth", "segmentation"}:
            raise ValueError("observations: any of pose, rgb, depth, segmentation")
        self.H, self.W = image_size[1], image_size[0]
        rand = mode == "train"
        self.rand = {"orientation": rand and randomize_orientation, "friction": rand and randomize_friction,
                     "stiffness": rand and randomize_stiffness, "obs_noise": float(obs_noise) if rand else 0.0}
        self.friction_range, self.stiffness_range = friction_range, stiffness_range
        self._defaults = randomize.defaults(self.env)
        self.max_lin = self.env.cfg.rig.max_linear_step
        self.max_ang = self.env.cfg.rig.max_angular_step
        self.T = self.env.puzzle.max_actions
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        obs = {}
        if "pose" in self.obs_keys:
            obs["pose"] = spaces.Box(-np.inf, np.inf, (7,), np.float64)
            obs["proprio"] = spaces.Box(-np.inf, np.inf, (4,), np.float64)  # blocked, actions left, contact force, grip lag
        if "rgb" in self.obs_keys:
            obs["rgb"] = spaces.Box(0, 255, (self.H, self.W, 3), np.uint8)
        if "depth" in self.obs_keys:
            obs["depth"] = spaces.Box(0.0, np.inf, (self.H, self.W), np.float32)
        if "segmentation" in self.obs_keys:
            obs["segmentation"] = spaces.Box(0, 4, (self.H, self.W), np.uint8)
        self.observation_space = spaces.Dict(obs)
        self.render_mode = render_mode
        self.Q = np.array([1.0, 0.0, 0.0, 0.0])  # observation frame (orientation randomisation)
        self.cam_azimuth = 135.0
        self._last_result: dict = {}

    # ------------------------------------------------------------------ helpers
    def _apply_physics(self, rng: np.random.Generator) -> dict:
        params = {}
        if self.rand["friction"]:
            params["friction"] = round(float(rng.uniform(*self.friction_range)), 5)
        if self.rand["stiffness"]:
            params["contact_timeconst_scale"] = round(float(rng.uniform(*self.stiffness_range)), 5)
        randomize.apply_physics(self.env, params, self._defaults)  # rounded values: the log replays exactly
        return params

    def _to_obs_frame(self, pose: np.ndarray) -> np.ndarray:
        R = quat_to_mat(self.Q)
        return np.concatenate([R @ pose[:3], quat_mul(self.Q, pose[3:])])

    def _obs(self) -> dict:
        e, rng, sig = self.env, self.np_random, self.rand["obs_noise"]
        out = {}
        if "pose" in self.obs_keys:
            p = self._to_obs_frame(e.target_pose())
            if sig:
                p[:3] += rng.normal(0.0, sig, 3)
                p[3:] = quat_mul(quat_from_rotvec(rng.normal(0.0, sig, 3)), p[3:])
            c = e.get_contacts(max_contacts=0)
            out["pose"] = p
            out["proprio"] = np.array([float(bool(self._last_result.get("blocked"))), (self.T - e.actions_used) / self.T,
                                       float(np.linalg.norm(c["total_contact_force"])),
                                       float(c["grip_lag"][0]["lag_m"]) if c["grip_lag"] else 0.0])
        mods = [k for k in ("rgb", "depth", "segmentation") if k in self.obs_keys]
        if mods:
            cam = e._camera("iso", self.cam_azimuth, None, None, None)
            arrays, _ = e._renderer.render_modalities(e.data, cam, mods, self.W, self.H)
            if "rgb" in arrays:
                img = arrays["rgb"].astype(np.float32)
                if sig:
                    img += rng.normal(0.0, 255 * sig, img.shape)
                out["rgb"] = np.clip(img, 0, 255).astype(np.uint8)
            if "depth" in arrays:
                dep = arrays["depth"].astype(np.float32)
                out["depth"] = dep + (rng.normal(0.0, sig, dep.shape).astype(np.float32) if sig else 0.0)
            if "segmentation" in arrays:
                lut = e._seg_labels()
                a = arrays["segmentation"]
                out["segmentation"] = lut[np.where(a < 0, len(lut) - 1, a)]
        return out

    def _info(self) -> dict:
        e = self.env
        info = {"actions_used": e.actions_used, "success": e.success, "violations": e.audit.count,
                "blocked": bool(self._last_result.get("blocked")), "rewards": dict(e.composer.last_step or {}),
                "cost_to_go": e.ctg, "start": self.start_info}
        if e.done:
            final = e.finalize()
            info.update(score=final["score"], done_reason=final["done_reason"])
        return info

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = dict(options or {})
        rng = self.np_random
        e = self.env
        if self.mode == "eval":
            e.start_pose, self.start_info = None, {"start_mode": "canonical"}
        else:
            override = {k: options[k] for k in ("curriculum_distance", "curriculum_source") if k in options}
            if "tie" in options:
                override["chirality"] = options["tie"]
            if "start" in options:
                if options["start"] == "canonical":
                    raise ValueError("the canonical start is never used in training mode")
                override["mode"] = options["start"]
            q, self.start_info = self.sampler.sample(rng, **override)
            e.start_pose = q
        self.Q = _random_rotation(rng) if self.rand["orientation"] else np.array([1.0, 0.0, 0.0, 0.0])
        self.cam_azimuth = float(rng.uniform(0, 360)) if self.rand["orientation"] else 135.0
        physics = self._apply_physics(rng)
        e.episode_meta = None if self.mode == "eval" else {
            **self.start_info, "physics": physics, "orientation_frame": np.round(self.Q, 9).tolist() if self.rand["orientation"] else None,
            "obs_noise": self.rand["obs_noise"]}
        e.reset(seed=seed, log_path=options.get("log_path"))
        self._last_result = e.act("grab", {"handle": 0, "target": e.puzzle.target})
        return self._obs(), self._info()

    def step(self, action):
        e = self.env
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        R = quat_to_mat(self.Q).T  # observation frame -> world
        dpos = R @ (a[:3] * self.max_lin)
        rv = R @ (a[3:] * self.max_ang)
        roll, pitch, yaw = _mat_to_euler_xyz(quat_to_mat(quat_from_rotvec(rv)))
        try:
            self._last_result = e.act("move", {"handle": 0, "dx": dpos[0], "dy": dpos[1], "dz": dpos[2],
                                               "droll": roll, "dpitch": pitch, "dyaw": yaw, "frame": "world"})
            reward = e.last_step_reward
        except (RigError, EpisodeOver):
            self._last_result, reward = {}, 0.0
        terminated = bool(e.success or e.audit.count > 0)
        truncated = bool(e.done and not terminated)
        return self._obs(), float(reward), terminated, truncated, self._info()

    def render(self):
        imgs, _ = self.env.render_views("iso", azimuth=self.cam_azimuth, width=self.W, height=self.H)
        from io import BytesIO

        from PIL import Image

        return np.asarray(Image.open(BytesIO(imgs[0][1])).convert("RGB"))

    def close(self):
        self.env.close()


def make_vec_env(num_envs: int, asynchronous: bool = True, **kwargs) -> gym.vector.VectorEnv:
    """`num_envs` independent AlphaEnvs (one simulator each; subprocesses when asynchronous)."""
    fns = [lambda: AlphaEnv(**kwargs) for _ in range(num_envs)]
    return gym.vector.AsyncVectorEnv(fns) if asynchronous else gym.vector.SyncVectorEnv(fns)


gym.register(id="Loophole/Alpha-v0", entry_point="loophole.gym:AlphaEnv")
