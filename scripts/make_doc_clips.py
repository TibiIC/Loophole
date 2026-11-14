#!/usr/bin/env python3
"""Short illustrative clips for the docs site (MP4 + palette-optimised GIF).

Two stages, so the expensive replay runs once and the cheap retiming can be iterated:

    # 1. replay an episode log and keep every captured frame (lossless) + which action it belongs to
    .venv/bin/python scripts/make_doc_clips.py capture <episode.jsonl> --out <dir>/<name> [--views iso,top]
    # 2. retime / compose captured streams into docs/media/<clip>.{mp4,gif}
    .venv/bin/python scripts/make_doc_clips.py alpha-reference <dir>/oracle -o docs/media/clip-alpha-reference
    .venv/bin/python scripts/make_doc_clips.py alpha-contrast <dir>/sbMy6Ld <dir>/Kt5DJ3h -o docs/media/clip-alpha-contrast
    .venv/bin/python scripts/make_doc_clips.py trim <in.mp4> -o docs/media/clip-x [--start S --duration D]
    # Rat Trap (its code is on the rattrap branch: pass a checkout of it)
    .venv/bin/python scripts/make_doc_clips.py rattrap-capture <run>/commands.jsonl --rattrap-root <checkout> \
        --out <dir>/kimi --width 480 --height 360
    .venv/bin/python scripts/make_doc_clips.py rattrap-failure <dir>/kimi -o docs/media/clip-rattrap-failure \
        --name kimi-k3-1-pro --first 45

docs/media/CLIPS.md records which run and which actions each committed clip shows.

Retiming is per action: every action gets an output duration and its captured frames are
resampled into it. The Alpha reference runs fast outside the tunnel (A-B clearance >= 3 mm)
and slows inside it; the contrast clip gives every action the same duration on both sides,
so the two runs stay aligned by action index.

Meshes: the Alpha geometry is read from LOOPHOLE_ALPHA_PAPER_DIR (set it to an existing
fetch, e.g. ../Loophole/.alpha_paper); point LOOPHOLE_MESH_CACHE outside the repo too.
Nothing derived from the meshes is written here except rendered frames.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_FPS = 30  # MP4 frame rate
GIF_FPS = 12
GIF_WIDTH = 600
TUNNEL_CLEARANCE_M = 0.003  # docs/RESULTS.md: tunnel = clearance < 3 mm


# ---------- capture ----------

def capture(a) -> None:
    from PIL import Image

    sys.path.insert(0, str(ROOT / "scripts"))
    from render_video import VIEWS, load_episode
    from loophole.sim.env import EpisodeOver, LoopholeEnv
    from loophole.sim.rig import RigError

    views = a.views.split(",")
    puzzle, cfg, seed, steps, logged, header = load_episode(a.episode, None, None)
    env = LoopholeEnv(puzzle, config=cfg, seed=seed, start_pose=(header or {}).get("start_pose"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    w, h = a.width * len(views), a.height
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25",
         "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p",
         str(a.out.with_suffix(".mkv"))],
        stdin=subprocess.PIPE,
    )
    frames: list[int] = []  # action index (0 = start pose) of every captured frame
    cur = {"i": 0}

    def frame() -> None:
        tiles = []
        for v in views:
            imgs, _ = env.render_views(**VIEWS[v], width=a.width, height=a.height)
            tiles.append(np.asarray(Image.open(io.BytesIO(imgs[0][1])).convert("RGB")))
        ff.stdin.write(np.ascontiguousarray(np.concatenate(tiles, axis=1), dtype=np.uint8).tobytes())
        frames.append(cur["i"])

    inner = env._substep

    def substep() -> None:
        inner()
        if env.substeps % a.every == 0:
            frame()

    env._substep = substep
    env.rig.substep = substep
    frame()
    actions = []
    for tool, args in steps:
        if env.done:
            break
        cur["i"] = env.actions_used + 1
        n0 = len(frames)
        try:
            res = env.act(tool, args)
        except (RigError, EpisodeOver, ValueError, TypeError):
            continue
        if len(frames) == n0:  # grab / release / blocked-at-once: still show one frame of it
            frame()
        actions.append({"i": cur["i"], "tool": tool, "blocked": bool(res.get("blocked")),
                        "min_distance_m": env._last_min_distance})
    cur["i"] = env.actions_used + 1
    frame()
    ff.stdin.close()
    ff.wait()
    final = env.finalize()
    env.close()
    meta = {"episode": str(a.episode), "views": views, "width": w, "height": h, "every_substeps": a.every,
            "frame_action": frames, "actions": actions,
            "final": {k: final.get(k) for k in ("score", "success", "actions_used", "done_reason")}}
    a.out.with_suffix(".json").write_text(json.dumps(meta))
    print(json.dumps({"out": str(a.out), "frames": len(frames), **meta["final"]}))


# ---------- compose ----------

def load_stream(base: Path):
    meta = json.loads(base.with_suffix(".json").read_text())
    return meta


def decode(base: Path, meta: dict, indices: list[int]):
    """Yield the captured frames at `indices` (non-decreasing), decoding the stream once."""
    w, h = meta["width"], meta["height"]
    src = base if base.suffix in (".mp4", ".mkv") else base.with_suffix(".mkv")
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(src), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    pos, buf = -1, None
    for idx in indices:
        while pos < idx:
            raw = p.stdout.read(n)
            if len(raw) < n:
                break
            buf, pos = raw, pos + 1
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.kill()  # the rest of the stream is not needed
    p.stdout.close()
    p.wait()


def frames_by_action(meta: dict) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for f, i in enumerate(meta["frame_action"]):
        out.setdefault(i, []).append(f)
    return out


def resample(frames: list[int], n_out: int) -> list[int]:
    if n_out <= 0:
        return []
    return [frames[min(len(frames) - 1, int(k * len(frames) / n_out))] for k in range(n_out)]


def font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_label(img: np.ndarray, text: str, where: str = "tl", size: int = 18, strong: bool = False) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = font(size)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    tw, th = x1 - x0, y1 - y0
    pad = 6
    W, H = im.size
    x = pad if where in ("tl", "bl") else (W - tw) // 2
    y = pad if where.startswith("t") else H - th - 3 * pad
    bg = (20, 20, 20) if strong else (255, 255, 255)
    fg = (255, 255, 255) if strong else (0, 0, 0)
    d.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=bg)
    d.text((x - x0, y - y0), text, fill=fg, font=f)
    return np.asarray(im)


def encode(frames_iter, w: int, h: int, out: Path, crf: int = 18) -> None:
    """Full-quality MP4 at OUT_FPS, then a two-pass palette GIF (<= GIF_WIDTH wide, GIF_FPS)."""
    mp4 = out.with_suffix(".mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(OUT_FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(mp4)],
        stdin=subprocess.PIPE,
    )
    for fr in frames_iter:
        ff.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    gif_from_mp4(mp4, out.with_suffix(".gif"))


def gif_from_mp4(mp4: Path, gif: Path, start: float | None = None, duration: float | None = None) -> None:
    cut = (["-ss", f"{start}"] if start is not None else []) + (["-t", f"{duration}"] if duration is not None else [])
    flt = f"fps={GIF_FPS},scale='min({GIF_WIDTH},iw)':-2:flags=lanczos"
    pal = gif.with_suffix(".palette.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cut, "-i", str(mp4), "-vf",
                    f"{flt},palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cut, "-i", str(mp4), "-i", str(pal), "-lavfi",
                    f"{flt}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-loop", "0", str(gif)], check=True)
    pal.unlink()


def tunnel_window(meta: dict, gap: int = 1) -> tuple[int, int]:
    """First and last action of the longest run of tunnel actions (clearance < 3 mm), allowing
    `gap` non-tunnel actions inside it, so isolated grazes elsewhere don't stutter the timing."""
    hits = [act["i"] for act in meta["actions"] if act["min_distance_m"] is not None and act["min_distance_m"] < TUNNEL_CLEARANCE_M]
    if not hits:
        return (0, -1)
    runs, start, prev = [], hits[0], hits[0]
    for i in hits[1:]:
        if i - prev > gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return max(runs, key=lambda r: r[1] - r[0])


def plan_reference(meta: dict, fast_s: float, slow_s: float) -> list[tuple[int, int]]:
    """(action index, output frame count): fast outside the tunnel window, slow inside it."""
    lo, hi = tunnel_window(meta)
    plan = [(0, int(0.5 * OUT_FPS))]
    for act in meta["actions"]:
        plan.append((act["i"], max(1, round((slow_s if lo <= act["i"] <= hi else fast_s) * OUT_FPS))))
    last = max(meta["frame_action"])
    plan.append((last, int(1.0 * OUT_FPS)))
    return plan


def alpha_reference(a) -> None:
    meta = load_stream(a.stream)
    groups = frames_by_action(meta)
    plan = plan_reference(meta, a.fast, a.slow)
    lo, hi = tunnel_window(meta)
    idx, labels = [], []
    for i, n in plan:
        sel = resample(groups[i], n)
        idx += sel
        labels += [i] * len(sel)
    total = meta["final"]["actions_used"]

    def frames():
        for fr, i in zip(decode(a.stream, meta, idx), labels):
            text = f"reference solve · action {min(i, total)}/{total}" + ("  · narrow passage, slowed" if lo <= i <= hi else "")
            yield draw_label(fr, text)

    encode(frames(), meta["width"], meta["height"], a.out)
    report = {"out": str(a.out), "seconds": round(len(idx) / OUT_FPS, 2), "fast_s_per_action": a.fast,
              "slow_s_per_action": a.slow, "slow_window": [lo, hi], "actions": total}
    a.out.with_suffix(".timing.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def end_label(meta: dict) -> str:
    f = meta["final"]
    if f["success"]:
        return f"solved, {f['actions_used']} actions"
    if f["done_reason"] == "action budget exhausted":
        return f"unsolved, budget exhausted ({f['actions_used']} actions)"
    return f"unsolved, stopped at {f['actions_used']} actions"


def alpha_contrast(a) -> None:
    L, R = load_stream(a.left), load_stream(a.right)
    gl, gr = frames_by_action(L), frames_by_action(R)
    nl, nr = max(L["frame_action"]), max(R["frame_action"])
    per = max(1, round(a.per_action * OUT_FPS))
    span = max(nl, nr)
    if a.max_action:
        span = min(span, a.max_action)
    plan = [(0, int(0.5 * OUT_FPS))] + [(i, per) for i in range(1, span + 1)] + [(span, int(a.hold * OUT_FPS))]

    def side(groups, n_last, i, n):
        return resample(groups[min(i, n_last)], n)

    il, ir, ai = [], [], []
    for i, n in plan:
        il += side(gl, nl, i, n)
        ir += side(gr, nr, i, n)
        ai += [i] * n
    # one view per run (the left tile of each capture)
    wl, wr = L["width"] // len(L["views"]), R["width"] // len(R["views"])
    h = L["height"]

    def frames():
        for fl, fr, i in zip(decode(a.left, L, il), decode(a.right, R, ir), ai):
            tl = draw_label(np.ascontiguousarray(fl[:, :wl]), f"{a.left_name} · action {min(i, L['final']['actions_used'])}")
            tr = draw_label(np.ascontiguousarray(fr[:, :wr]), f"{a.right_name} · action {min(i, R['final']['actions_used'])}")
            if i > L["final"]["actions_used"]:
                tl = draw_label(tl, end_label(L), where="bc", size=22, strong=True)
            if i > R["final"]["actions_used"]:
                tr = draw_label(tr, end_label(R), where="bc", size=22, strong=True)
            sep = np.full((h, 4, 3), 255, np.uint8)
            yield np.concatenate([tl, sep, tr], axis=1)

    encode(frames(), wl + 4 + wr, h, a.out)
    report = {"out": str(a.out), "seconds": round(len(ai) / OUT_FPS, 2), "s_per_action": a.per_action,
              "left": {"name": a.left_name, **L["final"]}, "right": {"name": a.right_name, **R["final"]},
              "shown_through_action": span}
    a.out.with_suffix(".timing.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def rattrap_capture(a) -> None:
    """Replay a Rat Trap agent run (commands.jsonl) through RatTrapEpisode; render only actions in [first, last].

    The Rat Trap code lives on the rattrap branch: pass its checkout as --rattrap-root."""
    import time

    sys.path.insert(0, str(a.rattrap_root.resolve()))
    import loophole

    if not Path(loophole.__file__).resolve().is_relative_to(a.rattrap_root.resolve()):
        sys.exit(f"loophole imported from {loophole.__file__}, not from --rattrap-root")
    from loophole.puzzles.rattrap.reward import RatTrapEpisode
    from loophole.puzzles.rattrap.spec import load_spec

    lines = [json.loads(l) for l in a.commands.read_text().splitlines() if l.strip()]
    hdr = next((r for r in lines if r.get("type") == "header"), {})
    cmds = [(r["tool"], r["args"]) for r in lines if r.get("type") != "header"]
    ep = RatTrapEpisode(load_spec(overrides=hdr.get("overrides") or None))
    sim = ep.sim
    last = min(a.last or len(cmds), a.max_actions or len(cmds), len(cmds))
    w, h = a.width * 2, a.height
    ff = None
    if a.first <= last and not a.no_render:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25",
             "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p",
             str(a.out.with_suffix(".mkv"))],
            stdin=subprocess.PIPE,
        )
    frames: list[int] = []
    cur = {"i": 0, "render": False, "n": 0}

    def frame() -> None:
        tiles = [sim.render("iso", distance=0.45, width=a.width, height=a.height),
                 sim.render("front", distance=0.4, width=a.width, height=a.height)]
        ff.stdin.write(np.ascontiguousarray(np.concatenate(tiles, axis=1), dtype=np.uint8).tobytes())
        frames.append(cur["i"])

    inner = sim._substep

    def substep() -> None:
        inner()
        cur["n"] += 1
        if cur["render"] and cur["n"] % a.every == 0:
            frame()

    sim._substep = substep
    if getattr(sim.rig, "substep", None) is not None:
        sim.rig.substep = substep
    actions, t0 = [], time.perf_counter()
    for k, (tool, args) in enumerate(cmds[:last], 1):
        if ep.done:
            break
        cur["i"], cur["render"] = k, ff is not None and k >= a.first
        if cur["render"] and not frames:
            frame()  # the state just before the window
        n0, ts = len(frames), time.perf_counter()
        out = ep.act(tool, args)
        if cur["render"] and len(frames) == n0:
            frame()
        actions.append({"i": k, "tool": tool, "args": args, "blocked": bool(out.get("blocked")),
                        "seconds": round(time.perf_counter() - ts, 3)})
    if ff is not None:
        ff.stdin.close()
        ff.wait()
    meta = {"commands": str(a.commands), "views": ["iso", "front"], "width": w, "height": h, "every_substeps": a.every,
            "frame_action": frames, "actions": actions, "wall_seconds": round(time.perf_counter() - t0, 1),
            "final": {"score": ep.score(), "success": ep.solved, "actions_used": ep.actions, "violation": ep.violation}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(meta))
    print(json.dumps({"out": str(a.out), "frames": len(frames), "actions": len(actions),
                      "blocked": sum(x["blocked"] for x in actions), "wall_seconds": meta["wall_seconds"]}))


def rattrap_failure(a) -> None:
    """The window of `seconds` with the most blocked moves, every action the same duration."""
    meta = load_stream(a.stream)
    groups = frames_by_action(meta)
    acts = [x for x in meta["actions"] if x["i"] in groups]
    n = max(1, round(a.seconds / a.per_action))
    if a.first:
        start = next(j for j, x in enumerate(acts) if x["i"] >= a.first)
    else:
        start = max(range(max(1, len(acts) - n + 1)), key=lambda j: (sum(x["blocked"] for x in acts[j:j + n]), -j))
    win = acts[start:start + n]
    per = max(1, round(a.per_action * OUT_FPS))
    idx, lab = [], []
    for x in win:
        sel = resample(groups[x["i"]], per)
        idx += sel
        lab += [x] * len(sel)
    total = meta["final"]["actions_used"]

    def frames():
        for fr, x in zip(decode(a.stream, meta, idx), lab):
            img = draw_label(fr, f"{a.name} · action {x['i']}/{total} · {x['tool']}")
            if x["blocked"]:
                img = draw_label(img, "move blocked", where="bc", size=22, strong=True)
            yield img

    encode(frames(), meta["width"], meta["height"], a.out)
    report = {"out": str(a.out), "seconds": round(len(idx) / OUT_FPS, 2), "s_per_action": a.per_action,
              "actions": [win[0]["i"], win[-1]["i"]], "blocked_in_clip": sum(x["blocked"] for x in win),
              "blocked_in_run": sum(x["blocked"] for x in meta["actions"]), "run_actions": total, **{"final": meta["final"]}}
    a.out.with_suffix(".timing.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def reference_action_times(recording: Path, commands: Path, settle_s: float, t0: float) -> list[tuple[float, float]]:
    """Video-time interval of every command of the converted reference solve.

    The reference video replays the teleop recording; each grab / release there corresponds to one
    grab_node / release command (same count, same order). Events are exact; the moves between two
    events are spread evenly over the time between them. Each event adds `settle_s` of rig settling
    to the video's sim time, and the video starts at `t0`."""
    rec = [json.loads(l) for l in recording.open() if l.strip()]
    ev = [r for r in rec if r.get("type") in ("grab", "release")]
    t_end = max(r.get("s", 0) for r in rec if "s" in r) / 1000.0 + t0 + settle_s * len(ev)
    cmds = [json.loads(l) for l in commands.open() if l.strip()]
    cmds = [c for c in cmds if c.get("tool")]
    n_ev = sum(c["tool"] in ("grab_node", "release") for c in cmds)
    if n_ev != len(ev):
        sys.exit(f"{n_ev} grab/release commands but {len(ev)} recorded events")
    tau = [t0 + r["s"] / 1000.0 + settle_s * k for k, r in enumerate(ev)]
    out: list[tuple[float, float] | None] = []
    k, pending = -1, []
    seg_start = t0

    def flush(seg_end: float) -> None:
        n = len(pending)
        for j, slot in enumerate(pending):
            out[slot] = (seg_start + (seg_end - seg_start) * j / n, seg_start + (seg_end - seg_start) * (j + 1) / n)
        pending.clear()

    for c in cmds:
        if c["tool"] in ("grab_node", "release"):
            k += 1
            flush(tau[k])
            out.append((tau[k], tau[k] + settle_s))
            seg_start = tau[k] + settle_s
        else:
            out.append(None)
            pending.append(len(out) - 1)
    flush(t_end)
    return out


def rattrap_contrast(a) -> None:
    """Human reference (existing 2-up video, left view) beside an agent run (captured stream, left view),
    aligned by action index: every action the same duration on both sides."""
    times = reference_action_times(a.recording, a.commands, a.settle, a.t0)
    n_ref = len(times)
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                       "stream=width,height,nb_read_frames", "-of", "json", str(a.reference)],
                                      capture_output=True, text=True, check=True).stdout)["streams"][0]
    rw, rh, rn = probe["width"], probe["height"], int(probe["nb_read_frames"])
    meta = load_stream(a.agent)
    groups = frames_by_action(meta)
    n_ag = meta["final"]["actions_used"]
    blocked = {x["i"] for x in meta["actions"] if x["blocked"]}
    per = max(1, round(a.per_action * OUT_FPS))
    span = max(n_ref, n_ag)
    plan = [(0, int(0.5 * OUT_FPS))] + [(i, per) for i in range(1, span + 1)] + [(span + 1, int(a.hold * OUT_FPS))]

    def ref_frame(i: int, u: float) -> int:
        if i == 0:
            t = a.t0
        elif i > n_ref:
            return rn - 1
        else:
            t0_, t1_ = times[i - 1]
            t = t0_ + u * (t1_ - t0_)
        return int(min(rn - 1, max(0, round(t * a.video_fps / a.speed))))

    ir, ia, ai = [], [], []
    for i, n in plan:
        ir += [ref_frame(i, k / n) for k in range(n)]
        ia += resample(groups.get(i) or groups[min(groups, key=lambda g: abs(g - i))], n)
        ai += [i] * n
    order = sorted(range(len(ir)), key=lambda j: ir[j])  # decode() needs non-decreasing indices
    ref_frames: dict[int, np.ndarray] = {}
    for j, fr in zip(order, decode(a.reference, {"width": rw, "height": rh}, [ir[j] for j in order])):
        ref_frames.setdefault(ir[j], fr.copy())
    aw = meta["width"] // len(meta["views"])
    h = meta["height"]

    def frames():
        from PIL import Image

        for k, (fa, i) in enumerate(zip(decode(a.agent, meta, ia), ai)):
            rf = ref_frames[ir[k]][:, : rw // 2]
            rf = np.asarray(Image.fromarray(np.ascontiguousarray(rf)).resize((aw, h), Image.LANCZOS))
            left = draw_label(rf, f"human reference · action {min(max(i, 0), n_ref)}/{n_ref}")
            left = draw_label(left, "human reference, solved", where="bc", size=22, strong=True)
            right = draw_label(np.ascontiguousarray(fa[:, :aw]), f"{a.agent_name} · action {min(i, n_ag)}/{n_ag}"
                               + ("  · move blocked" if i in blocked else ""))
            right = draw_label(right, "agent, 0/20 checkpoints", where="bc", size=22, strong=True)
            yield np.concatenate([left, np.full((h, 4, 3), 255, np.uint8), right], axis=1)

    encode(frames(), 2 * aw + 4, h, a.out, crf=a.crf)
    report = {"out": str(a.out), "seconds": round(len(ai) / OUT_FPS, 2), "s_per_action": a.per_action,
              "reference_actions": n_ref, "agent_actions": n_ag, "reference_source_frames": rn,
              "agent": meta["final"], "agent_blocked": len(blocked)}
    a.out.with_suffix(".timing.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def trim(a) -> None:
    mp4 = a.out.with_suffix(".mp4")
    cut = (["-ss", f"{a.start}"] if a.start is not None else []) + (["-t", f"{a.duration}"] if a.duration is not None else [])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cut, "-i", str(a.src), "-an", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    gif_from_mp4(mp4, a.out.with_suffix(".gif"))
    print(json.dumps({"out": str(a.out)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("episode", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--views", default="iso,top")
    c.add_argument("--width", type=int, default=480)
    c.add_argument("--height", type=int, default=360)
    c.add_argument("--every", type=int, default=40, help="capture a frame every N substeps (1 ms each)")
    c.set_defaults(fn=capture)
    r = sub.add_parser("alpha-reference")
    r.add_argument("stream", type=Path)
    r.add_argument("-o", "--out", type=Path, required=True)
    r.add_argument("--fast", type=float, default=0.1, help="output seconds per action outside the tunnel window")
    r.add_argument("--slow", type=float, default=0.35, help="output seconds per action inside the tunnel window")
    r.set_defaults(fn=alpha_reference)
    k = sub.add_parser("alpha-contrast")
    k.add_argument("left", type=Path)
    k.add_argument("right", type=Path)
    k.add_argument("-o", "--out", type=Path, required=True)
    k.add_argument("--left-name", default="left")
    k.add_argument("--right-name", default="right")
    k.add_argument("--per-action", type=float, default=0.1, help="output seconds per action, both sides")
    k.add_argument("--hold", type=float, default=1.5, help="seconds held on the last frame")
    k.add_argument("--max-action", type=int, default=0, help="stop after this action index (0 = the longer run's end)")
    k.set_defaults(fn=alpha_contrast)
    q = sub.add_parser("rattrap-capture")
    q.add_argument("commands", type=Path)
    q.add_argument("--rattrap-root", type=Path, required=True, help="a checkout of the rattrap branch")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--first", type=int, default=1, help="first action to render")
    q.add_argument("--last", type=int, default=0, help="last action to replay (0 = all)")
    q.add_argument("--max-actions", type=int, default=0)
    q.add_argument("--no-render", action="store_true", help="physics only: per-action blocked flags and timing")
    q.add_argument("--width", type=int, default=640)
    q.add_argument("--height", type=int, default=480)
    q.add_argument("--every", type=int, default=20, help="capture a frame every N physics steps")
    q.set_defaults(fn=rattrap_capture)
    f = sub.add_parser("rattrap-failure")
    f.add_argument("stream", type=Path)
    f.add_argument("-o", "--out", type=Path, required=True)
    f.add_argument("--name", default="agent")
    f.add_argument("--per-action", type=float, default=0.25, help="output seconds per action")
    f.add_argument("--seconds", type=float, default=7.0, help="clip length")
    f.add_argument("--first", type=int, default=0, help="first action (0 = the window with the most blocked moves)")
    f.set_defaults(fn=rattrap_failure)
    v = sub.add_parser("rattrap-contrast")
    v.add_argument("reference", type=Path, help="the reference solve's 2-up MP4 (left half is used)")
    v.add_argument("recording", type=Path, help="loophole/reference/rattrap-solve.jsonl")
    v.add_argument("commands", type=Path, help="loophole/reference/rattrap-solve.commands.jsonl")
    v.add_argument("agent", type=Path, help="captured agent stream (rattrap-capture --out)")
    v.add_argument("-o", "--out", type=Path, required=True)
    v.add_argument("--agent-name", default="agent")
    v.add_argument("--per-action", type=float, default=0.08)
    v.add_argument("--hold", type=float, default=2.0)
    v.add_argument("--crf", type=int, default=16)
    v.add_argument("--speed", type=float, default=20.0, help="the reference video's sim-time speed-up")
    v.add_argument("--video-fps", type=float, default=30.0)
    v.add_argument("--t0", type=float, default=0.3, help="sim time of the reference video's first frame")
    v.add_argument("--settle", type=float, default=0.2, help="rig settle time added per grab / release (s)")
    v.set_defaults(fn=rattrap_contrast)
    t = sub.add_parser("trim")
    t.add_argument("src", type=Path)
    t.add_argument("-o", "--out", type=Path, required=True)
    t.add_argument("--start", type=float)
    t.add_argument("--duration", type=float)
    t.set_defaults(fn=trim)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
