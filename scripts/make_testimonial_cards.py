#!/usr/bin/env python3
"""The failure cards re-cut as one-star product reviews, plus a checkpoint card; one MP4 and a PNG per card.

    python3 scripts/make_testimonial_cards.py --body-font SourceSans3[wght].ttf \
        --body-italic-font SourceSans3-Italic[wght].ttf

Run it with an environment that has MuJoCo (the checkpoint card renders the Rat Trap). Same background and fonts as
scripts/make_failure_cards.py.

Cards 1-4 (4 s each): a one-star rating, a quote cut verbatim from the committed transcript (the model's visible reply
at the given turn; an ellipsis wherever the cut is inside a sentence; the model's own *emphasis* in italic), and one
attribution line, "<model, turn> · <annotation>". About two seconds in, a hand-drawn red underline is drawn under the
annotation only (it starts at the rendered width of the prefix), left to right, over half a second.

Card 5 (6 s): the same one-star rating, an iso still of the Rat Trap at its start (rendered at 1920x1080, cropped to the
puzzle) on which the 20 checkpoints land as circled red X stamps, 0.15 s apart; once all have landed, "0 of 20 reached
· verified purchase" and "— every frontier model we tested". A checkpoint is a whole cord configuration; its stamp is
placed at the cord node the human grabbed or released at the grab / release event the checkpoint was taken at
(`meta.kept_boundary_indices` into the recording's events), at that node's position in the checkpoint's state,
projected with the render camera. The placements are written to card-5-stamps.json.

Every number is computed from the run's transcript, result.json or rescore.json, or from the checkpoint file.
Output: docs/media/testimonial-cards.mp4 (1920x1080, 30 fps) and docs/media/testimonial-cards/card-N.png.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_failure_cards import (BG, DEEPSEEK, FG, FPS, H, KIMI, MARGIN_X, MUTED, RUNS, ROOT, TEXT_W, W,  # noqa: E402
                                quote, tokens, wrap)

AMBER = (245, 181, 40)
RED = (236, 58, 52)
CARD_S, UNDERLINE_AT, UNDERLINE_S = 4.0, 2.0, 0.5
CP_CARD_S, CP_FIRST_AT, CP_GAP, CP_CAPTION_AT = 6.0, 0.5, 0.15, 3.6


# ---------- data from the committed logs ----------

def excerpt(sentence: str, start: str, end: str) -> str:
    """The verbatim span of `sentence` from `start` through `end`, with an ellipsis on each side that cuts it."""
    i = sentence.index(start)
    j = sentence.index(end, i) + len(end)  # pass a closing emphasis marker as part of `end`, e.g. "post*"
    lead = "" if i == 0 else "…"
    trail = "" if sentence[j:].strip() == "" else "…"
    return lead + sentence[i:j] + trail


def actions_used_at(run: Path, turn: int) -> int:
    for line in (run / "transcript.jsonl").open():
        r = json.loads(line)
        if r.get("type") == "tool" and r.get("turn") == turn and r.get("name") == "get_state":
            return int(r["result"]["actions_used"])
    raise SystemExit(f"{run.name}: no get_state at turn {turn}")


def all_runs_checkpoints() -> tuple[int, int]:
    best, K = 0, 20
    for f in list(RUNS.glob("*/rescore.json")) + list(RUNS.glob("*/result.json")):
        d = json.loads(f.read_text())
        if d.get("checkpoints_reached") is not None:
            best = max(best, int(d["checkpoints_reached"]))
        K = int(d.get("K") or d.get("checkpoints") or K)
    return best, K


# ---------- review cards ----------

def star(draw, cx, cy, r, filled):
    pts = [(cx + (r if k % 2 == 0 else r * 0.42) * math.sin(math.pi * k / 5),
            cy - (r if k % 2 == 0 else r * 0.42) * math.cos(math.pi * k / 5)) for k in range(10)]
    if filled:
        draw.polygon(pts, fill=AMBER)
    else:
        draw.polygon(pts, outline=AMBER, width=max(3, int(r * 0.09)))


def draw_lines(d, lines, y, lh, color):
    for line in lines:
        x = MARGIN_X
        for j, (word, f, space) in enumerate(line):
            if j and space:
                x += d.textlength(" ", font=f)
            d.text((x, y), word, font=f, fill=color)
            x += d.textlength(word, font=f)
        y += lh
    return y


def review_card(fonts, text, attribution, annotation):
    """The static card, and the box (x0, x1, baseline y) of the annotation for the underline. The attribution is one
    line, "<attribution> · <annotation>"; the box starts where the annotation's text starts, measured from the
    rendered width of the prefix."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    qf, qi, af = fonts["quote"], fonts["quote_italic"], fonts["attr"]
    q_lines = wrap(d, tokens(f"“{text}”", qf, qi), TEXT_W)
    q_lh, a_lh = int(qf.size * 1.25), int(af.size * 1.3)
    star_r, star_gap, g1, g2 = 46, 24, 56, 40
    total = 2 * star_r + g1 + q_lh * len(q_lines) + g2 + a_lh
    y = (H - total) // 2
    for k in range(5):
        star(d, MARGIN_X + star_r + k * (2 * star_r + star_gap), y + star_r, star_r, filled=(k == 0))
    y = draw_lines(d, q_lines, y + 2 * star_r + g1, q_lh, FG) + g2
    prefix = f"{attribution} · "
    d.text((MARGIN_X, y), prefix + annotation, font=af, fill=MUTED)
    x0 = MARGIN_X + d.textlength(prefix, font=af)
    asc, desc = af.getmetrics()
    box = (x0, x0 + d.textlength(annotation, font=af), y + asc + int(desc * 0.4))
    return im, box


def marker_stroke(box, seed):
    """A hand-drawn marker stroke under the box: points along it, each with a radius. Uneven thickness, a slight
    slope and wobble off horizontal, and ends that overshoot the text a little and taper."""
    rng = random.Random(seed)
    x0, x1, y = box
    x0 -= rng.uniform(3, 8)    # starts just before the annotation, never under the separator
    x1 += rng.uniform(10, 20)
    slope = rng.uniform(-0.012, 0.006)
    phase, freq = rng.uniform(0, 6.28), rng.uniform(1.2, 2.2)
    n = int((x1 - x0) / 2.5)
    pts = []
    for i in range(n + 1):
        t = i / n
        x = x0 + t * (x1 - x0)
        yy = (y + 9 + slope * (x - x0) + 2.2 * math.sin(phase + freq * 2 * math.pi * t)
              + 1.2 * math.sin(phase * 1.7 + 9 * t) + rng.uniform(-0.6, 0.6))
        taper = min(1.0, t / 0.06, (1 - t) / 0.09)
        r = (6.8 + 1.9 * math.sin(phase + 5 * t) + rng.uniform(-0.7, 0.7)) * (0.45 + 0.55 * max(0.0, taper))
        pts.append((x, yy, r))
    return pts


def draw_marker(base, pts, progress, seed):
    """Draw the first `progress` of the stroke as a textured red marker over `base`."""
    if progress <= 0:
        return base
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    rng = random.Random(seed + 1)
    upto = max(1, int(len(pts) * min(1.0, progress)))
    for x, y, r in pts[:upto]:
        for _ in range(3):  # several overlapping dabs per point: a fibrous marker edge
            jx, jy, jr = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), r * rng.uniform(0.82, 1.05)
            d.ellipse([x + jx - jr, y + jy - jr, x + jx + jr, y + jy + jr], fill=rng.randint(170, 235))
    grain = Image.effect_noise((W, H), 40).point(lambda v: 255 if v > 96 else 140)  # dry, uneven ink
    alpha = Image.fromarray((np.asarray(layer, np.float32) * np.asarray(grain, np.float32) / 255).astype(np.uint8))
    alpha = alpha.filter(ImageFilter.GaussianBlur(0.6))
    out = base.copy()
    out.paste(Image.new("RGB", (W, H), RED), (0, 0), alpha)
    return out


# ---------- checkpoint card ----------

def checkpoint_scene():
    """Render the Rat Trap's start (iso view) at W x H and project each checkpoint's representative node."""
    sys.path.insert(0, str(ROOT))
    from loophole.puzzles.rattrap.sim import RatTrapSim
    from loophole.puzzles.rattrap.spec import load_spec

    sim = RatTrapSim(load_spec())
    sim.model.vis.global_.offwidth, sim.model.vis.global_.offheight = W, H
    sim.render("iso", distance=0.45, width=W, height=H)  # creates the renderer at this size
    cam = sim._renderer.camera("iso", None, None, 0.45, None)
    arr, cm = sim._renderer.render_modalities(sim.data, cam, ["rgb"], W, H)
    pos, fwd, up, right = (np.array(cm[k]) for k in ("position", "forward", "up", "right"))
    fx, cx, cy = cm["intrinsics"]["fx"], cm["intrinsics"]["cx"], cm["intrinsics"]["cy"]

    def proj(p):
        d = np.asarray(p) - pos
        z = d @ fwd
        return float(cx + fx * (d @ right) / z - 0.5), float(cy - fx * (d @ up) / z - 0.5)

    z = np.load(ROOT / "loophole" / "reference" / "rattrap-checkpoints.npz")
    states, meta = z["states"], json.loads(str(z["meta"]))
    events, held = [], {}
    for line in (ROOT / "loophole" / "reference" / "rattrap-solve.jsonl").open():
        r = json.loads(line)
        if r.get("type") == "grab":
            held[r["handle"]] = r["node"]
            events.append(("grab", r["handle"], r["node"]))
        elif r.get("type") == "release":
            events.append(("release", r["handle"], held.get(r["handle"])))
    stamps = []
    for k, b in enumerate(meta["kept_boundary_indices"]):
        kind, handle, node = events[b]
        i = sim.node_index(node)
        u, v = proj(states[k][i])
        stamps.append({"checkpoint": k + 1, "boundary_event": b, "event": kind, "handle": handle, "node": int(node),
                       "xyz_m": [round(float(c), 5) for c in states[k][i]], "uv_px": [round(u, 1), round(v, 1)]})
    return Image.fromarray(arr["rgb"]), stamps, cm


def stamp_sprite(size, seed):
    """A circled red X with a rubber-stamp texture, on transparency."""
    rng = random.Random(seed)
    s = size * 3  # draw large, then downsample
    im = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(im)
    w = int(s * 0.085)
    d.ellipse([w, w, s - w, s - w], outline=255, width=w)
    m = s * 0.3
    d.line([(m, m), (s - m, s - m)], fill=255, width=int(w * 1.15))
    d.line([(s - m, m), (m, s - m)], fill=255, width=int(w * 1.15))
    grain = Image.effect_noise((s, s), 60).point(lambda v: 255 if v > 70 + rng.randint(-10, 10) else 60)
    alpha = Image.fromarray((np.asarray(im, np.float32) * np.asarray(grain, np.float32) / 255 * 0.92).astype(np.uint8))
    alpha = alpha.resize((size, size), Image.LANCZOS)
    sprite = Image.new("RGBA", (size, size), RED + (0,))
    sprite.putalpha(alpha)
    return sprite


def checkpoint_frames(still, stamps, fonts, best, K):
    """Stars on top, the still (cropped to the puzzle) with the stamps landing in sequence, then the headline and the
    attribution once all have landed."""
    rng = random.Random(7)
    # crop the still around the puzzle and place it below the stars
    crop = (540, 150, 1440, 1000)  # the whole frame and base, and every stamp (u 754-1079, v 240-772)
    img_h = 590
    k_scale = img_h / (crop[3] - crop[1])
    img_w = round((crop[2] - crop[0]) * k_scale)
    pic = still.crop(crop).resize((img_w, img_h), Image.LANCZOS)
    mask = Image.new("L", pic.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img_w - 1, img_h - 1], radius=16, fill=255)
    hf, af = fonts["headline"], fonts["attr"]
    star_r, star_gap, g1, g2, g3 = 46, 24, 40, 44, 16
    h_lh, a_lh = int(hf.size * 1.2), int(af.size * 1.3)
    total = 2 * star_r + g1 + img_h + g2 + h_lh + g3 + a_lh
    y0 = (H - total) // 2
    pic_y = y0 + 2 * star_r + g1
    text_y = pic_y + img_h + g2
    headline = f"{best} of {K} reached · verified purchase"
    attribution = "— every frontier model we tested"

    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    for k in range(5):
        star(d, MARGIN_X + star_r + k * (2 * star_r + star_gap), y0 + star_r, star_r, filled=(k == 0))
    base.paste(pic, (MARGIN_X, pic_y), mask)
    final_text = base.copy()
    dt = ImageDraw.Draw(final_text)
    dt.text((MARGIN_X, text_y), headline, font=hf, fill=FG)
    dt.text((MARGIN_X, text_y + h_lh + g3), attribution, font=af, fill=MUTED)

    size = round(92 * k_scale)
    sprites = [stamp_sprite(size, 100 + i) for i in range(len(stamps))]
    rot = [rng.uniform(-14, 14) for _ in stamps]
    land = [round((CP_FIRST_AT + i * CP_GAP) * FPS) for i in range(len(stamps))]
    cap_at = round(CP_CAPTION_AT * FPS)
    clip = Image.new("L", (W, H), 0)  # stamps stay inside the picture
    clip.paste(mask, (MARGIN_X, pic_y))
    frames = []
    for f in range(round(CP_CARD_S * FPS)):
        im = (final_text if f >= cap_at else base).convert("RGBA")
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for i, st in enumerate(stamps):
            if f < land[i]:
                continue
            age = f - land[i]
            scale = 1.0 + 0.4 * max(0.0, 1 - age / 3)  # 1.4 on arrival, 1.0 three frames later: a hard impact
            sz = max(8, int(size * scale))
            spr = sprites[i].resize((sz, sz), Image.LANCZOS).rotate(rot[i], resample=Image.BICUBIC, expand=True)
            u = MARGIN_X + (st["uv_px"][0] - crop[0]) * k_scale
            v = pic_y + (st["uv_px"][1] - crop[1]) * k_scale
            layer.alpha_composite(spr, (int(u - spr.width / 2), int(v - spr.height / 2)))
        a_ = Image.fromarray((np.asarray(layer.split()[3], np.float32) * np.asarray(clip, np.float32) / 255).astype(np.uint8))
        layer.putalpha(a_)
        im.alpha_composite(layer)
        frames.append(im.convert("RGB"))
    return frames


# ---------- assembly ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-font", required=True)
    ap.add_argument("--body-italic-font", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media" / "testimonial-cards.mp4")
    a = ap.parse_args()

    def font(path, size, weight):
        f = ImageFont.truetype(path, size)
        try:
            f.set_variation_by_axes([weight])
        except (OSError, AttributeError):
            pass
        return f

    fonts = {"quote": font(a.body_font, 88, 600), "quote_italic": font(a.body_italic_font, 88, 600),
             "attr": font(a.body_font, 50, 400)}

    s462 = quote(KIMI, 462, "move the upper bight around")
    s551 = quote(KIMI, 551, "unnecessarily cyclic")
    s180 = quote(DEEPSEEK, 180, "doesn't change topology")
    s629 = quote(DEEPSEEK, 629, "cannot solve this")
    kimi = json.loads((KIMI / "result.json").read_text())
    ds = json.loads((DEEPSEEK / "result.json").read_text())
    after_551 = int(kimi["actions"]) - actions_used_at(KIMI, 551)
    unused, budget = int(ds["max_actions"]) - int(ds["actions"]), int(ds["max_actions"])
    best, K = all_runs_checkpoints()
    if best != 0:
        raise SystemExit(f"a committed run reached {best} checkpoints: card 5's '0 of {K}' would be false")

    reviews = [
        (excerpt(s462, "move the upper bight", "upper post*"), "— Kimi K3 Pro, turn 462", "It did not."),
        (excerpt(s551, "The action pattern", "unnecessarily cyclic"), "— Kimi K3 Pro, turn 551",
         f"it continued for another {after_551} actions"),
        (excerpt(s180, "moving it around", "topology."), "— DeepSeek V4.1", "correct, as it happens"),
        (s629, "— DeepSeek V4.1", f"{unused} of {budget} actions unused"),
    ]
    png_dir = a.out.with_suffix("")
    png_dir.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(a.out)],
        stdin=subprocess.PIPE,
    )
    n_card, u0, u1 = round(CARD_S * FPS), round(UNDERLINE_AT * FPS), round((UNDERLINE_AT + UNDERLINE_S) * FPS)
    for c, (text, attribution, annotation) in enumerate(reviews, 1):
        base, box = review_card(fonts, text, attribution, annotation)
        pts = marker_stroke(box, seed=c)
        for f in range(n_card):
            p = 0.0 if f < u0 else min(1.0, (f - u0 + 1) / (u1 - u0))
            p = 1 - (1 - p) ** 2  # a quick start that slows into the end of the stroke
            frame = draw_marker(base, pts, p, seed=c)
            ff.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        draw_marker(base, pts, 1.0, seed=c).save(png_dir / f"card-{c}.png", optimize=True)

    still, stamps, cam = checkpoint_scene()
    cp = checkpoint_frames(still, stamps, {"headline": font(a.body_font, 80, 600), "attr": fonts["attr"]}, best, K)
    for frame in cp:
        ff.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    cp[-1].save(png_dir / "card-5.png", optimize=True)
    (png_dir / "card-5-stamps.json").write_text(json.dumps(
        {"representative": "per checkpoint, the cord node the human grabbed or released at the grab / release event "
                           "the checkpoint was taken at, at its position in that checkpoint's state, projected with "
                           "the render camera (iso view of the start state)",
         "camera": {k: cam[k] for k in ("position", "forward", "up", "right", "fovy_deg", "width", "height")},
         "stamps": stamps}, indent=1))
    ff.stdin.close()
    ff.wait()
    seconds = (len(reviews) * n_card + len(cp)) / FPS
    print(json.dumps({"out": str(a.out), "seconds": seconds,
                      "reviews": [{"quote": t, "attribution": at, "annotation": an} for t, at, an in reviews],
                      "numbers": {"after_551": after_551, "deepseek_unused": unused, "budget": budget,
                                  "best_checkpoints_any_run": best, "K": K}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
