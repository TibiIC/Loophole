#!/usr/bin/env python3
"""The failure cards re-cut as one-star product reviews: five cards, 4 s each, one MP4 plus one PNG per card.

    python3 scripts/make_testimonial_cards.py --body-font SourceSans3[wght].ttf \
        --body-italic-font SourceSans3-Italic[wght].ttf --mono-font RobotoMono[wght].ttf \
        --mono-italic-font RobotoMono-Italic[wght].ttf

Same background and fonts as scripts/make_failure_cards.py. Each quote is cut verbatim from the committed transcript
(the model's visible reply at the given turn), with an ellipsis wherever the cut is inside a sentence; the model's
own *emphasis* is kept as italic. Every number is computed from the run's transcript, result.json or rescore.json.

Output: docs/media/testimonial-cards.mp4 (1920x1080, 30 fps, 20 s, hard cuts) and docs/media/testimonial-cards/card-N.png.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_failure_cards import (BG, DEEPSEEK, FG, FPS, H, KIMI, MARGIN_X, MUTED, RUNS, ROOT, TEXT_W, W,  # noqa: E402
                                quote, tokens, wrap)

AMBER = (245, 181, 40)


def excerpt(sentence: str, start: str, end: str) -> str:
    """The verbatim span of `sentence` from `start` through `end`, with an ellipsis on each side that cuts it."""
    i = sentence.index(start)
    j = sentence.index(end, i) + len(end)  # pass a closing emphasis marker as part of `end`, e.g. "post*"
    span = sentence[i:j]
    lead = "" if i == 0 else "…"
    trail = "" if j >= len(sentence) or sentence[j:].strip() == "" else "…"
    return lead + span + trail


def actions_used_at(run: Path, turn: int) -> int:
    """actions_used reported by the get_state call of `turn`."""
    for line in (run / "transcript.jsonl").open():
        r = json.loads(line)
        if r.get("type") == "tool" and r.get("turn") == turn and r.get("name") == "get_state":
            return int(r["result"]["actions_used"])
    raise SystemExit(f"{run.name}: no get_state at turn {turn}")


def all_runs_checkpoints() -> tuple[int, int]:
    """(best checkpoints reached, K) over every committed Rat Trap agent run's rescore / result."""
    best, K = 0, 20
    for f in list(RUNS.glob("*/rescore.json")) + list(RUNS.glob("*/result.json")):
        d = json.loads(f.read_text())
        if d.get("checkpoints_reached") is not None:
            best = max(best, int(d["checkpoints_reached"]))
        K = int(d.get("K") or d.get("checkpoints") or K)
    return best, K


def star(draw, cx, cy, r, filled):
    pts = [(cx + (r if k % 2 == 0 else r * 0.42) * math.sin(math.pi * k / 5),
            cy - (r if k % 2 == 0 else r * 0.42) * math.cos(math.pi * k / 5)) for k in range(10)]
    if filled:
        draw.polygon(pts, fill=AMBER)
    else:
        draw.polygon(pts, outline=AMBER, width=max(3, int(r * 0.09)))


def card(fonts, text, attribution, quoted=True) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    qf, qi, af = fonts["quote"], fonts["quote_italic"], fonts["attr"]
    body = f"“{text}”" if quoted else text
    lines = wrap(d, tokens(body, qf, qi), TEXT_W)
    q_lh = int(qf.size * 1.25)
    a_lines = wrap(d, tokens(attribution, af, af), TEXT_W)
    a_lh = int(af.size * 1.3)
    star_r, star_gap, g1, g2 = 46, 24, 56, 44
    total = 2 * star_r + g1 + q_lh * len(lines) + g2 + a_lh * len(a_lines)
    y = (H - total) // 2
    for k in range(5):  # one star of five
        star(d, MARGIN_X + star_r + k * (2 * star_r + star_gap), y + star_r, star_r, filled=(k == 0))
    y += 2 * star_r + g1
    for ls, lh, color in ((lines, q_lh, FG), (a_lines, a_lh, MUTED)):
        for line in ls:
            x = MARGIN_X
            for j, (word, f, space) in enumerate(line):
                if j and space:
                    x += d.textlength(" ", font=f)
                d.text((x, y), word, font=f, fill=color)
                x += d.textlength(word, font=f)
            y += lh
        if ls is lines:
            y += g2
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-font", required=True)
    ap.add_argument("--body-italic-font", required=True)
    ap.add_argument("--mono-font", required=True)
    ap.add_argument("--mono-italic-font", required=True)
    ap.add_argument("--seconds-per-card", type=float, default=4.0)
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
             "attr": font(a.body_font, 48, 400)}

    s462 = quote(KIMI, 462, "move the upper bight around")
    s551 = quote(KIMI, 551, "unnecessarily cyclic")
    s180 = quote(DEEPSEEK, 180, "doesn't change topology")
    s629 = quote(DEEPSEEK, 629, "cannot solve this")
    q1 = excerpt(s462, "move the upper bight", "upper post*")
    q2 = excerpt(s551, "The action pattern", "unnecessarily cyclic")
    q3 = excerpt(s180, "moving it around", "topology.")
    q4 = s629

    kimi = json.loads((KIMI / "result.json").read_text())
    ds = json.loads((DEEPSEEK / "result.json").read_text())
    after_551 = int(kimi["actions"]) - actions_used_at(KIMI, 551)
    unused, budget = int(ds["max_actions"]) - int(ds["actions"]), int(ds["max_actions"])
    best, K = all_runs_checkpoints()

    specs = [
        (q1, "— Kimi K3 Pro, turn 462 · It did not.", True),
        (q2, f"— Kimi K3 Pro, turn 551 · it continued for another {after_551} actions", True),
        (q3, "— DeepSeek V4.1 · correct, as it happens", True),
        (q4, f"— DeepSeek V4.1 · {unused} of {budget} actions unused", True),
        (f"{best} of {K} checkpoints · verified purchase", "— every frontier model we tested", False),
    ]
    images = [card(fonts, t, att, quoted) for t, att, quoted in specs]
    png_dir = a.out.with_suffix("")
    png_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images, 1):
        im.save(png_dir / f"card-{i}.png", optimize=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-tune", "stillimage", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(a.out)],
        stdin=subprocess.PIPE,
    )
    n = round(a.seconds_per_card * FPS)
    for im in images:
        raw = np.asarray(im, dtype=np.uint8).tobytes()
        for _ in range(n):
            ff.stdin.write(raw)
    ff.stdin.close()
    ff.wait()
    print(json.dumps({"out": str(a.out), "seconds": n * len(images) / FPS,
                      "cards": [{"quote": t, "attribution": att} for t, att, _ in specs],
                      "numbers": {"kimi_actions_used_at_551": actions_used_at(KIMI, 551), "after_551": after_551,
                                  "deepseek_unused": unused, "budget": budget, "best_checkpoints_any_run": best, "K": K}},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
