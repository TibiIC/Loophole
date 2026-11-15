#!/usr/bin/env python3
"""One features title card in the style of failure-cards.mp4: four lines appearing one at a time (hard cuts), all four
held together for the last two seconds.

    python3 scripts/make_features_card.py --body-font SourceSans3[wght].ttf --mono-font RobotoMono[wght].ttf \
        --mono-italic-font RobotoMono-Italic[wght].ttf

Same fonts, background and typesetting as scripts/make_failure_cards.py. Output: docs/media/features-card.mp4
(1920x1080, 30 fps, 6 s) and docs/media/features-card/step-N.png (the frame after line N appears).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_failure_cards import BG, FG, FPS, H, ROOT, TEXT_W, W, Fonts, block, render  # noqa: E402

LINES = [
    "Connect any agent over MCP",
    "Choose what it can see",
    "Choose how much the reward tells it",
    "Two puzzles, procedurally extendable",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-font", required=True)
    ap.add_argument("--mono-font", required=True)
    ap.add_argument("--mono-italic-font", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds with all lines shown, at the end")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media" / "features-card.mp4")
    a = ap.parse_args()
    F = Fonts(a.body_font, a.mono_font, a.mono_italic_font)
    d = ImageDraw.Draw(Image.new("RGB", (W, H)))
    size = 104  # the size of the cards' own lines, reduced until every line fits on one line
    while max(d.textlength(t, font=F.get("body", size, 600)) for t in LINES) > TEXT_W:
        size -= 2
    face = F.get("body", size, 600)
    gap = round(size * 0.35)

    # The layout is that of all four lines; a line not yet shown is drawn in the background colour, so nothing moves.
    def frame(shown: int) -> Image.Image:
        blocks = [block(d, t, face, color=FG if i < shown else BG) for i, t in enumerate(LINES)]
        return render(blocks, [gap] * (len(LINES) - 1))

    steps = [frame(k) for k in range(1, len(LINES) + 1)]
    build = a.seconds - a.hold  # lines 1..4 appear at 0, build/3, 2*build/3, build
    starts = [round(i * build / (len(LINES) - 1) * FPS) for i in range(len(LINES))]
    total = round(a.seconds * FPS)
    png_dir = a.out.with_suffix("")
    png_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(steps, 1):
        im.save(png_dir / f"step-{i}.png", optimize=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-tune", "stillimage", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(a.out)],
        stdin=subprocess.PIPE,
    )
    raws = [np.asarray(im, dtype=np.uint8).tobytes() for im in steps]
    for f in range(total):
        k = sum(f >= s for s in starts)  # lines shown at this frame
        ff.stdin.write(raws[k - 1])
    ff.stdin.close()
    ff.wait()
    print(json.dumps({"out": str(a.out), "seconds": total / FPS, "font_px": size,
                      "line_starts_s": [s / FPS for s in starts], "all_four_from_s": starts[-1] / FPS}))


if __name__ == "__main__":
    main()
