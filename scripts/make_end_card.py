#!/usr/bin/env python3
"""The end card: title, tagline, URL, and the fine print. One MP4 (hard reveals) plus a PNG per step.

    python3 scripts/make_end_card.py --body-font SourceSans3[wght].ttf

Same background, colours and body font as scripts/make_failure_cards.py; the title is set bold, like og-card.png.

0-1 s the title alone; 1 s the tagline appears under it; 3-4 s both hold; 4 s the URL appears below, and the fine
print appears at the bottom-left corner of the frame, at legal-disclaimer size, outside the centred group.
Output: docs/media/end-card.mp4 (1920x1080, 30 fps, 9 s) and docs/media/end-card/step-N.png.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_failure_cards import ACCENT, BG, FG, FPS, H, MUTED, ROOT, TEXT_W, W  # noqa: E402

TITLE = "Loophole"
TAGLINE = "Children’s 3D puzzles that frontier AI models can’t solve."
URL = "tibiic.github.io/Loophole"
FINE_PRINT = "Model not included."
TAGLINE_AT, URL_AT, SECONDS = 1.0, 4.0, 9.0
SAFE_X, SAFE_Y = 96, 54  # title-safe inset (5 %) for the fine print


def face(path: str, size: int, weight: int):
    f = ImageFont.truetype(path, size)
    try:
        f.set_variation_by_axes([weight])
    except (OSError, AttributeError):
        pass
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-font", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media" / "end-card.mp4")
    a = ap.parse_args()
    d = ImageDraw.Draw(Image.new("RGB", (W, H)))
    title = face(a.body_font, 220, 700)
    tag_size = 64
    while d.textlength(TAGLINE, font=face(a.body_font, tag_size, 400)) > TEXT_W:
        tag_size -= 2
    tagline = face(a.body_font, tag_size, 400)
    url = face(a.body_font, 56, 600)
    fine = face(a.body_font, 24, 400)

    # The centred group is laid out once, with every line, so nothing moves when a line appears.
    lines = [(TITLE, title, FG), (TAGLINE, tagline, MUTED), (URL, url, ACCENT)]
    gaps = [round(tag_size * 0.5), round(tag_size * 1.6)]
    boxes = [d.textbbox((0, 0), t, font=f, anchor="ls") for t, f, _ in lines]  # relative to the baseline
    heights = [b[3] - b[1] for b in boxes]
    y = (H - sum(heights) - sum(gaps)) // 2
    baselines = []
    for i, b in enumerate(boxes):
        baselines.append(y - b[1])
        y += heights[i] + (gaps[i] if i < len(gaps) else 0)

    def frame(shown: int, fine_print: bool) -> Image.Image:
        im = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(im)
        for (t, f, c), base in list(zip(lines, baselines))[:shown]:
            dr.text((W / 2, base), t, font=f, fill=c, anchor="ms")
        if fine_print:
            dr.text((SAFE_X, H - SAFE_Y), FINE_PRINT, font=fine, fill=MUTED, anchor="ls")
        return im

    steps = [frame(1, False), frame(2, False), frame(3, True)]
    starts = [0, round(TAGLINE_AT * FPS), round(URL_AT * FPS)]
    total = round(SECONDS * FPS)
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
        ff.stdin.write(raws[sum(f >= s for s in starts) - 1])
    ff.stdin.close()
    ff.wait()
    print(json.dumps({"out": str(a.out), "seconds": total / FPS, "step_starts_s": [s / FPS for s in starts],
                      "px": {"title": title.size, "tagline": tag_size, "url": url.size, "fine_print": fine.size}}))


if __name__ == "__main__":
    main()
