#!/usr/bin/env python3
"""Six title cards about two Rat Trap agent failures, as one MP4 (hard cuts) plus one PNG per card.

    python3 scripts/make_failure_cards.py --body-font SourceSans3[wght].ttf --mono-font RobotoMono[wght].ttf \
        --mono-italic-font RobotoMono-Italic[wght].ttf [--durations 3.5,4,3,4,4.5,5]

Fonts are the docs site's (Source Sans for body text, Roboto Mono for code), from github.com/google/fonts (OFL).
Every agent quote is read from the committed transcripts, never retyped: the full sentence of the model's visible
reply (`content`) at the given turn that contains the given phrase. Every number is read from the run's
`rescore.json` / `result.json`. Markdown emphasis in a quote (*...*) is set in italic.

Output: docs/media/failure-cards.mp4 (1920x1080, 30 fps, H.264) and docs/media/failure-cards/card-N.png.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "rattrap-agents"
KIMI = RUNS / "kimi-20261114-205209"        # Kimi K3 Pro, round 2 (full budget)
DEEPSEEK = RUNS / "deepseek-20261114-205209"  # DeepSeek V4.1 Flash, round 2 (gave up)

W, H, FPS = 1920, 1080, 30
BG = (22, 32, 58)          # the site's text navy (#16203a), as the dark ground
FG = (255, 255, 255)
MUTED = (184, 196, 222)
ACCENT = (125, 155, 255)   # the site's accent (#2b5cff), lightened for contrast on the navy
MARGIN_X = 180
TEXT_W = W - 2 * MARGIN_X


# ---------- data from the committed logs ----------

def quote(run: Path, turn: int, phrase: str) -> str:
    """The sentence of the reply at `turn` that contains `phrase`, verbatim."""
    for line in (run / "transcript.jsonl").open():
        r = json.loads(line)
        if r.get("type") == "reply" and r.get("turn") == turn:
            text = r["message"].get("content") or ""
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if phrase in sentence:
                    return sentence.strip()
            raise SystemExit(f"{run.name} turn {turn}: no sentence contains {phrase!r}")
    raise SystemExit(f"{run.name}: no reply at turn {turn}")


def numbers() -> dict:
    k = json.loads((KIMI / "rescore.json").read_text())
    d = json.loads((DEEPSEEK / "result.json").read_text())
    return {"kimi_actions": k["actions"], "kimi_blocked": k["blocked_moves"], "kimi_k": k["checkpoints_reached"],
            "kimi_K": k["K"], "ds_used": d["actions"], "ds_budget": d["max_actions"],
            "ds_unused": d["max_actions"] - d["actions"]}


# ---------- typesetting ----------

class Fonts:
    def __init__(self, body: str, mono: str, mono_italic: str):
        self.paths = {"body": body, "mono": mono, "mono_italic": mono_italic}
        self.cache: dict = {}

    def get(self, face: str, size: int, weight: int):
        key = (face, size, weight)
        if key not in self.cache:
            f = ImageFont.truetype(self.paths[face], size)
            try:
                f.set_variation_by_axes([weight])
            except (OSError, AttributeError):
                pass
            self.cache[key] = f
        return self.cache[key]


def tokens(text: str, upright, italic):
    """(word, font, space_before) triples; *...* spans are italic, the asterisks dropped. Punctuation right after a
    span keeps no space before it; a newline forces a line break (word None)."""
    out = []
    for li, line in enumerate(text.split("\n")):
        if li:
            out.append((None, upright, False))
        prev_part = ""
        for i, part in enumerate(re.split(r"\*([^*]+)\*", line)):
            f = italic if i % 2 else upright
            for j, w in enumerate(part.split()):
                # join to the previous word only where the source has no space on either side of the span edge
                glue = (j == 0 and i > 0 and out and out[-1][0] is not None
                        and not part[:1].isspace() and not prev_part[-1:].isspace())
                out.append((w, f, not glue))
            prev_part = part
    return out


def wrap(draw, toks, width):
    lines, cur, cur_w = [], [], 0
    for word, f, space in toks:
        if word is None:  # forced break
            lines.append(cur)
            cur, cur_w = [], 0
            continue
        ww = draw.textlength(word, font=f)
        sp = draw.textlength(" ", font=f) if cur and space else 0
        if cur and cur_w + sp + ww > width:
            lines.append(cur)
            cur, cur_w, sp, space = [], 0, 0, True
        cur.append((word, f, space))
        cur_w += sp + ww
    if cur:
        lines.append(cur)
    return lines


def block(draw, text, upright, italic=None, color=FG, width=TEXT_W, leading=1.32):
    """Wrapped lines of (word, font); returns (lines, height, line_height, color)."""
    lines = wrap(draw, tokens(text, upright, italic or upright), width)
    lh = int(upright.size * leading)
    return lines, lh * len(lines), lh, color


def render(blocks, gaps) -> Image.Image:
    """Stack blocks vertically, centred as a group, left-aligned at the margin."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    total = sum(b[1] for b in blocks) + sum(gaps)
    y = (H - total) // 2
    for i, (lines, h, lh, color) in enumerate(blocks):
        for line in lines:
            x = MARGIN_X
            for j, (word, f, space) in enumerate(line):
                if j and space:
                    x += d.textlength(" ", font=f)
                d.text((x, y), word, font=f, fill=color)
                x += d.textlength(word, font=f)
            y += lh
        if i < len(gaps):
            y += gaps[i]
    return im


def cards(F: Fonts, n: dict) -> list[Image.Image]:
    d = ImageDraw.Draw(Image.new("RGB", (W, H)))
    title = F.get("body", 112, 700)
    stat = F.get("body", 72, 400)
    label = F.get("body", 44, 600)
    mono, mono_i = F.get("mono", 60, 400), F.get("mono_italic", 60, 400)
    mine = F.get("body", 104, 600)
    after = F.get("body", 80, 600)

    q462 = quote(KIMI, 462, "move the upper bight around")
    q551 = quote(KIMI, 551, "unnecessarily cyclic")
    q_topo = quote(DEEPSEEK, 180, "doesn't change topology")
    q_quit = quote(DEEPSEEK, 629, "cannot solve this")

    def q(text):
        return f"“{text}”"

    out = [
        render([block(d, "Kimi K3 Pro — full budget", title),
                block(d, f"{n['kimi_actions']} actions, {n['kimi_blocked']} blocked, "
                         f"{n['kimi_k']} of {n['kimi_K']} checkpoints", stat, color=MUTED)], [44]),
        render([block(d, "TURN 462", label, color=ACCENT), block(d, q(q462), mono, mono_i)], [28]),
        render([block(d, "That is the correct move.\nIt never executed it.", mine)], []),
        render([block(d, "TURN 551", label, color=ACCENT), block(d, q(q551), mono, mono_i),
                block(d, "It continued anyway.", after)], [28, 64]),
        render([block(d, "DEEPSEEK V4.1 FLASH · TURN 180", label, color=ACCENT), block(d, q(q_topo), mono, mono_i),
                block(d, "TURN 629", label, color=ACCENT), block(d, q(q_quit), mono, mono_i),
                block(d, f"It stopped with {n['ds_unused']} of {n['ds_budget']} actions unused.", after)],
               [28, 56, 28, 64]),
        render([block(d, "Neither failure was\na lack of information.", mine)], []),
    ]
    return out, {"462": q462, "551": q551, "180": q_topo, "629": q_quit}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-font", required=True)
    ap.add_argument("--mono-font", required=True)
    ap.add_argument("--mono-italic-font", required=True)
    ap.add_argument("--durations", default="3.5,4,3,4,4.5,5", help="seconds per card; card 6 holds longest")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media" / "failure-cards.mp4")
    a = ap.parse_args()
    durations = [float(x) for x in a.durations.split(",")]
    F = Fonts(a.body_font, a.mono_font, a.mono_italic_font)
    n = numbers()
    images, quotes = cards(F, n)
    if len(durations) != len(images):
        raise SystemExit(f"--durations needs {len(images)} values")
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
    for im, dur in zip(images, durations):  # hard cuts: each card's frames, then the next
        raw = np.asarray(im, dtype=np.uint8).tobytes()
        for _ in range(round(dur * FPS)):
            ff.stdin.write(raw)
    ff.stdin.close()
    ff.wait()
    print(json.dumps({"out": str(a.out), "pngs": str(png_dir), "seconds": sum(round(x * FPS) for x in durations) / FPS,
                      "durations": durations, "numbers": n, "quotes": quotes}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
