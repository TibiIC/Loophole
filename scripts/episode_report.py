#!/usr/bin/env python3
"""Episode report: cost-to-go vs the topological diagnostics, per action.

    python scripts/episode_report.py runs/episode.jsonl -o runs/report.png

Panels: roadmap cost-to-go (the reward's navigation function), the Gauss
linking integral of the two tube centre-lines (log only), and the freely
reduced passage words of each centre-line through the other's loop disc.
They show at which "topological" stage an agent got stuck.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("-o", "--out", default="report.png")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.episode) if l.strip()]
    hdr = recs[0]
    acts = [r for r in recs if r.get("type") == "action"]
    final = next((r for r in recs if r.get("type") == "final"), None)
    xs = [0] + [r["i"] for r in acts]
    ctg = [hdr.get("start_cost_to_go")] + [r.get("cost_to_go") for r in acts]
    lk = [None] + [r.get("diagnostics", {}).get("linking_integral") for r in acts]
    words = [r.get("topology") or {} for r in acts]
    keys = sorted({k for w in words for k in w if not k.endswith("_raw") and k != "target"})
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(xs, ctg, "-o", ms=3, color="#2a6fdb")
    ax[0].set_ylabel("cost-to-go")
    ax[0].set_title(f"{hdr['puzzle']}: score={final['score'] if final else '?'} success={final['success'] if final else '?'}")
    ax[1].plot(xs, lk, "-o", ms=3, color="#d9622b")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_ylabel("Gauss linking\nintegral (log only)")
    for i, k in enumerate(keys):
        ax[2].step(xs[1:], [len(w.get(k, "")) for w in words], where="post", label=f"|{k}| reduced word")
    if keys:
        ax[2].legend(loc="upper right", fontsize=8)
    ax[2].set_ylabel("passage word length")
    ax[2].set_xlabel("action")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
