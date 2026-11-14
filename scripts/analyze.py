#!/usr/bin/env python3
"""Analyse an eval run: per-trial behaviour from the simulator episode logs, plots, a Markdown report.

    .venv/bin/python scripts/analyze.py runs/20261114-080705 [--out docs/RESULTS.md] [--media docs/media]

Per trial (oracle, noop, stop-early, each OpenCode run) it reads verifier/episode.jsonl and the
score (verifier/reverified.json if present, else verifier/report.json), and reports:
actions used, blocked moves, observation calls by tool (render / get_state / ...), progress
(1 - held cost-to-go / start), the final outcome, plus a plot of cost-to-go per action
(<media>/<run>-progress.png) and of the observation/action mix (<media>/<run>-calls.png).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("oracle", "noop", "stop-early", "opencode")


def load_trial(trial: Path) -> dict | None:
    ep = trial / "verifier" / "episode.jsonl"
    if not ep.exists():
        return None
    recs = [json.loads(l) for l in ep.open() if l.strip()]
    rv = trial / "verifier" / "reverified.json"
    rep = json.loads(rv.read_text()) if rv.exists() else json.loads((trial / "verifier" / "report.json").read_text())
    acts = [r for r in recs if r["type"] == "action"]
    obs = collections.Counter(r["tool"] for r in recs if r["type"] == "observe")
    calls = collections.Counter(r["tool"] for r in acts)
    rejected = sum(r["type"] == "rejected" for r in recs)
    ctg = [r.get("cost_to_go") for r in acts]
    hdr = recs[0]
    c0 = hdr.get("start_cost_to_go")
    final = rep.get("final") or next((r for r in recs if r["type"] == "final"), {})
    stable = (final.get("cost_to_go") or {}).get("stable_best")
    return {
        "trial": trial.name, "score": rep.get("reward"), "success": bool(rep.get("success")), "reason": rep.get("reason"),
        "log_consistent": rep.get("log_consistent"), "actions": len(acts), "calls": dict(calls), "observations": dict(obs),
        "rejected": rejected, "blocked": sum(bool((r.get("result") or {}).get("blocked")) for r in acts),
        "ctg": ctg, "c0": c0, "progress": None if not (c0 and stable is not None) else max(0.0, 1 - stable / c0),
        "min_distance": [r.get("min_distance") for r in acts],
    }


def plots(run: str, data: dict, media: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    media.mkdir(parents=True, exist_ok=True)
    out = []
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for group, trials in data.items():
        for t in trials:
            if not t["ctg"] or not t["c0"]:
                continue
            ys = [None if c is None else 1 - c / t["c0"] for c in t["ctg"]]
            lbl = f"{group} {t['trial'].split('__')[-1]} ({t['score']:.3f})"
            ax.plot(range(1, len(ys) + 1), ys, marker="." if len(ys) < 3 else None, label=lbl, lw=2 if group == "oracle" else 1.2)
    ax.set_xlabel("action")
    ax.set_ylabel("progress 1 - ctg / ctg0")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Progress along the way out (roadmap cost-to-go), per action")
    fig.tight_layout()
    p = media / f"{run}-progress.png"
    fig.savefig(p, dpi=130)
    out.append(p)
    plt.close(fig)

    rows = [(f"{g}\n{t['trial'].split('__')[-1]}", t) for g, ts in data.items() for t in ts]
    keys = ["render", "get_state", "get_contacts", "grab", "move", "release", "wait"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bottom = [0] * len(rows)
    for k in keys:
        vals = [t["observations"].get(k, 0) + t["calls"].get(k, 0) for _, t in rows]
        if any(vals):
            ax.bar(range(len(rows)), vals, bottom=bottom, label=k)
            bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(rows)), [r for r, _ in rows], fontsize=7)
    ax.set_yscale("symlog")
    ax.set_ylabel("MCP calls (logged)")
    ax.legend(fontsize=7, ncol=4)
    ax.set_title("What each run did: observations vs actions")
    fig.tight_layout()
    p = media / f"{run}-calls.png"
    fig.savefig(p, dpi=130)
    out.append(p)
    plt.close(fig)
    return [str(x) for x in out]


def report(run_dir: Path, data: dict, figs: list[str], out: Path) -> str:
    cfg = json.loads((run_dir / "config.json").read_text())
    lines = [f"## Run `{run_dir.name}`: `{cfg['model']}` (OpenCode), task `tasks/loophole-alpha`", "",
             "| run | trial | score | solved | actions | blocked moves | renders | get_state | progress (held) | outcome |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for g, ts in data.items():
        for t in ts:
            prog = "–" if t["progress"] is None else f"{t['progress']:.3f}"
            lines.append(f"| {g} | {t['trial']} | {t['score']:.3f} | {'yes' if t['success'] else 'no'} | {t['actions']} | {t['blocked']} | "
                         f"{t['observations'].get('render', 0)} | {t['observations'].get('get_state', 0)} | {prog} | {t['reason']} |")
    agents = [t["score"] for t in data.get("opencode", [])]
    if agents:
        import statistics

        sd = statistics.stdev(agents) if len(agents) > 1 else 0.0
        lines += ["", f"OpenCode `{cfg['model']}`: **{statistics.mean(agents):.3f} ± {sd:.3f}** (n = {len(agents)}, mean ± sd), "
                  f"solved {sum(t['success'] for t in data['opencode'])}/{len(agents)}."]
    lines += [""] + [f"![{Path(f).stem}]({Path(f).relative_to(out.parent) if Path(f).is_relative_to(out.parent) else f})" for f in figs]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--media", type=Path, default=ROOT / "docs" / "media")
    ap.add_argument("--out", type=Path, default=None, help="write the Markdown section here (default: print)")
    a = ap.parse_args()
    data = {g: [t for t in (load_trial(p) for p in sorted((a.run_dir / g).glob("*__*"))) if t] for g in GROUPS}
    data = {g: ts for g, ts in data.items() if ts}
    figs = plots(a.run_dir.name, data, a.media)
    md = report(a.run_dir, data, figs, (a.out or ROOT / "docs" / "RESULTS.md").resolve())
    (a.run_dir / "analysis.json").write_text(json.dumps({g: [{k: v for k, v in t.items() if k not in ("ctg", "min_distance")} for t in ts]
                                                         for g, ts in data.items()}, indent=1))
    if a.out:
        a.out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
