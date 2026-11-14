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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("oracle", "oracle-planner", "noop", "stop-early", "opencode")


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
    if "--axes" in sys.argv:
        print(axes(ROOT / "docs" / "RESULTS.md"))
        return
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



# ----------------------------------------------------------------------------------------------------------------
# Behaviour axes across every agent run (python scripts/analyze.py --axes): which axes separate solves from failures?
# ----------------------------------------------------------------------------------------------------------------
PROFILE = ROOT / "docs" / "alpha-oracle-profile.json"
TUNNEL_CLEARANCE = 0.003  # m: the published path is "in the tunnel" where its clearance is below this


def stages_from_profile() -> dict:
    """Stage boundaries in cost-to-go, from the published path's clearance profile (Zhang et al. 2020's vocabulary:
    wide 'bubbles' of free C-space joined by narrow 'tunnels'; 'key configurations' are the narrowest points)."""
    p = json.loads(PROFILE.read_text())
    rows = [r for r in p["rows"] if r["ctg"] is not None]
    tun = [r for r in rows if r["clearance"] < TUNNEL_CLEARANCE and not r["free"]]
    if not tun:
        return {"start_ctg": p["start_ctg"], "tunnel_entry": None, "key": None, "tunnel_exit": None}
    key = min(tun, key=lambda r: r["clearance"])
    return {"start_ctg": p["start_ctg"], "tunnel_entry": max(r["ctg"] for r in tun), "key": key["ctg"],
            "tunnel_exit": min(r["ctg"] for r in tun), "key_clearance_mm": round(key["clearance"] * 1000, 2)}


def stage_of(ctg: float | None, solved: bool, st: dict) -> str:
    if solved:
        return "free"
    if ctg is None or st["tunnel_entry"] is None:
        return "bubble (start)"
    if ctg > st["tunnel_entry"]:
        return "bubble (start)"
    if ctg > st["key"]:
        return "tunnel (before key)"
    if ctg > st["tunnel_exit"]:
        return "past key configuration"
    return "bubble (exit side)"


def strategy_class(trial: Path, actions: int) -> str:
    import re

    f = trial / "agent" / "opencode.txt"
    code = ""
    if f.exists():
        for line in f.open():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = e.get("part") or {}
            if p.get("type") == "tool" and (p.get("state") or {}).get("status") == "completed":
                i = p["state"].get("input") or {}
                code += str(i.get("command") or i.get("content") or i.get("newString") or "") + "\n"
    if actions == 0:
        return "aborted / no action"
    recon = len(re.findall(r"(?i)\bicp\b|point.?cloud|carv|voxel|kdtree", code))
    plan = len(re.findall(r"(?i)\brrt\b|rrt_|planner|solve_alpha|a_star|astar", code)) + len(re.findall(r"g\+\+", code))
    grad = len(re.findall(r"(?i)grad.{0,20}(cost_to_go|progress)|(cost_to_go|progress).{0,40}grad", code))
    if grad:
        return "gradient search"
    if recon and plan:
        return "reconstruct-and-plan"
    return "direct interaction"


def axes_row(trial: Path, group: str, model: str) -> dict:
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    from diagnose_run import diagnose

    dg = diagnose(trial, substeps=False)
    recs = [json.loads(l) for l in (trial / "verifier" / "episode.jsonl").open() if l.strip()]
    first = next((k for k, r in enumerate(recs) if r["type"] == "action"), len(recs))
    renders_before = sum(1 for r in recs[:first] if r.get("type") == "observe" and r.get("tool") == "render")
    acts = [r for r in recs if r["type"] == "action"]
    gaps = np.diff([r["wall_time"] for r in acts]) if len(acts) > 1 else np.array([])
    burst_frac = float(np.mean(gaps < 3.0)) if gaps.size else None
    ctgs = [r.get("cost_to_go") for r in acts if r.get("cost_to_go") is not None]
    rv = trial / "verifier" / "reverified.json"
    rep = json.loads(rv.read_text()) if rv.exists() else json.loads((trial / "verifier" / "report.json").read_text())
    solved = bool(rep.get("success"))
    moves = dg["moves"] or 0
    tr, rot = dg["translation_m_endpoints"], dg["rotation_rad_endpoints"]
    return {"run": trial.name.split("__")[-1], "group": group, "model": model, "class": strategy_class(trial, dg["actions"]),
            "score": rep.get("reward"), "solved": solved, "actions": dg["actions"],
            "blocked_rate": None if not moves else dg["blocked_moves"] / moves,
            "reversal_rate": None if not moves else dg["probe_then_reverse_pairs"] / moves,
            "renders_before_first_action": renders_before,
            "burst_fraction": burst_frac,
            "pattern": None if burst_frac is None else ("burst" if burst_frac >= 0.8 else "interleaved"),
            "translation_m": tr, "rotation_rad": rot, "rot_per_trans": None if not tr else rot / tr,
            "furthest_ctg": min(ctgs) if ctgs else None}


# harness-invalid trials (docs/HARNESS_NOTES.md, scripts/harness_audit.py): the model never got to act on the puzzle
EXCLUDED_MODELS = {"gemini/gemini-2.5-flash": "model no longer available to new API users (0 turns)"}
EXCLUDED_TRIALS = {"wRU3ykj": "API 400 'Requests ending with a model turn are not supported' before any action"}


def axes(out_md: Path) -> str:
    st = stages_from_profile()
    rows, excluded = [], []
    for run_dir in sorted(ROOT.glob("runs/2026*")):
        cfgp = run_dir / "config.json"
        if not cfgp.exists():
            continue
        cfg = json.loads(cfgp.read_text())
        for trial in sorted((run_dir / "opencode").glob("*__*")):
            tid, model = trial.name.split("__")[-1], cfg.get("model", "?")
            if model in EXCLUDED_MODELS or tid in EXCLUDED_TRIALS:
                excluded.append(f"`{tid}` (`{model}`, {run_dir.name}): {EXCLUDED_MODELS.get(model) or EXCLUDED_TRIALS[tid]}")
                continue
            if (trial / "verifier" / "episode.jsonl").exists():
                r = axes_row(trial, run_dir.name, cfg.get("model", "?"))
                r["stage"] = stage_of(r["furthest_ctg"], r["solved"], st)
                rows.append(r)
    numeric = ["blocked_rate", "reversal_rate", "renders_before_first_action", "burst_fraction", "rotation_rad", "translation_m", "rot_per_trans", "actions"]
    flags = {}
    for k in numeric:
        s = [r[k] for r in rows if r["solved"] and r[k] is not None]
        f = [r[k] for r in rows if not r["solved"] and r[k] is not None and r["class"] != "aborted / no action"]
        if not s or not f:
            flags[k] = "untested (needs both solves and failures)"
        elif max(s) < min(f) or max(f) < min(s):
            flags[k] = "SEPARATES"
        else:
            flags[k] = "does not separate (ranges overlap)"
    for k in ("pattern", "class", "stage"):
        mixed = [v for v in {r[k] for r in rows} if v is not None and len({r["solved"] for r in rows if r[k] == v}) > 1]
        flags[k] = "SEPARATES" if not mixed else f"does not separate (mixed: {', '.join(mixed)})"

    def f(x, nd=2):
        return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    lines = ["| class | run | model (interface run) | score | solved | actions | blocked rate | reversal rate | renders before 1st action | burst fraction (pattern) | translation m | rotation rad | furthest ctg (stage) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ["reconstruct-and-plan", "direct interaction", "gradient search", "aborted / no action"]
    for cls in order + sorted({r["class"] for r in rows} - set(order)):
        for r in sorted((x for x in rows if x["class"] == cls), key=lambda x: -(x["score"] or 0)):
            lines.append(f"| {cls} | {r['run']} | `{r['model']}` ({r['group']}) | {f(r['score'], 3)} | {'yes' if r['solved'] else 'no'} | {r['actions']} | "
                         f"{f(r['blocked_rate'])} | {f(r['reversal_rate'])} | {r['renders_before_first_action']} | {f(r['burst_fraction'])} ({r['pattern'] or '–'}) | "
                         f"{f(r['translation_m'])} | {f(r['rotation_rad'])} | {f(r['furthest_ctg'], 1)} ({r['stage']}) |")
    n_s = sum(r["solved"] for r in rows)
    n_f = sum((not r["solved"]) and r["class"] != "aborted / no action" for r in rows)
    lines += ["", f"Does the axis separate solves from failures? Sample: {n_s} solves, {n_f} non-aborted failures"
                  + (" - too few failures for any verdict to be a real signal yet." if n_f < 3 else ".")
                  + " (Aborted runs are excluded from the numeric checks.)", "",
              "| axis | verdict |", "|---|---|"] + [f"| {k} | {v} |" for k, v in flags.items()]
    lines += ["", f"Stages from the published path's clearance profile (`docs/alpha-oracle-profile.json`): tunnel = clearance < "
                  f"{TUNNEL_CLEARANCE * 1000:.0f} mm; tunnel entry at cost-to-go {f(st['tunnel_entry'], 1)}, key configuration (narrowest, "
                  f"{st.get('key_clearance_mm', '–')} mm) at {f(st['key'], 1)}, tunnel exit at {f(st['tunnel_exit'], 1)}; start {f(st['start_ctg'], 1)}. "
                  "Runs that never had a visible roadmap node report no cost-to-go (stage: start bubble) unless solved."]
    if excluded:
        lines += ["", "Excluded as harness-invalid (the model never acted on the puzzle; not counted as failures):"] + [f"* {e}" for e in excluded]
    table = "\n".join(lines)
    md = out_md.read_text() if out_md.exists() else ""
    start, end = "<!-- axes:start -->", "<!-- axes:end -->"
    block = f"{start}\n## Behaviour axes, every agent run to date\n\nGenerated by `scripts/analyze.py --axes`.\n\n{table}\n{end}"
    md = (md[:md.index(start)] + block + md[md.index(end) + len(end):]) if start in md else (md + "\n" + block + "\n")
    out_md.write_text(md)
    return table


if __name__ == "__main__":
    main()
