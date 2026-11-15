#!/usr/bin/env python3
"""Audit agent runs for harness-induced false failures (docs/HARNESS_NOTES.md):

* output-limit truncation: a model turn that used (nearly) its whole output budget and produced no tool call
  (OpenCode: finish reason "length", or reasoning + output tokens >= 97 % of the run's cap: OpenCode's 32,000 default,
  or the model's limit.output where the run raised it; OpenAI-compatible harness: finish_reason "length",
  or for older logs completion_tokens within 2 % of the request's max_tokens with no content and no tool call);
* permission blocks: a tool call denied by the harness (OpenCode "permission" / "denied" errors);
* unknown tools / harness errors: OpenCode error events (e.g. "Unexpected server error", API 400s), tool errors that
  name an unknown tool, or a run that ended before its first action.

    .venv/bin/python scripts/harness_audit.py [--runs runs] [--jobs ../Loophole/jobs] [--rattrap runs/rattrap-agents] [--md out.md]

By default it audits the committed logs: Harbor trials under runs/, and the Rat Trap runs under runs/rattrap-agents/
(aborted/ included).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENCODE_CAP = 32000  # OpenCode 1.19.5's default per-turn output cap (OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX)
# Harbor trials do not record their task's compose, so a raised cap is listed by job directory (relative to runs/)
HARBOR_CAP = {"harbor-rattrap/rattrap-gemini-output-cap": 65536}
# runs whose figures are harness artefacts or partial by design, labelled in the table (keyed by Run label prefix)
RUN_NOTES = {
    "harbor-rattrap/rattrap-gemini-output-cap/": "the output-cap fix through Harbor: 0 empty turns, 0 connection resets; ended "
                                                 "after 52 turns (34 min) when the provider rejected the API key (UNAUTHENTICATED); "
                                                 "not a full-budget run",
    "harbor-rattrap/rattrap-gemini-thinking-low/": "harness artefact: thinkingLevel low (the pinned workaround); 0 reasoning "
                                                   "tokens in every turn, then empty turns until the 14,400 s timeout. Fixed by "
                                                   "raising the output cap (docs/HARNESS_NOTES.md §1)",
    "aborted/google-gemini-4-carbon-opencode-20261114-2117": "harness artefact: opening plan truncated at OpenCode's 32,000 cap",
    "aborted/google-gemini-4-carbon-opencode-20261115-133646-0": "harness artefact: no includeThoughts, long turn lost to header timeouts",
    "google-gemini-4-carbon-opencode-20261115-140522-0": "50-step probe of the output-cap fix, stopped by us; not a scored run",
    "google-gemini-4-carbon-opencode-20261114-21374": "thinkingLevel low (the pinned workaround)",
}


def note(run: str) -> str:
    return next((v for k, v in RUN_NOTES.items() if run.startswith(k)), "")


def opencode_cap(cfg: dict) -> int:
    """The per-turn cap a run's OpenCode config set: limit.output of its model entry, else OpenCode's default."""
    caps = [m.get("limit", {}).get("output") for p in (cfg.get("provider") or {}).values() for m in (p.get("models") or {}).values()]
    return min([c for c in caps if c] or [OPENCODE_CAP])


def opencode_events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def audit_opencode(path: Path, model: str, task: str, reward: str | None, cap: int = OPENCODE_CAP) -> dict:
    ev = opencode_events(path)
    steps = [e for e in ev if e.get("type") == "step_finish"]
    tools = [e for e in ev if e.get("type") == "tool_use"]
    game = [t for t in tools if str(t["part"].get("tool", "")).startswith(("loophole_", "rattrap_"))]
    capped = []
    for e in steps:
        tk = e["part"].get("tokens") or {}
        if e["part"].get("reason") == "length" or (tk.get("reasoning") or 0) + (tk.get("output") or 0) >= 0.97 * cap:
            capped.append(tk)
    errors = [json.dumps(e.get("error"))[:200] for e in ev if e.get("type") == "error"]
    tool_err = [t for t in tools if (t["part"].get("state") or {}).get("status") == "error"]
    denied = [t for t in tool_err if re.search(r"permission|denied|not allowed", json.dumps(t["part"]["state"]), re.I)]
    unknown = [t for t in tool_err if re.search(r"unknown tool|no such tool|invalid tool|tool .{0,40}not found", json.dumps(t["part"]["state"]), re.I)
               and "File not found" not in json.dumps(t["part"]["state"])]
    acts = [t for t in game if str(t["part"]["tool"]).split("_", 1)[1] in ("grab", "move", "release", "wait")]
    # agents may also drive the server over raw HTTP from bash (OpenCode does not forward MCP images to Gemini)
    http = [t for t in tools if t["part"].get("tool") == "bash" and re.search(r"loophole-sim|:8000/mcp", json.dumps(t["part"].get("state", {}).get("input", {})))
            and re.search(r"\b(grab|move|release|wait)\b", json.dumps(t["part"].get("state", {}).get("input", {})))]
    empty = [e for e in steps if not ((e["part"].get("tokens") or {}).get("output") or (e["part"].get("tokens") or {}).get("reasoning"))]
    return {"task": task, "model": model, "run": path.parent.parent.name if path.parent.name == "agent" else path.parent.name, "reward": reward,
            "turns": len(steps), "empty_turns": len(empty), "game_calls": len(game), "actions": len(acts), "capped_turns": len(capped),
            "last_capped": bool(steps) and bool(capped) and steps[-1]["part"].get("tokens") == capped[-1], "permission_denied": len(denied),
            "unknown_tool": len(unknown), "error_events": errors[:3], "no_action": len(acts) == 0 and len(http) == 0, "http_action_calls": len(http)}


def audit_fc(run: Path) -> dict:
    rs = [json.loads(l) for l in (run / "transcript.jsonl").read_text().splitlines() if l.strip()]
    res = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {}
    replies = [r for r in rs if r["type"] == "reply"]
    max_tok = 32768 if "gemini" in run.name or "nemotron-2026111421" in run.name or (res.get("model") == "nemotron" and run.name > "nemotron-20261114-2140") else 8192
    capped = 0
    for r in replies:
        m, u = r["message"], r.get("usage") or {}
        empty = not (m.get("content") or "").strip() and not m.get("tool_calls")
        if r.get("finish_reason") == "length" or (empty and (u.get("completion_tokens") or 0) >= 0.98 * max_tok) or \
           (empty and (u.get("total_tokens") or 0) - (u.get("prompt_tokens") or 0) >= 0.98 * max_tok):
            capped += 1
    tools = [r for r in rs if r["type"] == "tool"]
    unknown = sum(1 for t in tools if "unknown tool" in json.dumps(t.get("result")))
    acts = [t for t in tools if t["name"] in ("grab", "move", "release")]
    empty = sum(1 for r in replies if not (r["message"].get("content") or "").strip() and not r["message"].get("tool_calls"))
    return {"task": "rattrap", "model": res.get("model_id") or res.get("model") or run.name, "run": run.name, "reward": res.get("score"),
            "turns": len(replies), "empty_turns": empty, "game_calls": len(tools), "actions": res.get("actions", len(acts)), "capped_turns": capped,
            "last_capped": bool(replies) and capped > 0 and (replies[-1].get("finish_reason") == "length"), "permission_denied": 0,
            "unknown_tool": unknown, "error_events": [res["error"][:200]] if res.get("error") else [], "no_action": len(acts) == 0,
            "end": res.get("end")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs", help="committed Harbor trials: runs/<run>/[opencode/]<trial>/")
    ap.add_argument("--jobs", type=Path, default=None, help="also audit a local Harbor jobs directory")
    ap.add_argument("--rattrap", type=Path, default=ROOT / "runs" / "rattrap-agents")
    ap.add_argument("--md", type=Path, default=None)
    a = ap.parse_args()
    rows = []
    trials = set(a.runs.glob("*/opencode/*/agent/opencode.txt")) | set(a.runs.glob("*/*__*/agent/opencode.txt")) \
        | set(a.runs.glob("*/*/*__*/agent/opencode.txt"))  # runs/<run>/opencode/<trial>, runs/<run>/<trial>, runs/<group>/<job>/<trial>
    for oc in sorted(trials):
        trial = oc.parent.parent
        cfg = json.loads((trial / "config.json").read_text()) if (trial / "config.json").exists() else {}
        rw = trial / "verifier" / "reward.txt"
        run_dir = trial.parent.parent if trial.parent.name == "opencode" else trial.parent
        label = f"{run_dir.relative_to(a.runs)}/{trial.name.split('__')[1]}"
        row = audit_opencode(oc, (cfg.get("agent") or {}).get("model_name") or "?", trial.name.split("__")[0],
                             rw.read_text().strip() if rw.exists() else None, HARBOR_CAP.get(str(run_dir.relative_to(a.runs)), OPENCODE_CAP))
        row["run"] = label
        ep = next(iter(trial.glob("artifacts/**/episode.jsonl")), None)  # the simulator's own log: ground truth
        rt = trial / "artifacts" / "episode" / "result.json"  # the Rat Trap simulator's own result
        if ep is not None:
            n = sum(1 for l in ep.read_text().splitlines() if '"type": "action"' in l or '"type":"action"' in l)
            row["actions"], row["no_action"] = n, n == 0
        elif rt.exists() and "actions" in (res := json.loads(rt.read_text())):
            row["actions"], row["no_action"] = res["actions"], res["actions"] == 0
        rows.append(row)
    for cfgf in sorted(a.jobs.glob("*/config.json")) if a.jobs else []:
        try:
            cfg = json.loads(cfgf.read_text())
        except Exception:
            continue
        models = [x.get("model_name") for x in cfg.get("agents", [])]
        if not models or not models[0]:
            continue
        for trial in sorted(cfgf.parent.glob("*/")):
            oc = trial / "agent" / "opencode.txt"
            if not oc.exists():
                continue
            rw = trial / "verifier" / "reward.txt"
            row = audit_opencode(oc, models[0], trial.name.split("__")[0], rw.read_text().strip() if rw.exists() else None)
            ep = next(iter(trial.glob("artifacts/**/episode.jsonl")), None)  # the simulator's own log: ground truth
            if ep is not None:
                n = sum(1 for l in ep.read_text().splitlines() if '"type": "action"' in l or '"type":"action"' in l)
                row["actions"], row["no_action"] = n, n == 0
            rows.append(row)
    for run in sorted(a.rattrap.glob("*/")) + sorted(a.rattrap.glob("aborted/*/")):
        if run.name == "aborted":
            continue
        if (run / "transcript.jsonl").exists():
            r = audit_fc(run)
        elif (run / "opencode.jsonl").exists():
            res = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {}
            cfg = json.loads((run / "opencode-config.json").read_text()) if (run / "opencode-config.json").exists() else {}
            r = audit_opencode(run / "opencode.jsonl", res.get("model") or run.name, "rattrap", res.get("score"), opencode_cap(cfg))
            if "actions" in res:  # the simulator's count, not tool calls
                r["actions"] = res["actions"]
        else:
            continue
        r["run"] = ("aborted/" if run.parent.name == "aborted" else "") + run.name
        rows.append(r)
    flagged = [r for r in rows if r["capped_turns"] or r["permission_denied"] or r["unknown_tool"] or r["error_events"] or r["no_action"]]
    print(json.dumps({"runs": len(rows), "flagged": len(flagged)}))
    for r in rows:
        mark = "FLAG" if r in flagged else "ok  "
        print(mark, json.dumps({k: r[k] for k in ("task", "model", "run", "reward", "turns", "actions", "capped_turns", "last_capped",
                                                   "permission_denied", "unknown_tool", "error_events", "no_action")}))
    if a.md:
        lines = ["# Harness audit", "",
                 "Generated by `scripts/harness_audit.py --md runs/harness-audit.md` over the committed logs (docs/HARNESS_NOTES.md).",
                 "*Reward* is the run's own record: Harbor's `verifier/reward.txt`, or the Rat Trap harness's score. The",
                 "`20261114-080705` agent trials read 0 there because the verifier image changed mid-eval; their reported",
                 "scores come from replay (docs/RESULTS.md). *Actions* is the simulator's count. *Turns at the output cap*",
                 "counts turns that ended with finish reason `length` or used at least 97 % of that run's cap. *Empty turns*",
                 "returned no text, no reasoning and no tool call. *Note* labels runs whose figures are harness artefacts.", "",
                 "| Task | Model | Run | Reward | Turns | Empty turns | Actions | Turns at the output cap | Permission denials | Unknown tool | Errors | No action | Note |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['task']} | {r['model']} | {r['run']} | {r['reward']} | {r['turns']} | {r['empty_turns']} | {r['actions']} | {r['capped_turns']}"
                         f"{' (last turn)' if r['last_capped'] else ''} | {r['permission_denied']} | {r['unknown_tool']} | "
                         f"{'; '.join(r['error_events'])[:120] or '–'} | {'yes' if r['no_action'] else ''} | {note(r['run'])} |")
        a.md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
