#!/usr/bin/env python3
"""Audit agent runs for harness-induced false failures (docs/HARNESS_NOTES.md):

* output-limit truncation: a model turn that used (nearly) its whole output budget and produced no tool call
  (OpenCode: reasoning + output tokens >= 31,000 of its 32k cap; OpenAI-compatible harness: finish_reason "length",
  or for older logs completion_tokens within 2 % of the request's max_tokens with no content and no tool call);
* permission blocks: a tool call denied by the harness (OpenCode "permission" / "denied" errors);
* unknown tools / harness errors: OpenCode error events (e.g. "Unexpected server error", API 400s), tool errors that
  name an unknown tool, or a run that ended before its first action.

    .venv/bin/python scripts/harness_audit.py [--jobs ../Loophole/jobs] [--rattrap runs/rattrap-agents] [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = 31000


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


def audit_opencode(path: Path, model: str, task: str, reward: str | None) -> dict:
    ev = opencode_events(path)
    steps = [e for e in ev if e.get("type") == "step_finish"]
    tools = [e for e in ev if e.get("type") == "tool_use"]
    game = [t for t in tools if str(t["part"].get("tool", "")).startswith(("loophole_", "rattrap_"))]
    capped = []
    for e in steps:
        tk = e["part"].get("tokens") or {}
        if (tk.get("reasoning") or 0) + (tk.get("output") or 0) >= CAP:
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
    return {"task": task, "model": model, "run": path.parent.parent.name if path.parent.name == "agent" else path.parent.name, "reward": reward,
            "turns": len(steps), "game_calls": len(game), "actions": len(acts), "capped_turns": len(capped),
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
    return {"task": "rattrap", "model": res.get("model_id") or res.get("model") or run.name, "run": run.name, "reward": res.get("score"),
            "turns": len(replies), "game_calls": len(tools), "actions": len(acts), "capped_turns": capped,
            "last_capped": bool(replies) and capped > 0 and (replies[-1].get("finish_reason") == "length"), "permission_denied": 0,
            "unknown_tool": unknown, "error_events": [res["error"][:200]] if res.get("error") else [], "no_action": len(acts) == 0,
            "end": res.get("end")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, default=ROOT.parent / "Loophole" / "jobs")
    ap.add_argument("--rattrap", type=Path, default=ROOT / "runs" / "rattrap-agents")
    ap.add_argument("--md", type=Path, default=None)
    a = ap.parse_args()
    rows = []
    for cfgf in sorted(a.jobs.glob("*/config.json")):
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
    for run in sorted(a.rattrap.glob("*/")):
        if (run / "transcript.jsonl").exists():
            rows.append(audit_fc(run))
        elif (run / "opencode.jsonl").exists():
            res = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {}
            r = audit_opencode(run / "opencode.jsonl", res.get("model") or run.name, "rattrap", res.get("score"))
            r["run"] = run.name
            rows.append(r)
    flagged = [r for r in rows if r["capped_turns"] or r["permission_denied"] or r["unknown_tool"] or r["error_events"] or r["no_action"]]
    print(json.dumps({"runs": len(rows), "flagged": len(flagged)}))
    for r in rows:
        mark = "FLAG" if r in flagged else "ok  "
        print(mark, json.dumps({k: r[k] for k in ("task", "model", "run", "reward", "turns", "actions", "capped_turns", "last_capped",
                                                   "permission_denied", "unknown_tool", "error_events", "no_action")}))
    if a.md:
        lines = ["| Task | Model | Run | Reward | Turns | Actions | Turns at the output cap | Permission denials | Unknown tool | Errors | No action |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['task']} | {r['model']} | {r['run']} | {r['reward']} | {r['turns']} | {r['actions']} | {r['capped_turns']}"
                         f"{' (last turn)' if r['last_capped'] else ''} | {r['permission_denied']} | {r['unknown_tool']} | "
                         f"{'; '.join(r['error_events'])[:120] or '–'} | {'yes' if r['no_action'] else ''} |")
        a.md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
