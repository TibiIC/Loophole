#!/usr/bin/env python3
"""Scrub agent logs before they are archived into runs/ (and committed).

    python scripts/scrub_logs.py DIR [DIR ...]          # in place; prints what it removed
    from scrub_logs import scrub_tree                     # used by scripts/eval.py and scripts/opencode_rattrap.py

* Deletes OpenCode's internal state: `opencode.db*` (its session database), the `xdg-data` / `xdg-state` trees
  (database, git snapshots, logs), and a local `opencode-home` (packages, caches; its opencode.json is kept as
  `opencode-config.json`). They reproduce nothing; the transcript (`opencode.txt`, `trajectory.json`) and the
  simulator's own episode log are what the analysis reads.
* In every remaining text file, replaces anything matching a provider key pattern with `<redacted:KIND>`, and replaces
  environment dumps (a run of `NAME=value` lines, or a Python / JSON mapping of environment variables, as printed by an
  agent that dumped `os.environ` / `env` / `/proc/*/environ`) with `<environment dump removed>`, and the value of any
  credential-named variable (`*_API_KEY`, `*_TOKEN`, `*SECRET*`) with `<redacted:env>`, whatever its format.
Nothing we report depends on the removed content (docs/HARNESS_NOTES.md).
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

KEY_PATTERNS = {
    "google": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "openai": re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}"),
    "github": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "verda": re.compile(r"\bdc_[A-Za-z0-9]{20,}"),
    "nvidia": re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}"),
    "aws": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer": re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}"),
}
# an environment mapping: several 'NAME': 'value' pairs that include typical environment variables
ENV_MAP = re.compile(r"\{[^{}]{0,20000}?['\"](?:PATH|HOME|HOSTNAME|NVM_DIR|PYTHONPATH|SHLVL)['\"]\s*:\s*['\"][^{}]{0,20000}?\}")
# shell `env` output: 4+ consecutive NAME=value lines (in raw text, or as escaped \n inside JSON strings)
ENV_LINES = re.compile(r"(?:(?:^|\\n|\n)[A-Z][A-Z0-9_]{2,}=[^\n\\]*){4,}")
# /proc/<pid>/environ: NAME=value pairs separated by NUL, raw or JSON-escaped as \u0000
ENV_NUL = re.compile(r"(?:[A-Z][A-Z0-9_]{2,}=[^\x00\\\n\"]*(?:\x00|\\u0000)){3,}(?:[A-Z][A-Z0-9_]{2,}=[^\x00\\\n\"]*)?")
# the value of any credential-named variable, whatever its format (provider keys are not all AIza / sk- shaped)
SECRET_VAR = re.compile(r"\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)[A-Z0-9_]*\s*[=:]\s*['\"]?)([^\s'\"\\\x00,}]{8,})")
INTERNAL = ("xdg-data", "xdg-state")


def _env_lines(m: re.Match) -> str:
    g = m.group(0)
    sep = "\\n" if g.startswith("\\n") else ("\n" if g.startswith("\n") else "")
    return sep + "<environment dump removed>"


def scrub_text(s: str) -> tuple[str, dict]:
    n = {}
    s, k = ENV_MAP.subn("<environment dump removed>", s)
    n["env_map"] = k
    s, k = ENV_LINES.subn(_env_lines, s)
    n["env_lines"] = k
    s, k = ENV_NUL.subn("<environment dump removed>", s)
    n["env_nul"] = k
    s, k = SECRET_VAR.subn(lambda m: m.group(1) + "<redacted:env>", s)
    n["secret_var"] = k
    for kind, rx in KEY_PATTERNS.items():
        s, k = rx.subn(f"<redacted:{kind}>", s)
        if k:
            n[kind] = k
    return s, {a: b for a, b in n.items() if b}


def _walk(o, tally: dict):
    """Scrub every string inside a parsed JSON value (keeps the JSON valid)."""
    if isinstance(o, str):
        t, n = scrub_text(o)
        for a, b in n.items():
            tally[a] = tally.get(a, 0) + b
        return t
    if isinstance(o, list):
        return [_walk(x, tally) for x in o]
    if isinstance(o, dict):
        if sum(1 for k in o if isinstance(k, str) and k in ("PATH", "HOME", "HOSTNAME", "NVM_DIR", "SHLVL")) >= 2:
            tally["env_map"] = tally.get("env_map", 0) + 1
            return "<environment dump removed>"
        return {k: _walk(v, tally) for k, v in o.items()}
    return o


def scrub_file_text(p: Path, s: str) -> tuple[str, dict]:
    import json

    tally: dict = {}
    if p.suffix == ".json":
        try:
            t = json.dumps(_walk(json.loads(s), tally), indent=1, ensure_ascii=False) + "\n"
            return (t if tally else s), tally
        except ValueError:
            pass
    lines = s.split("\n")
    if any(l.lstrip().startswith("{") for l in lines[:50]):  # JSON lines (OpenCode's --format=json transcript)
        out = []
        for l in lines:
            if l.lstrip().startswith("{"):
                try:
                    o = json.loads(l)
                except ValueError:
                    t, n = scrub_text(l)
                    out.append(t)
                    for a, b in n.items():
                        tally[a] = tally.get(a, 0) + b
                    continue
                t = json.dumps(_walk(o, tally), ensure_ascii=False)
                out.append(t if tally else l)
            else:
                t, n = scrub_text(l)
                out.append(t)
                for a, b in n.items():
                    tally[a] = tally.get(a, 0) + b
        return ("\n".join(out) if tally else s), tally
    return scrub_text(s)


def scrub_tree(root: Path, verbose: bool = True) -> dict:
    root = Path(root)
    report = {"deleted": [], "scrubbed": {}}
    for home in sorted(root.rglob("opencode-home")):  # a local OpenCode home (scripts/opencode_rattrap.py): keep its config
        if home.is_dir():
            cfg = home / "config" / "opencode.json"
            if cfg.exists():
                shutil.copy(cfg, home.parent / "opencode-config.json")
            shutil.rmtree(home, ignore_errors=True)
            report["deleted"].append(str(home))
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name in INTERNAL:
            shutil.rmtree(p, ignore_errors=True)
            report["deleted"].append(str(p))
    for p in sorted(root.rglob("opencode.db*")):
        if p.is_file():
            p.unlink()
            report["deleted"].append(str(p))
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() in (".png", ".jpg", ".npz", ".mp4", ".gif"):
            continue
        try:
            s = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        t, n = scrub_file_text(p, s)
        if t != s:
            p.write_text(t)
            report["scrubbed"][str(p)] = n
    if verbose:
        print(f"scrub {root}: deleted {len(report['deleted'])} internal paths, scrubbed {len(report['scrubbed'])} files")
    return report


def main() -> int:
    for d in sys.argv[1:]:
        scrub_tree(Path(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
