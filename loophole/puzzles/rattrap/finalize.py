"""Harbor collect hook for tasks/loophole-rattrap (runs inside the sim sidecar after the agent stops): asks the
server for its final record over the loopback-only /admin/finalize route and writes final.json next to the log.
Informational only: the verifier recomputes the score from commands.jsonl."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    port = os.environ.get("LOOPHOLE_PORT", "8000")
    out = Path(os.environ.get("LOOPHOLE_LOG_DIR", "/var/loophole/episode"))
    req = urllib.request.Request(f"http://127.0.0.1:{port}/admin/finalize", data=b"{}", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            final = json.loads(r.read())
    except Exception as e:  # server gone: the verifier replays whatever log exists
        print(f"finalize failed: {e}", file=sys.stderr)
        return 0
    (out / "final.json").write_text(json.dumps(final, indent=1))
    print(json.dumps({k: final.get(k) for k in ("score", "solved", "actions", "checkpoints_reached", "end")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
