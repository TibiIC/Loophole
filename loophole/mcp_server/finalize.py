"""Collect hook (runs inside the sim sidecar): end the episode and write its final record.

Harbor runs this via [[verifier.collect]] before collecting the sidecar's episode
directory as an artifact. The request goes to the loopback admin route, which
agents (in another container) cannot reach.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    port = os.environ.get("LOOPHOLE_PORT", "8000")
    out_dir = Path(os.environ.get("LOOPHOLE_LOG_DIR", "/var/loophole/episode"))
    req = urllib.request.Request(f"http://127.0.0.1:{port}/admin/finalize", data=b"{}", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            final = json.loads(r.read())
    except Exception as e:  # server gone: the verifier will replay whatever log exists
        print(f"finalize failed: {e}", file=sys.stderr)
        return 0
    (out_dir / "final.json").write_text(json.dumps(final, indent=1))
    print(json.dumps({k: final[k] for k in ("score", "success", "actions_used", "done_reason")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
