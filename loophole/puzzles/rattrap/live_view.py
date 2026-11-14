"""Live run viewer for the Rat Trap (for us, never for the agent). Off unless explicitly enabled (docs/watching.md).

    LiveView(out_dir, port).attach(harness)       # after every action: render a frame, keep it, append it to disk

* Serves, on a separate port bound to 127.0.0.1 by default (never through MCP): `/` (one HTML page, no framework,
  polling once a second), `/frame.png` (the latest frame), `/status.json` (the overlay data), `/tail.json` (the last
  ten tool calls).
* Appends every frame to OUT/frames/NNNNNN.png and its overlay record to OUT/frames/frames.jsonl, so the run's video
  exists when it ends (`make watch-last` encodes it; nothing is re-simulated).
* Reads the episode only after an action has finished; it never steps the simulator, so physics, state hashes and
  scores are the same with it on or off (asserted by tests/test_rattrap_live_view.py). Nothing of it is ever returned
  to the agent: tool results and get_state are untouched.
"""

from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Rat Trap live</title>
<style>body{font:14px system-ui,sans-serif;margin:16px;background:#111;color:#eee}img{max-width:100%;border:1px solid #333}
.row{display:flex;gap:16px;flex-wrap:wrap}.k{color:#999}td{padding:2px 8px}#tail{font:12px ui-monospace,monospace;white-space:pre}
.blocked{color:#f96}.ok{color:#9e9}</style></head><body>
<div class="row"><img id="f" src="/frame.png" width="960"><table id="s"></table></div><h3>last ten tool calls</h3><div id="tail"></div>
<script>
async function tick(){
  try{
    const s=await (await fetch('/status.json',{cache:'no-store'})).json();
    document.getElementById('f').src='/frame.png?'+s.action;
    const b=s.last_blocked===null?'–':(s.last_blocked?'<span class=blocked>yes</span>':'<span class=ok>no</span>');
    document.getElementById('s').innerHTML=
      `<tr><td class=k>actions</td><td>${s.action} / ${s.budget}</td></tr>`+
      `<tr><td class=k>checkpoints</td><td>${s.checkpoints} / ${s.checkpoints_total}</td></tr>`+
      `<tr><td class=k>distance to next</td><td>${s.next_checkpoint_mm===null?'–':s.next_checkpoint_mm.toFixed(1)+' mm'}</td></tr>`+
      `<tr><td class=k>last move blocked</td><td>${b}</td></tr>`+
      `<tr><td class=k>last call</td><td>${s.last_call||'–'}</td></tr>`+
      `<tr><td class=k>state</td><td>${s.done?(s.solved?'solved':(s.violation?'violation':'ended')):'running'}</td></tr>`;
    const t=await (await fetch('/tail.json',{cache:'no-store'})).json();
    document.getElementById('tail').textContent=t.map(x=>`#${x.action} ${x.call}${x.blocked?'  (blocked)':''}`).join('\\n');
  }catch(e){}
}
setInterval(tick,1000);tick();
</script></body></html>"""


def summarize(tool: str, args: dict) -> str:
    a = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in args.items()}
    return f"{tool}(" + ", ".join(f"{k}={v}" for k, v in a.items()) + ")"


class LiveView:
    def __init__(self, out_dir: Path, port: int, host: str = "127.0.0.1", width: int = 480, height: int = 360):
        self.dir = Path(out_dir) / "frames"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.w, self.h = width, height
        self.lock = threading.Lock()
        self.png = b""
        self.status: dict = {"action": 0, "budget": None, "checkpoints": 0, "checkpoints_total": None, "next_checkpoint_mm": None,
                             "last_blocked": None, "last_call": None, "done": False, "solved": False, "violation": False}
        self.tail: list[dict] = []
        self.n = 0
        self.seconds = 0.0  # time spent rendering and writing (overhead measurement)
        self._log = open(self.dir / "frames.jsonl", "a", buffering=1)
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                with view.lock:
                    if self.path.startswith("/frame.png"):
                        return self._send(200, "image/png", view.png) if view.png else self._send(404, "text/plain", b"no frame yet")
                    if self.path.startswith("/status.json"):
                        return self._send(200, "application/json", json.dumps(view.status).encode())
                    if self.path.startswith("/tail.json"):
                        return self._send(200, "application/json", json.dumps(view.tail[-10:]).encode())
                if self.path in ("/", "/index.html"):
                    return self._send(200, "text/html; charset=utf-8", PAGE.encode())
                self._send(404, "text/plain", b"not found")

        self.server = ThreadingHTTPServer((host, int(port)), Handler)
        self.url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{self.server.server_address[1]}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def attach(self, harness) -> "LiveView":
        harness.live = self
        return self

    def on_action(self, ep, tool: str, args: dict, result: dict) -> None:
        """Called by the harness after an action has completed (and after its result was logged)."""
        from PIL import Image

        t0 = time.time()
        sim = ep.sim
        img = np.concatenate([sim.render("iso", distance=0.45, width=self.w, height=self.h),
                              sim.render("front", distance=0.4, width=self.w, height=self.h)], axis=1)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "PNG")
        png = buf.getvalue()
        nxt = None
        if ep.tracker is not None and ep.tracker.k < ep.tracker.cps.K:
            from .checkpoints import max_dev

            nxt = round(max_dev(sim.node_positions(), ep.tracker.cps.states[ep.tracker.k]) * 1000, 1)
        call = summarize(tool, args)
        blocked = bool(result.get("blocked")) if tool == "move" else None  # tool: the agent-facing name
        status = {"action": ep.actions, "budget": ep.T, "checkpoints": 0 if ep.tracker is None else ep.tracker.k,
                  "checkpoints_total": None if ep.tracker is None else ep.tracker.cps.K, "next_checkpoint_mm": nxt,
                  "last_blocked": blocked, "last_call": call, "done": ep.done, "solved": ep.solved, "violation": ep.violation}
        self.n += 1
        (self.dir / f"{self.n:06d}.png").write_bytes(png)
        self._log.write(json.dumps({"frame": self.n, **status}) + "\n")
        with self.lock:
            self.png, self.status = png, status
            self.tail.append({"action": ep.actions, "call": call, "blocked": blocked})
            self.tail = self.tail[-10:]
        self.seconds += time.time() - t0

    def close(self) -> None:
        self.server.shutdown()
        self._log.close()
