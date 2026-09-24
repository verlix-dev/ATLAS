"""End-to-end verification of the React ATLAS UI against the real backend.

    python tools/e2e_react.py                 # via the Vite dev server + proxy
    python tools/e2e_react.py --base http://127.0.0.1:8000   # via server.py/dist

Drives the ACTUAL application in headless Chrome over the DevTools Protocol --
no hand-built harness page, no bundle imported into a stub document. The app is
loaded from its own URL, the real textarea receives a real input event, and the
real START MISSION button is clicked.

Every claim is cross-checked against the backend's own JSON for the SAME
mission: the mission id is recovered from the browser's resource timings, then
/api/missions/<id>/summary is fetched directly and compared with what the DOM
renders. The UI therefore cannot pass by rendering something merely plausible.

Stdlib only -- the WebSocket client below exists so this adds no dependency.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
API = "http://127.0.0.1:8000"
TASK = "Predict whether a customer will churn. Missing a churner is more costly than a false alarm."

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  -- ' + str(detail)}")
    if not ok:
        failures.append(name)
    return bool(ok)


# --- minimal WebSocket + CDP client --------------------------------------------------


class WS:
    """RFC6455 client, only what CDP needs: text frames, client masking."""

    def __init__(self, url: str) -> None:
        u = urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port), timeout=30)
        self.sock.settimeout(180)
        key = base64.b64encode(os.urandom(16)).decode()
        path = u.path + (f"?{u.query}" if u.query else "")
        self.sock.sendall(
            (
                f"GET {path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        self.buf = b""
        while b"\r\n\r\n" not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("websocket handshake closed early")
            self.buf += chunk
        head, _, rest = self.buf.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n")[0] + b" ":
            raise RuntimeError(f"websocket handshake refused: {head[:120]!r}")
        self.buf = rest

    def _exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise RuntimeError("websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, text: str) -> None:
        payload = text.encode()
        n = len(payload)
        header = bytearray([0x81])
        if n < 126:
            header.append(0x80 | n)
        elif n < 1 << 16:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        self.sock.sendall(
            bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        )

    def recv(self) -> str:
        parts: list[bytes] = []
        while True:
            b0, b1 = self._exact(2)
            fin, opcode = b0 & 0x80, b0 & 0x0F
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._exact(8))[0]
            data = self._exact(n)
            if opcode == 0x8:
                raise RuntimeError("websocket closed by peer")
            if opcode == 0x9:  # ping -> pong, keeps long waits alive
                self.sock.sendall(b"\x8a\x80" + os.urandom(4))
                continue
            if opcode in (0x0, 0x1):
                parts.append(data)
                if fin:
                    return b"".join(parts).decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class CDP:
    def __init__(self, ws_url: str) -> None:
        self.ws = WS(ws_url)
        self._id = 0

    def call(self, method: str, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expression: str):
        r = self.call(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=True,
        )
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS threw: {json.dumps(r['exceptionDetails'])[:400]}")
        return r.get("result", {}).get("value")

    def wait(self, expression: str, timeout: float, label: str = "") -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.js(f"!!({expression})"):
                    return True
            except RuntimeError:
                pass  # the page may still be swapping documents
            time.sleep(0.25)
        return False

    def text(self) -> str:
        return self.js("document.body.innerText") or ""


# --- process helpers -----------------------------------------------------------------


def find_chrome() -> str | None:
    for path in CHROME_CANDIDATES:
        if Path(path).is_file():
            return path
    return shutil.which("chrome") or shutil.which("msedge")


def wait_http(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except (urllib.error.HTTPError,):
            return True  # answering at all is enough
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    return False


def get_json(url: str, payload: dict | None = None, timeout: float = 180):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


# --- the run -------------------------------------------------------------------------


def run(base: str, cdp: CDP) -> None:
    # 0. Backend truth, fetched independently of the browser.
    datasets = get_json(f"{API}/api/datasets")["datasets"]
    check("backend lists real datasets", bool(datasets), datasets)
    plan = get_json(f"{API}/api/plan", {"task": TASK})["plan"]
    check(
        "planner declines a prose target (the bug that blocked M7)",
        plan["target_candidate"] is None,
        f"target_candidate={plan['target_candidate']!r}",
    )
    metric = plan["primary_metric"]

    # 1. Load the real application.
    cdp.call("Page.navigate", url=base)
    ok = cdp.wait("document.body && document.body.innerText.includes('Configure your')", 40)
    if not check("app boots at " + base, ok, cdp.text()[:160]):
        return

    # 2. Datasets reached the UI through the proxy / static server.
    body = cdp.text()
    check(
        "dataset list rendered from GET /api/datasets",
        all(d["name"] in body for d in datasets),
        body[:200],
    )
    check("no demo data leaked in", "employee_attrition" not in body)

    # 3. Real task -> real POST /api/plan, rendered in "ATLAS Understands".
    cdp.js(
        "(() => { const ta = document.querySelector('textarea');"
        " Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set"
        f".call(ta, {json.dumps(TASK)});"
        " ta.dispatchEvent(new Event('input', { bubbles: true })); return true; })()"
    )
    shown = metric.replace("_", " ")
    ok = cdp.wait(
        f"document.body.innerText.toLowerCase().includes({json.dumps(shown)})", 25
    )
    check(f"planner panel shows the real primary metric ({metric})", ok, cdp.text()[:300])
    check(
        "planner reason came from the backend",
        plan["metric_reason"][:40] in cdp.text(),
        plan["metric_reason"],
    )

    # 4. Start the mission with a real click on the real button.
    enabled = cdp.js(
        "(() => { const b = [...document.querySelectorAll('button')]"
        ".find(x => /START MISSION/i.test(x.textContent || ''));"
        " return b ? !b.disabled : null; })()"
    )
    if not check("START MISSION is enabled", enabled is True, f"enabled={enabled}"):
        return
    cdp.js(
        "(() => { [...document.querySelectorAll('button')]"
        ".find(x => /START MISSION/i.test(x.textContent || '')).click(); return true; })()"
    )

    ok = cdp.wait("document.body.innerText.includes('Score Progression')", 45)
    if not check("mission accepted; dashboard reached", ok, cdp.text()[:300]):
        return

    # 5. The SSE stream actually drove the UI.
    check(
        "experiment card populated from real events",
        cdp.wait("/EXPERIMENT \\d/.test(document.body.innerText)", 30),
        cdp.text()[:200],
    )
    check(
        "score chart rendered",
        cdp.js("document.querySelectorAll('svg.recharts-surface').length") > 0,
    )
    check(
        "chart is labelled with the objective's own metric",
        metric.replace("_", " ").upper() in cdp.text().upper(),
    )

    ok = cdp.wait("document.body.innerText.includes('MISSION COMPLETE')", 180)
    check("mission_end handled (MISSION COMPLETE)", ok, cdp.text()[-300:])

    # 6. Cross-check the DOM against the backend's own summary for THIS mission.
    mission_id = cdp.js(
        "(() => { const e = performance.getEntriesByType('resource')"
        ".map(r => r.name).find(n => /\\/api\\/missions\\/[^/]+\\/events/.test(n));"
        " return e ? e.match(/\\/api\\/missions\\/([^/]+)\\/events/)[1] : null; })()"
    )
    if not check("recovered the real mission id from the browser", bool(mission_id), mission_id):
        return
    summary = get_json(f"{API}/api/missions/{mission_id}/summary")
    inner = summary["summary"]
    check("backend reports the mission finished", summary["done"] is True, summary)
    check("worker did not crash", summary["error"] is None, summary["error"])

    dom = cdp.text()
    best = inner["best_score"]
    check(
        f"best score in the DOM equals the backend's ({best})",
        best is not None and f"{best:.4f}" in dom,
        f"backend={best}",
    )
    check(
        f"experiment count matches the backend ({inner['experiments']})",
        f"{inner['experiments']}" in dom,
    )
    check(
        "stop reason came from the backend",
        bool(inner["stop_reason"]) and inner["stop_reason"][:30] in dom,
        inner["stop_reason"],
    )
    check(
        "objective metric matches the backend",
        inner["objective"]["primary_metric"] == metric,
        inner["objective"],
    )
    check("no fabricated models rendered", not any(
        w in dom for w in ("SMOTE", "XGBoost", "Stacked Ensemble")
    ))
    print(f"    backend summary: {json.dumps(inner)[:220]}")

    # 7. Comparison, through the real button and POST /api/compare.
    cdp.js(
        "(() => { const b = [...document.querySelectorAll('button')]"
        ".find(x => /Compare strategies/i.test(x.textContent || ''));"
        " if (!b || b.disabled) return false; b.click(); return true; })()"
    )
    ok = cdp.wait("document.body.innerText.includes('Learning Trajectories')", 300)
    if not check("comparison screen rendered from POST /api/compare", ok, cdp.text()[:300]):
        return
    dom = cdp.text()
    check("fairness asserted by the server", "Same Dataset object" in dom and "✓ yes" in dom)
    check(
        "both trajectories drawn",
        cdp.js("document.querySelectorAll('svg.recharts-surface').length") > 0,
    )
    check(
        "comparison reports a real verdict",
        any(w in dom for w in ("achieved the better score", "Both strategies tied", "not comparable")),
        dom[:200],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8443",
                    help="where the app is served (Vite dev server by default)")
    ap.add_argument("--port", type=int, default=9333, help="CDP port")
    args = ap.parse_args()

    chrome = find_chrome()
    if not chrome:
        print("No Chrome/Edge found; cannot run browser verification.")
        return 2
    if not wait_http(f"{API}/api/datasets", 5):
        print(f"Backend not answering at {API} -- start server.py first.")
        return 2
    if not wait_http(args.base, 5):
        print(f"App not answering at {args.base}.")
        return 2

    print(f"browser: {chrome}\napp:     {args.base}\napi:     {API}\n")
    profile = tempfile.mkdtemp(prefix="atlas-e2e-")
    browser = subprocess.Popen(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
         "--disable-extensions", "--window-size=1440,2400",
         f"--remote-debugging-port={args.port}", f"--user-data-dir={profile}",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    cdp = None
    try:
        if not wait_http(f"http://127.0.0.1:{args.port}/json/version", 30):
            print("Chrome did not expose the DevTools endpoint.")
            return 2
        targets = get_json(f"http://127.0.0.1:{args.port}/json/list")
        page = next((t for t in targets if t.get("type") == "page"), None)
        if not page:
            print("No page target in Chrome.")
            return 2
        cdp = CDP(page["webSocketDebuggerUrl"])
        run(args.base, cdp)
    finally:
        if cdp:
            cdp.ws.close()
        browser.terminate()
        try:
            browser.wait(timeout=10)
        except subprocess.TimeoutExpired:
            browser.kill()
        shutil.rmtree(profile, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("All end-to-end checks passed against the real backend.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
