"""ATLAS HTTP/SSE adapter — stdlib only.

    python server.py            # http://127.0.0.1:8000

This is a TRANSPORT LAYER AND NOTHING ELSE. It starts a Mission, forwards the
events that Mission already emits, and serves static files. It contains no ML
logic, no planning logic, no ranking, and no metric computation -- every one of
those lives in atlas/ and is called, never reimplemented.

Mission.events is the single UI contract, so events are forwarded verbatim:
this module never invents a field, reshapes a payload, or fills in a value the
backend did not produce.

Deliberately no FastAPI/uvicorn/Starlette/pydantic: ThreadingHTTPServer already
gives us the background-thread model a blocking pipeline.fit() needs, and six
hand-written endpoints do not justify the dependency.
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import threading
import traceback
import uuid
from dataclasses import asdict
from email.parser import BytesParser
from email.policy import default as EMAIL_POLICY
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import pandas as pd

from atlas import planner
from atlas.data_engineer import load
from atlas.decision import Budget
from atlas.loop import Mission

ROOT = Path(__file__).resolve().parent
# The built React bundle. Vite emits hashed assets that index.html references
# as absolute /assets/... paths, so the build output -- not ui/ -- is the
# document root. Run `npm run build` in ui/ before serving.
UI_DIR = ROOT / "ui" / "dist"
DATA_DIR = ROOT / "data"
# User uploads live outside data/ so the repo's demo CSVs can never be
# modified or shadowed by an upload. Git-ignored; created on first upload.
UPLOAD_DIR = ROOT / "uploads"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
HOST, PORT = "0.0.0.0", int(os.environ.get("PORT", 8000))
# The browser UI is normally Vite on this origin locally, and Vercel in
# production. Keep this intentionally specific rather than allowing every
# website to make state-changing API requests to a deployed ATLAS instance.
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

# SSE keep-alive. Not pacing: pacing is the browser's job, and adding a sleep
# here would invent compute time the engine never spent.
HEARTBEAT_SECONDS = 15.0

SAFE_TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
              ".json": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon"}


class MissionRun:
    """One mission plus the plumbing to stream its events to many readers.

    Mission.events is already an append-only log, so a late subscriber simply
    replays from index 0 and then follows. That removes any need for a broker,
    per-client queues, or a replay buffer.
    """

    def __init__(self, mission_id: str, mission: Mission) -> None:
        self.id = mission_id
        self.mission = mission
        self.events: list[dict] = []
        self.done = False
        self.error: str | None = None
        self._cond = threading.Condition()

    def sink(self, event: dict) -> None:
        """Called by Mission._emit on the worker thread, once per stage."""
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def finish(self, error: str | None = None) -> None:
        with self._cond:
            self.error = error
            self.done = True
            self._cond.notify_all()

    def follow(self, cursor: int):
        """Yield events from `cursor` onward, blocking until more arrive.

        Yields None as a keep-alive tick so the handler can emit an SSE comment
        and notice a dropped connection. Ends when the mission is done and the
        reader has caught up.
        """
        while True:
            with self._cond:
                while cursor >= len(self.events) and not self.done:
                    if not self._cond.wait(timeout=HEARTBEAT_SECONDS):
                        yield None  # heartbeat
                if cursor < len(self.events):
                    event = self.events[cursor]
                    cursor += 1
                else:
                    return
            yield event


MISSIONS: dict[str, MissionRun] = {}
MISSIONS_LOCK = threading.Lock()


# Token prefix per dataset source. The prefix is part of the API contract;
# the directory it maps to is configuration.
DATASET_PREFIX = {"builtin": "data", "upload": "uploads"}


def _dataset_dirs() -> dict[str, Path]:
    """Token prefix -> directory, read fresh on every call.

    Deliberately not a module constant: UPLOAD_DIR is configurable (and tests
    redirect it), so binding it once at import would freeze the wrong path.
    """
    return {"data": DATA_DIR, "uploads": UPLOAD_DIR}


def _dataset_row(path: Path, source: str) -> dict:
    """One row of /api/datasets. `csv` is the token the client sends back.

    The prefix comes from `source`, never from the directory's name on disk, so
    the token round-trips through resolve_dataset() wherever that directory
    actually lives.
    """
    return {
        "name": path.name,
        "csv": f"{DATASET_PREFIX[source]}/{path.name}",
        "bytes": path.stat().st_size,
        "source": source,
    }


def resolve_dataset(csv: str) -> Path:
    """Resolve a client-supplied dataset token to a real file, or raise.

    A token is exactly '<prefix>/<filename>'. The filename is joined to the
    directory the prefix names -- NOT to ROOT -- because UPLOAD_DIR need not sit
    under ROOT. Requiring exactly two segments rejects '../x.csv', 'data/../x',
    'C:/x.csv' and '/etc/passwd' before any filesystem access happens.

    Containment is then re-checked after resolve(), so traversal hidden in the
    filename and a symlink pointing outside the directory are both refused.
    """
    parts = PurePosixPath(str(csv).replace("\\", "/")).parts
    if len(parts) != 2:
        raise ValueError(f"dataset not available: {csv}")
    prefix, name = parts
    base = _dataset_dirs().get(prefix)
    if base is None:
        raise ValueError(f"dataset not available: {csv}")

    base = base.resolve()
    path = (base / name).resolve()
    if not path.is_file() or path.parent != base:
        raise ValueError(f"dataset not available: {csv}")
    return path


def _safe_stem(filename: str) -> str:
    """An inert display stem from the user's filename. NEVER a path.

    Takes the basename, drops the extension, and keeps only word characters, so
    '../../etc/passwd.csv' and 'C:\\evil.csv' both reduce to plain text. The
    caller appends a generated id, so the result cannot address an existing file.
    """
    base = PurePosixPath(filename.replace("\\", "/")).name
    stem = base.rsplit(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")[:40] or "upload"


def _first_file_part(content_type: str, body: bytes) -> tuple[str, bytes | None]:
    """Pull the first file part out of a multipart/form-data body.

    Uses `email` rather than `cgi`, which was removed in Python 3.13.
    """
    message = BytesParser(policy=EMAIL_POLICY).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body
    )
    if not message.is_multipart():
        return "", None
    for part in message.iter_parts():
        filename = part.get_filename()
        if filename:
            return filename, part.get_payload(decode=True) or b""
    return "", None


def _make_proposer(spec):
    """Resolve the `llm` field to a Callable[[dict], str], or None.

    Only the offline deterministic proposer is wired up here. A real client
    would be constructed the same way run_atlas.py does it; this server does
    not reach for the network on a demo machine.
    """
    if not spec:
        return None
    if spec in (True, "fake", "offline", "evidence-led"):
        from compare_strategies import EvidenceLedProposer

        return EvidenceLedProposer()
    raise ValueError(f"unknown llm proposer {spec!r}; use false or \"offline\"")


def start_mission(body: dict) -> str:
    """Build a Mission from the request and drive it on a background thread.

    load() and Mission() run on the CALLING thread so their errors (missing
    file, bad target) surface as a clean 400 instead of vanishing into a worker.
    """
    csv = body.get("csv") or "data/churn.csv"
    path = resolve_dataset(csv)

    task = (body.get("task") or "").strip()
    plan = planner.plan(task) if task else None

    target = body.get("target") or (plan.target_candidate if plan else None)
    budget_size = body.get("budget") or (plan.experiment_budget if plan else None) or 6
    budget = Budget(max_experiments=int(budget_size))

    mission_id = uuid.uuid4().hex[:12]
    run = MissionRun(mission_id, None)  # type: ignore[arg-type]

    # Data Engineer runs here: an unconfirmed target must fail loudly, now.
    dataset = load(path, target)
    run.mission = Mission(
        dataset, budget=budget, sink=run.sink, plan=plan, llm=_make_proposer(body.get("llm"))
    )

    with MISSIONS_LOCK:
        MISSIONS[mission_id] = run

    def drive() -> None:
        try:
            for _ in run.mission.run():
                pass
            run.finish()
        except Exception:  # noqa: BLE001 - a worker crash must reach the client
            run.finish(traceback.format_exc(limit=4))

    threading.Thread(target=drive, name=f"mission-{mission_id}", daemon=True).start()
    return mission_id


class Handler(BaseHTTPRequestHandler):
    server_version = "ATLAS"
    protocol_version = "HTTP/1.1"

    # ---- plumbing --------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}")

    def _cors_headers(self) -> dict[str, str]:
        """CORS headers for the one configured browser UI origin.

        Requests without an Origin header are non-browser/same-origin calls and
        do not need CORS. Rejecting unconfigured origins by omitting the header
        lets the browser enforce the deployment boundary without affecting the
        server's ordinary HTTP behaviour.
        """
        if self.headers.get("Origin") != FRONTEND_ORIGIN:
            return {}
        return {
            "Access-Control-Allow-Origin": FRONTEND_ORIGIN,
            "Vary": "Origin",
        }

    def _send(self, code: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in self._cors_headers().items():
            self.send_header(key, value)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _fail(self, code: int, message: str) -> None:
        self._json({"error": message}, code)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    def _run_for(self, path_parts: list[str]) -> MissionRun | None:
        with MISSIONS_LOCK:
            return MISSIONS.get(path_parts[2]) if len(path_parts) > 2 else None

    # ---- routing ---------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Answer browser preflight without consuming a request body."""
        self.send_response(204)
        for key, value in self._cors_headers().items():
            self.send_header(key, value)
        if self.headers.get("Origin") == FRONTEND_ORIGIN:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        parts = [p for p in route.split("/") if p]

        if route in ("/", "/index.html"):
            return self._static("index.html")
        if route == "/api/datasets":
            return self._datasets()
        if len(parts) == 4 and parts[:2] == ["api", "missions"]:
            run = self._run_for(parts)
            if run is None:
                return self._fail(404, "unknown mission (the server may have restarted)")
            if parts[3] == "events":
                return self._events(run)
            if parts[3] == "summary":
                return self._summary(run)
        if parts and parts[0] != "api":
            # Anything not under /api is a static asset request. _static()
            # resolves and confirms containment in UI_DIR, so traversal still
            # 404s; this only widens which paths reach that check.
            return self._static("/".join(parts))
        return self._fail(404, f"no route {route}")

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path

        # Multipart: the body is not JSON, so this must precede _body(), which
        # would consume the stream and fail to parse it.
        if route == "/api/datasets/upload":
            return self._upload()

        try:
            body = self._body()
        except ValueError as exc:
            return self._fail(400, str(exc))

        if route == "/api/plan":
            return self._plan(body)
        if route == "/api/missions":
            return self._create(body)
        if route == "/api/compare":
            return self._compare(body)
        return self._fail(404, f"no route {route}")

    # ---- endpoints -------------------------------------------------------

    def _datasets(self) -> None:
        """Demo datasets on disk, plus whatever the user has uploaded.

        Both are discovered from the filesystem -- no names are hardcoded, so an
        upload appears here the moment it lands.
        """
        rows = [_dataset_row(p, "builtin") for p in sorted(DATA_DIR.glob("*.csv"))]
        rows += [_dataset_row(p, "upload") for p in sorted(UPLOAD_DIR.glob("*.csv"))]
        self._json({"datasets": rows})

    def _upload(self) -> None:
        """POST /api/datasets/upload -- one multipart CSV.

        Validates, then stores under a generated name. This endpoint does NOT
        profile the file: rows and columns are counted facts, while the target,
        problem type and class balance remain the Data Engineer's to determine
        when the mission starts.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return self._fail(400, "no file was uploaded")
        if length > MAX_UPLOAD_BYTES:
            return self._fail(413, f"file exceeds the {MAX_UPLOAD_BYTES // (1 << 20)} MB limit")
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            return self._fail(400, "expected a multipart/form-data upload")

        filename, payload = _first_file_part(content_type, self.rfile.read(length))
        if payload is None:
            return self._fail(400, "no file part in the upload")
        if not filename.lower().endswith(".csv"):
            return self._fail(400, f"only .csv files are accepted, got {filename!r}")
        if not payload.strip():
            return self._fail(400, "the uploaded file is empty")

        # Parse before storing, so an unreadable file never reaches disk.
        try:
            frame = pd.read_csv(io.BytesIO(payload))
        except Exception as exc:  # noqa: BLE001 - any parse failure is a 400
            return self._fail(400, f"could not parse as CSV: {type(exc).__name__}: {exc}")
        if frame.empty or frame.shape[1] == 0:
            return self._fail(400, "the CSV parsed but has no data rows")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        # Generated id, so an upload can never overwrite an existing file.
        stored = UPLOAD_DIR / f"{_safe_stem(filename)}-{uuid.uuid4().hex[:8]}.csv"
        stored.write_bytes(payload)

        row = _dataset_row(stored, "upload")
        row["rows"] = int(frame.shape[0])
        row["columns"] = int(frame.shape[1])
        self._json({"dataset": row}, 201)

    def _plan(self, body: dict) -> None:
        """Planner preview. Pure, offline, instant -- runs no experiment."""
        task = (body.get("task") or "").strip()
        if not task:
            return self._json({"plan": None})
        try:
            self._json({"plan": asdict(planner.plan(task))})
        except ValueError as exc:
            self._fail(400, str(exc))

    def _create(self, body: dict) -> None:
        try:
            mission_id = start_mission(body)
        except (FileNotFoundError, ValueError) as exc:
            return self._fail(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._fail(500, f"{type(exc).__name__}: {exc}")
        self._json({"mission_id": mission_id}, 201)

    def _summary(self, run: MissionRun) -> None:
        self._json(
            {
                "mission_id": run.id,
                "done": run.done,
                "error": run.error,
                "summary": run.mission.summary(),
            }
        )

    def _compare(self, body: dict) -> None:
        """M6 comparison. Delegates wholly to compare_strategies.run_comparison().

        Both strategies already share one loaded Dataset object inside that
        function -- one load(), one split -- so fairness is guaranteed upstream
        and this endpoint only serialises the result. Runs synchronously (a few
        seconds): there is nothing to stream, both missions are already over.
        """
        from compare_strategies import run_comparison

        csv = body.get("csv") or "data/churn.csv"
        try:
            path = resolve_dataset(csv)
        except ValueError as exc:
            return self._fail(400, str(exc))

        budget = Budget(max_experiments=int(body.get("budget") or 6))
        try:
            dataset = load(path, body.get("target"))
            report = run_comparison(dataset, budget)
        except (FileNotFoundError, ValueError) as exc:
            return self._fail(400, str(exc))

        deterministic, guided = report["deterministic"], report["guided"]
        self._json(
            {
                "csv": csv,
                "target": dataset.profile.target,
                "budget": asdict(budget),
                # Fairness, asserted from the objects themselves rather than claimed.
                "same_dataset_object": deterministic.data is guided.data,
                "same_objective": deterministic.objective is guided.objective,
                "train_rows": int(dataset.X_train.shape[0]),
                "test_rows": int(dataset.X_test.shape[0]),
                "deterministic": {
                    "events": deterministic.events,
                    "summary": deterministic.summary(),
                },
                "guided": {
                    "events": guided.events,
                    "summary": guided.summary(),
                },
                "novel": [
                    {
                        "id": e.id,
                        "model": e.config.model,
                        "params": e.config.params,
                        "preprocess": e.config.preprocess,
                    }
                    for e in report["novel"]
                ],
                "rejections": report["rejections"],
            }
        )

    def _events(self, run: MissionRun) -> None:
        """SSE. Replays from the start, then follows live.

        Because Mission.events is append-only and buffered, a client that
        connects late (or reconnects) sees the whole run with no gap. The
        Last-Event-ID header resumes mid-stream.
        """
        try:
            cursor = int(self.headers.get("Last-Event-ID", "")) + 1
        except ValueError:
            cursor = 0

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        for key, value in self._cors_headers().items():
            self.send_header(key, value)
        self.end_headers()

        try:
            for event in run.follow(cursor):
                if event is None:
                    self.wfile.write(b": keep-alive\n\n")
                else:
                    payload = json.dumps(event)
                    frame = f"id: {event['seq']}\nevent: {event['stage']}\ndata: {payload}\n\n"
                    self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()

            # Terminal frame: lets the client close cleanly and report a crash.
            done = json.dumps({"done": True, "error": run.error})
            self.wfile.write(f"event: done\ndata: {done}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away; nothing to clean up

    def _static(self, relative: str) -> None:
        target = (UI_DIR / relative).resolve()
        if UI_DIR not in target.parents or not target.is_file():
            return self._fail(404, f"no file {relative}")
        suffix = target.suffix.lower()
        if suffix not in SAFE_TYPES:
            return self._fail(403, f"refusing to serve {suffix}")
        ctype = SAFE_TYPES[suffix]
        if suffix in (".html", ".css", ".js", ".json", ".svg"):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)


def main() -> int:
    mimetypes.init()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"ATLAS UI   http://{HOST}:{PORT}")
    print(f"static     {UI_DIR}")
    print("adapter only - all ML logic lives in atlas/\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
