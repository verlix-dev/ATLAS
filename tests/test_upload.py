"""Upload verification: a user CSV reaches the same pipeline the demo CSVs do.

The upload endpoint is a transport concern, so these tests drive it over real
HTTP against a live server on an ephemeral port. UPLOAD_DIR is redirected into
tmp_path, so nothing here touches the repo's uploads/ or data/ directories.

What is deliberately NOT tested here: target inference, profiling, metrics and
ranking. Those belong to the Data Engineer and the loop, are already covered,
and must behave identically whatever directory the CSV came from -- which is
the point of test_uploaded_dataset_runs_the_normal_pipeline below.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import server
from atlas.data_engineer import TargetConfidence, load

# 24 rows, 2 classes, one categorical and two numeric features: enough for the
# Data Engineer to profile and for a real experiment to train on.
GOOD_CSV = "tenure,charge,plan,left\n" + "".join(
    f"{i},{10 + i * 2},{'basic' if i % 2 else 'pro'},{'yes' if i % 3 == 0 else 'no'}\n"
    for i in range(24)
)
# Row 2 has more fields than the header: pandas raises ParserError.
MALFORMED_CSV = "a,b\n1,2\n3,4,5,6,7\n"


@pytest.fixture()
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real server whose upload directory is a throwaway."""
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def multipart(filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = f"----atlas{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def post_file(base: str, filename: str, content: bytes):
    """Returns (status, payload). A rejection is a result, not an exception."""
    body, content_type = multipart(filename, content)
    request = urllib.request.Request(
        f"{base}/api/datasets/upload", data=body, headers={"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def get(base: str, path: str):
    with urllib.request.urlopen(f"{base}{path}", timeout=60) as response:
        return json.loads(response.read())


def post_json(base: str, path: str, payload: dict):
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# --- accepting a good file -----------------------------------------------------------


def test_valid_csv_upload_succeeds(live: str) -> None:
    status, payload = post_file(live, "customer_data.csv", GOOD_CSV.encode())
    assert status == 201, payload
    dataset = payload["dataset"]
    assert dataset["source"] == "upload"
    assert dataset["csv"].startswith("uploads/")
    assert dataset["csv"].endswith(".csv")
    # Counted from the parsed frame, not guessed from the request.
    assert dataset["rows"] == 24
    assert dataset["columns"] == 4
    assert dataset["bytes"] == len(GOOD_CSV.encode())


def test_uploaded_csv_appears_in_discovery(live: str) -> None:
    _, payload = post_file(live, "customer_data.csv", GOOD_CSV.encode())
    token = payload["dataset"]["csv"]

    rows = get(live, "/api/datasets")["datasets"]
    assert token in [r["csv"] for r in rows]
    assert next(r for r in rows if r["csv"] == token)["source"] == "upload"


def test_builtin_datasets_still_listed_and_unchanged(live: str) -> None:
    """The three demo CSVs keep working exactly as before the upload feature."""
    post_file(live, "customer_data.csv", GOOD_CSV.encode())

    rows = get(live, "/api/datasets")["datasets"]
    builtin = {r["csv"]: r for r in rows if r["source"] == "builtin"}
    assert {"data/churn.csv", "data/fraud.csv", "data/houses.csv"} <= set(builtin)
    for row in builtin.values():
        assert row["csv"].startswith("data/")
        assert row["bytes"] > 0


# --- rejecting a bad one -------------------------------------------------------------


def test_empty_file_is_rejected(live: str) -> None:
    status, payload = post_file(live, "empty.csv", b"")
    assert status == 400
    assert "empty" in payload["error"].lower()


def test_whitespace_only_file_is_rejected(live: str) -> None:
    status, payload = post_file(live, "blank.csv", b"   \n\n  \n")
    assert status == 400
    assert "empty" in payload["error"].lower()


def test_non_csv_extension_is_rejected(live: str) -> None:
    status, payload = post_file(live, "payload.exe", b"MZ\x90\x00binary")
    assert status == 400
    assert ".csv" in payload["error"]


def test_malformed_csv_is_rejected_with_a_reason(live: str) -> None:
    status, payload = post_file(live, "broken.csv", MALFORMED_CSV.encode())
    assert status == 400
    assert "could not parse as CSV" in payload["error"]


def test_header_only_csv_is_rejected(live: str) -> None:
    status, payload = post_file(live, "headers.csv", b"a,b,c\n")
    assert status == 400
    assert "no data rows" in payload["error"]


def test_missing_multipart_body_is_rejected(live: str) -> None:
    request = urllib.request.Request(
        f"{live}/api/datasets/upload", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, payload = response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        status, payload = exc.code, json.loads(exc.read())
    assert status == 400
    assert "no file was uploaded" in payload["error"]


# --- the filename is never a path ----------------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "../../evil.csv",
        "../../../../../../tmp/evil.csv",
        "..\\..\\windows\\evil.csv",
        "/etc/passwd.csv",
        "....//....//evil.csv",
    ],
)
def test_hostile_filename_cannot_escape_the_upload_directory(
    live: str, tmp_path: Path, hostile: str
) -> None:
    status, payload = post_file(live, hostile, GOOD_CSV.encode())
    assert status == 201, payload

    # The invariant is containment in the CURRENT upload directory, so the
    # token is resolved the way the server resolves it -- not rebuilt from ROOT,
    # which would only be correct when UPLOAD_DIR happens to sit under it.
    stored = server.resolve_dataset(payload["dataset"]["csv"])
    assert stored.parent == (tmp_path / "uploads").resolve()
    assert stored.is_file()
    assert ".." not in payload["dataset"]["name"]
    assert "/" not in payload["dataset"]["name"]
    # Nothing was written outside that directory, here or in the repo.
    assert not (tmp_path / "evil.csv").exists()
    assert not (server.ROOT / "evil.csv").exists()


def test_absolute_path_token_is_refused(live: str) -> None:
    """An existing file outside the dataset directories is still not a dataset."""
    for token in (
        str(server.ROOT / "server.py"),
        str(server.DATA_DIR / "churn.csv"),  # real file, but addressed absolutely
        "C:/Windows/win.ini",
        "/etc/passwd",
    ):
        with pytest.raises(ValueError, match="dataset not available"):
            server.resolve_dataset(token)


def test_builtin_token_still_resolves(live: str) -> None:
    path = server.resolve_dataset("data/churn.csv")
    assert path == (server.DATA_DIR / "churn.csv").resolve()
    assert path.is_file()


def test_nonexistent_token_is_refused(live: str) -> None:
    for token in ("data/nope.csv", "uploads/nope.csv"):
        with pytest.raises(ValueError, match="dataset not available"):
            server.resolve_dataset(token)


def test_dataset_token_outside_the_allowed_dirs_is_refused(live: str) -> None:
    for token in ("../server.py", "data/../server.py", "uploads/../../server.py", "server.py"):
        with pytest.raises(ValueError, match="dataset not available"):
            server.resolve_dataset(token)


def test_uploads_never_touch_the_demo_datasets(live: str) -> None:
    before = {p.name: p.read_bytes() for p in server.DATA_DIR.glob("*.csv")}
    post_file(live, "churn.csv", GOOD_CSV.encode())  # same name as a demo file
    after = {p.name: p.read_bytes() for p in server.DATA_DIR.glob("*.csv")}
    assert before == after


def test_two_uploads_of_the_same_name_do_not_collide(live: str) -> None:
    _, first = post_file(live, "same.csv", GOOD_CSV.encode())
    _, second = post_file(live, "same.csv", GOOD_CSV.encode())
    assert first["dataset"]["csv"] != second["dataset"]["csv"]


# --- the uploaded file enters the existing pipeline ----------------------------------


def test_target_inference_still_belongs_to_the_data_engineer(live: str) -> None:
    """No target supplied: the Data Engineer infers it, as for any dataset."""
    _, payload = post_file(live, "customer_data.csv", GOOD_CSV.encode())
    path = server.resolve_dataset(payload["dataset"]["csv"])

    data = load(path)  # target omitted
    assert data.profile.target == "left"  # last column, inferred -- not passed in
    assert data.profile.target_confidence is TargetConfidence.INFERRED
    assert data.profile.rows == 24


def test_explicit_target_is_still_honoured_for_an_upload(live: str) -> None:
    _, payload = post_file(live, "customer_data.csv", GOOD_CSV.encode())
    path = server.resolve_dataset(payload["dataset"]["csv"])

    data = load(path, "plan")
    assert data.profile.target == "plan"
    assert data.profile.target_confidence is TargetConfidence.EXPLICIT


def test_uploaded_dataset_runs_the_normal_pipeline(live: str) -> None:
    """The whole point: an upload is just another dataset to the mission flow."""
    _, payload = post_file(live, "customer_data.csv", GOOD_CSV.encode())
    token = payload["dataset"]["csv"]

    status, created = post_json(
        live, "/api/missions", {"csv": token, "task": "Detect who left.", "budget": 2}
    )
    assert status == 201, created
    mission_id = created["mission_id"]

    summary = None
    for _ in range(600):  # the engine finishes in seconds; this is a ceiling
        summary = get(live, f"/api/missions/{mission_id}/summary")
        if summary["done"]:
            break
    assert summary is not None and summary["done"], summary
    assert summary["error"] is None, summary["error"]

    inner = summary["summary"]
    assert inner["experiments"] >= 1
    assert inner["objective"]["primary_metric"]
    # A real experiment ran on the uploaded file and produced a real timeline.
    assert inner["timeline"], inner
    assert inner["timeline"][0]["model"]
