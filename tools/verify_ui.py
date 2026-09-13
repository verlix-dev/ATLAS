"""Browser-level verification for the ATLAS UI.

    python tools/verify_ui.py

Drives headless Chrome (already installed -- no npm package, no new dependency)
against a real server, and asserts on the DOM *after* JavaScript has run.

Every expectation is checked against the backend's own /api/summary for the
same mission, so the UI cannot pass by rendering something plausible: it has to
render what ATLAS actually measured.

Writes screenshots to tools/screens/ for eyeballing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "tools" / "screens"
BASE = "http://127.0.0.1:8000"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else ' -- ' + detail}")
    if not ok:
        failures.append(name)
    return ok


def find_chrome() -> str | None:
    for path in CHROME_CANDIDATES:
        if Path(path).is_file():
            return path
    return shutil.which("chrome") or shutil.which("msedge")


def api(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def dump_dom(chrome: str, url: str, shot: str | None = None, budget_ms: int = 25000) -> str:
    """Render a URL and return the post-JavaScript DOM.

    --virtual-time-budget fast-forwards timers, so the client-side pacing
    (700ms per stage) resolves deterministically instead of being raced.
    """
    args = [
        chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-first-run", "--disable-extensions", "--hide-scrollbars",
        f"--virtual-time-budget={budget_ms}", "--window-size=1600,1200",
    ]
    if shot:
        SHOTS.mkdir(parents=True, exist_ok=True)
        args.append(f"--screenshot={SHOTS / shot}")
    args += ["--dump-dom", url]
    result = subprocess.run(args, capture_output=True, text=True, timeout=120,
                            encoding="utf-8", errors="replace")
    return result.stdout or ""


def text_of(dom: str) -> str:
    """Strip tags so assertions read like what a human sees."""
    body = re.sub(r"<script.*?</script>", " ", dom, flags=re.S)
    body = re.sub(r"<style.*?</style>", " ", body, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def main() -> int:
    chrome = find_chrome()
    if not chrome:
        print("No Chrome/Edge found; cannot run browser verification.")
        return 2
    print(f"browser: {chrome}\n")

    server = subprocess.Popen(
        [sys.executable, str(ROOT / "server.py")],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    threading.Thread(target=lambda: [_ for _ in server.stdout], daemon=True).start()
    try:
        for _ in range(40):
            try:
                api("/api/datasets")
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.25)
        else:
            print("server never came up")
            return 2

        return run_checks(chrome)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


def run_checks(chrome: str) -> int:
    # ---------------------------------------------------------------- setup
    print("SETUP SCREEN + PLANNER PREVIEW")
    # Task and dataset must agree: a fraud task beside churn.csv is a
    # contradiction in the demo state, not a demonstration of anything.
    task = "Detect fraudulent transactions. Missing fraud is more costly than a false alarm. Try at most 4 experiments."
    plan = api("/api/plan", {"task": task})["plan"]
    from urllib.parse import quote

    dom = dump_dom(chrome, f"{BASE}/?task={quote(task)}", "1-setup.png", 8000)
    body = text_of(dom)
    check("setup renders", "Start a" in body and "ATLAS understands" in body)
    check("dataset picker populated from /api/datasets",
          "churn.csv" in body and "fraud.csv" in body)
    check("planner preview shows the REAL inferred metric",
          plan["primary_metric"].replace("_", " ") in body, plan["primary_metric"])
    check("planner preview shows the real reason verbatim",
          plan["metric_reason"][:50] in body)
    check("planner preview shows confidence", plan["metric_confidence"] in body)
    check("planner preview shows priority", (plan["priority"] or "") in body)
    # A typed task must never leave the panel claiming no task exists.
    check("planner state reflects the typed task", "no task yet" not in body)
    check("LLM toggle present and defaults off", 'role="switch"' in dom and 'aria-checked="false"' in dom)

    # ----------------------------------------------------------- dashboard
    print("\nLIVE DASHBOARD (llm ON, budget 4)")
    mission = api("/api/missions", {"csv": "data/churn.csv", "target": "churned",
                                    "budget": 4, "llm": "offline"})["mission_id"]
    for _ in range(60):
        summary = api(f"/api/missions/{mission}/summary")
        if summary["done"]:
            break
        time.sleep(0.25)
    truth = summary["summary"]

    dom = dump_dom(chrome, f"{BASE}/?mission={mission}", "2-dashboard.png", 30000)
    body = text_of(dom)

    check("SSE delivered events to the browser", "EXPERIMENT" in body)
    check("mission reached the end in-browser", "MISSION COMPLETE" in body.upper()
          or "Mission complete" in body)

    for row in truth["timeline"]:
        shown = f"{row['score']:.4f}"
        check(f"experiment {row['id']} measured score {shown} rendered",
              shown in body, shown)

    check("best score matches backend summary",
          f"{truth['best_score']:.4f}" in body, f"{truth['best_score']:.4f}")
    check("stop reason matches backend summary", truth["stop_reason"] in body)
    check("progress shows a ceiling, not a false total", "of \u22644" in body or "of ≤4" in body)

    # measured vs reasoned, structurally
    check("MEASURED block present", 'class="stage-block measured"' in dom)
    check("REASONED block present", 'class="stage-block reasoned"' in dom)
    check("measured and reasoned are different elements",
          dom.count("stage-block measured") >= 1 and dom.count("stage-block reasoned") >= 1)
    check("decision chip rendered with an action", 'class="decision-chip"' in dom
          and 'data-action=' in dom)
    check("provenance badge rendered", 'class="badge" data-source=' in dom
          or 'data-source="llm"' in dom)
    check("connector between stages rendered", 'class="connector' in dom)
    check("collapsed history rows exist", 'class="collapsed"' in dom)
    check("score trend svg rendered (no chart library)", "<svg" in dom and "polyline" in dom)

    # Citations must resolve. Gate 8 of the validator accepts either an exact
    # diagnosis finding OR an evidence key (when findings is empty), so both
    # kinds must render -- and a finding citation must match a rendered finding.
    cites = re.findall(r'data-cite="([^"]+)"', dom)
    cite_keys = re.findall(r'data-cite-key="([^"]+)"', dom)
    findings = re.findall(r'data-finding="([^"]+)"', dom)
    check("at least one citation rendered", bool(cites or cite_keys),
          "no data-cite or data-cite-key found")
    check("every finding citation matches a rendered finding exactly",
          all(c in findings for c in cites),
          f"cites={cites[:2]} findings={findings[:2]}")
    if cite_keys:
        check("evidence-key citations carry their measured value",
              'class="cite-key-value"' in dom, "bare key with no number")

    # ------------------------------------------------------ honesty rules
    print("\nHONESTY RULES")
    # Structural: measured values live in solid-border blocks, reasoned prose in
    # dashed ones. Asserting on the rendered word would be case-fragile.
    check("measured block present and distinct from reasoned",
          'class="stage-block measured"' in dom and 'class="stage-block reasoned"' in dom)
    check("no metric cell renders a fabricated 0 for an unavailable value",
          'data-unavailable="true"' not in dom or "N/A" in body,
          "an unavailable metric rendered without N/A")
    check("rejected proposals labelled as never-ran",
          "never ran" in body or "rejected" not in body.lower())

    # -------------------------------------------------------- blocked path
    print("\nBLOCKED MISSION (Data Engineer refusal)")
    blocked = api("/api/missions", {"csv": "data/houses.csv", "budget": 3})["mission_id"]
    time.sleep(1.5)
    dom = dump_dom(chrome, f"{BASE}/?mission={blocked}", "3-blocked.png", 8000)
    body = text_of(dom)
    check("blocked mission renders as blocked", "MISSION BLOCKED" in body.upper())
    check("refusal reason shown verbatim", "not a confident classification target" in body)
    # Assert on structure, not on the word "MEASURED": the label's textContent is
    # mixed-case ("Measured · experiment engine") and CSS uppercases it visually,
    # so an uppercase text match can never fail and tests nothing. A blocked
    # mission must have produced no metric cells at all.
    check("no measured metric cell rendered for a blocked mission",
          'class="metric"' not in dom and 'class="metrics"' not in dom,
          "a metric cell exists on a mission that never trained")
    # A blocked mission still emits mission_end, so the rail must not claim the
    # mission "completed" or imply experiments are still coming.
    check("blocked mission is not labelled complete", "Mission complete" not in body)
    check("blocked mission does not imply pending experiments",
          "no experiments yet" not in body)
    # Nothing was measured, so the best-so-far panel must not be labelled
    # "measured". Positive assertion -- more robust than a windowed negative.
    check("blocked mission labels its empty best panel honestly",
          "nothing ran" in body, "best-so-far aside does not read 'nothing ran'")

    # ---------------------------------------------------------- comparison
    print("\nCOMPARISON VIEW")
    dom = dump_dom(chrome, f"{BASE}/?view=comparison&run=1", "4-comparison.png", 60000)
    body = text_of(dom)
    comparison = api("/api/compare", {"csv": "data/churn.csv", "target": "churned", "budget": 6})
    det = comparison["deterministic"]["summary"]
    gui = comparison["guided"]["summary"]
    check("comparison rendered", "Identical conditions" in body)
    check("fairness: same Dataset object shown", "same Dataset object" in body)
    check("deterministic best score shown", f"{det['best_score']:.4f}" in body,
          f"{det['best_score']:.4f}")
    check("llm-guided best score shown", f"{gui['best_score']:.4f}" in body,
          f"{gui['best_score']:.4f}")
    delta = gui["best_score"] - det["best_score"]
    check("delta shown with correct sign",
          f"{delta:+.4f}" in body or f"{abs(delta):.4f}" in body, f"{delta:+.4f}")
    winner = "Deterministic" if delta < -1e-4 else ("LLM-guided" if delta > 1e-4 else "Tie")
    check(f"verdict reports the real winner ({winner})", winner in body)

    # CSS ellipsis is invisible to the DOM -- textContent still holds the full
    # string -- so a text assertion cannot detect a truncated model name. Check
    # the rule instead: the model cell must not be set to ellipsise.
    css = (ROOT / "ui" / "style.css").read_text(encoding="utf-8")
    model_rules = re.findall(r"\.c-model\s*\{([^}]*)\}", css)
    check("model cells are not set to ellipsise (would hide a long name)",
          all("text-overflow: ellipsis" not in rule for rule in model_rules),
          "a .c-model rule sets text-overflow: ellipsis")
    check("model column sizes to its content",
          css.count("max-content") >= 2, "fixed ch width truncates random_forest")

    print(f"\nscreenshots -> {SHOTS}")
    return check_clicks(chrome)


# The transport controls can only be proven by clicking them. Rather than add a
# browser-automation dependency, we write a throwaway page that is index.html
# plus an injected script which performs real .click() calls and records what
# happened into the DOM, where --dump-dom can read it. The file is deleted in a
# finally block, so nothing permanent is added to ui/.
CLICK_HARNESS = """
<script type="module">
const log = [];
const rec = (name, ok, detail = "") => log.push({ name, ok, detail });
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const buffered = () => {
  const m = /(\\d+) event/.exec($("transport-status").textContent);
  return m ? Number(m[1]) : 0;
};
async function waitFor(fn, timeout = 20000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (fn()) return true;
    await sleep(40);
  }
  return false;
}
function finish() {
  const out = document.createElement("pre");
  out.id = "click-results";
  out.textContent = JSON.stringify(log);
  document.body.append(out);
}

(async () => {
  try {
    const arrived = await waitFor(() => buffered() > 4);
    rec("events buffered before any click", arrived, String(buffered()));

    // --- PAUSE ---------------------------------------------------------
    $("play").click();
    rec("PAUSE click flips aria-pressed to false",
        $("play").getAttribute("aria-pressed") === "false");
    rec("PAUSE click relabels the button to Play",
        $("play").textContent.trim() === "Play");

    const held = buffered();
    await sleep(2500);
    rec("PAUSE actually halts the reveal", buffered() === held,
        `was ${held}, now ${buffered()}`);

    // --- STEP ----------------------------------------------------------
    const beforeStep = buffered();
    $("step").click();
    const afterOne = buffered();
    rec("STEP reveals exactly one event", beforeStep - afterOne === 1,
        `${beforeStep} -> ${afterOne}`);

    $("step").click();
    $("step").click();
    rec("three STEP clicks reveal exactly three events",
        beforeStep - buffered() === 3, `${beforeStep} -> ${buffered()}`);
    rec("STEP leaves playback paused",
        $("play").getAttribute("aria-pressed") === "false");

    // --- PLAY resumes --------------------------------------------------
    const beforePlay = buffered();
    $("play").click();
    rec("PLAY click flips aria-pressed back to true",
        $("play").getAttribute("aria-pressed") === "true");
    await sleep(2500);
    rec("PLAY resumes the reveal", buffered() < beforePlay,
        `${beforePlay} -> ${buffered()}`);

    // --- SKIP TO END ---------------------------------------------------
    $("play").click(); // pause again so SKIP is unambiguous
    rec("events still pending before SKIP", buffered() > 0, String(buffered()));
    $("skip").click();
    rec("SKIP TO END drains the whole queue", buffered() === 0, String(buffered()));
    rec("SKIP reaches the mission end",
        document.querySelector("#evidence").textContent.includes("Mission complete"));

    // --- collapsed row expands ----------------------------------------
    const row = document.querySelector("#loop .collapsed");
    if (row) {
      const id = row.dataset.experimentId;
      row.click();
      await sleep(100);
      const expanded = document.querySelector(
        `#loop .experiment[data-experiment-id="${id}"]`);
      rec("clicking a collapsed row expands that experiment", Boolean(expanded), `exp ${id}`);
    } else {
      rec("clicking a collapsed row expands that experiment", false, "no collapsed row");
    }

    // --- citation chip highlights its finding ---------------------------
    const chip = document.querySelector("#loop .cite[data-cite]");
    if (chip) {
      const wanted = chip.dataset.cite;
      chip.click();
      await sleep(100);
      const lit = document.querySelector("#loop .findings li.highlight");
      rec("clicking a citation highlights its finding",
          Boolean(lit) && lit.dataset.finding === wanted,
          lit ? lit.dataset.finding : "nothing highlighted");
      chip.click();
      await sleep(100);
      rec("clicking the citation again clears the highlight",
          !document.querySelector("#loop .findings li.highlight"));
    } else {
      rec("clicking a citation highlights its finding", false, "no finding-citation chip");
    }

    // --- nav ------------------------------------------------------------
    document.querySelector('.nav button[data-view="comparison"]').click();
    rec("nav switches to the comparison view",
        !document.getElementById("view-comparison").hidden
        && document.getElementById("view-dashboard").hidden);
    document.querySelector('.nav button[data-view="dashboard"]').click();
    rec("nav switches back to the dashboard",
        !document.getElementById("view-dashboard").hidden);
  } catch (error) {
    rec("harness ran without throwing", false, String(error && error.message));
  }
  finish();
})();
</script>
"""


def check_clicks(chrome: str) -> int:
    """Drive the transport controls with real clicks on a throwaway page."""
    print("\nTRANSPORT CONTROLS (real clicks)")
    mission = api("/api/missions", {"csv": "data/churn.csv", "target": "churned",
                                    "budget": 6, "llm": "offline"})["mission_id"]
    for _ in range(80):
        if api(f"/api/missions/{mission}/summary")["done"]:
            break
        time.sleep(0.25)

    harness = ROOT / "ui" / "_clicktest.html"
    try:
        source = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
        harness.write_text(source.replace("</body>", CLICK_HARNESS + "</body>"),
                           encoding="utf-8")
        dom = dump_dom(chrome, f"{BASE}/ui/_clicktest.html?mission={mission}",
                       "5-clicks.png", 60000)
    finally:
        harness.unlink(missing_ok=True)

    match = re.search(r'<pre id="click-results">(.*?)</pre>', dom, re.S)
    if not match:
        check("click harness produced results", False, "no #click-results in DOM")
        return finish()

    import html as html_mod

    for row in json.loads(html_mod.unescape(match.group(1))):
        check(row["name"], row["ok"], row.get("detail", ""))
    return finish()


def finish() -> int:
    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("All browser checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
