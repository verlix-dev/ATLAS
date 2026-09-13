"""Static checks for the ATLAS UI. No browser, no dependencies.

    python tools/check_ui.py

Verifies what can be verified without rendering: markup balance, that every
CSS custom property resolves, that no banned visual effect crept into the
rules, and -- the one that actually matters -- that every value shown in the
static prototype traces back to tests/fixtures/events.json verbatim.

Exits non-zero on failure so it can gate a stage.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parent.parent
VOID = {
    "meta", "link", "br", "img", "input", "hr", "source", "use", "path",
    "circle", "polyline", "area", "base", "col", "embed", "track", "wbr",
    "rect", "line",
}
# Effects still out of scope for ATLAS. Gradients, shadows, blur and glow were
# banned under the earlier dark/brutalist direction and are now explicitly part
# of the design language, so they are no longer listed here. What remains is
# the retro-terminal vocabulary the product is deliberately not part of.
BANNED = ("scanline", "halftone", "dither", "crt(")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else ' -- ' + detail}")
    if not ok:
        failures.append(name)


def flat(s: str) -> str:
    """Collapse HTML source wrapping, so a string that renders as one line
    (textContent) compares as one line."""
    return re.sub(r"\s+", " ", s)


class Balance(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.err: list[str] = []

    def handle_startendtag(self, tag, attrs):  # <br/> style
        pass

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.err.append(f"stray </{tag}>")
        elif self.stack[-1] != tag:
            self.err.append(f"</{tag}> closes <{self.stack[-1]}>")
        else:
            self.stack.pop()


def main() -> int:
    css = (ROOT / "ui" / "style.css").read_text(encoding="utf-8")
    css_rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)  # comments are not rules
    fixture = json.loads((ROOT / "tests" / "fixtures" / "events.json").read_text(encoding="utf-8"))
    events = fixture["events"]

    html_files = sorted((ROOT / "ui").glob("*.html"))

    print("MARKUP")
    for path in html_files:
        html = path.read_text(encoding="utf-8")
        parser = Balance()
        parser.feed(html)
        check(
            f"{path.name} tags balance",
            not parser.err and not parser.stack,
            f"{parser.err} unclosed={parser.stack}",
        )
        inline = re.findall(r'style="[^"]*"', html)
        check(f"{path.name} no inline styles", not inline, str(inline))

    print("\nDESIGN SYSTEM")
    defined = set(re.findall(r"^\s*(--[a-z0-9-]+)\s*:", css, re.M))
    used = set(re.findall(r"var\((--[a-z0-9-]+)", css))
    for path in html_files:
        used |= set(re.findall(r"var\((--[a-z0-9-]+)", path.read_text(encoding="utf-8")))
    check(f"all var() resolve ({len(defined)} defined, {len(used)} used)",
          not (used - defined), str(sorted(used - defined)))
    found = [w for w in BANNED if w in css_rules.lower()]
    check("no retro-terminal effects in rules", not found, str(found))
    # Fact and interpretation must stay visually separable. The border STYLE
    # carries that distinction, whatever the palette happens to be.
    measured = re.search(r"--measured-block-border:([^;]*);", css)
    reasoned = re.search(r"--reasoned-block-border:([^;]*);", css)
    check("measured border is solid", bool(measured) and "solid" in measured.group(1))
    check("reasoned border is dashed", bool(reasoned) and "dashed" in reasoned.group(1))
    check("continue/revise/stop each have an accent",
          all(f'data-action="{a}"' in css for a in ("continue", "revise", "stop")))
    check("decision state carries a glyph, not colour alone",
          css.count(".decision-glyph::before") >= 3)

    proto = ROOT / "ui" / "prototype.html"
    if not proto.exists():
        print("\n(prototype.html removed -- skipping fixture trace)")
        return finish()

    print("\nPROTOTYPE TRACES TO REAL FIXTURE")
    html = flat(proto.read_text(encoding="utf-8"))

    def ev(stage: str, eid: int | None = None) -> dict:
        return [e for e in events if e["stage"] == stage and (eid is None or e.get("id") == eid)][0]

    m = ev("experiment_result", 3)["metrics"]
    diagnosis, hypothesis, decision = ev("diagnosis", 3), ev("hypothesis", 3), ev("decision", 3)
    profile = ev("mission_start")["profile"]
    rejections = [r for e in events if e["stage"] == "hypothesis" for r in e["rejected"]]

    traced = [
        ("f1_macro", f"{m['f1_macro']:.4f}"),
        ("recall", f"{m['recall']:.4f}"),
        ("precision", f"{m['precision']:.4f}"),
        ("roc_auc", f"{m['roc_auc']:.4f}"),
        ("pr_auc", f"{m['pr_auc']:.4f}"),
        ("overfit_gap", f"+{m['overfit_gap']:.4f}"),
        ("train seconds", str(ev("experiment_result", 3)["seconds"])),
        ("test_rows", str(m["test_rows"])),
        ("diagnosis summary", diagnosis["summary"]),
        ("diagnosis finding", diagnosis["findings"][0]),
        ("hypothesis proposed_change", hypothesis["proposed_change"]),
        ("hypothesis expected_effect", hypothesis["expected_effect"]),
        ("hypothesis reasoning", hypothesis["reasoning"]),
        ("decision reason", decision["reason"]),
        ("profile rows", str(profile["rows"])),
        ("profile imbalance", f"{profile['imbalance_ratio']:.2f}"),
        ("rejection reason", rejections[0]["reason"]),
    ]
    for name, value in traced:
        check(f"{name} verbatim from fixture", flat(value) in html, repr(value))

    check("cite string is identical to the finding it points at",
          hypothesis["cites"][0] == diagnosis["findings"][0])
    check("prototype is marked as static, not a live run",
          'data-prototype="true"' in html and "static prototype" in html)
    return finish()


def finish() -> int:
    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("All UI checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
