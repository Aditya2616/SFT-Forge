"""Load the UI in headless Chromium at desktop and phone sizes, capture screenshots and console errors.

Usage: python scripts/shoot.py [--url http://127.0.0.1:8104/] [--out docs/assets] [--final]
The page is served separately (sftforge serve, or python -m http.server --directory docs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8104/")
    ap.add_argument("--out", default=str(ROOT / ".shots"))
    ap.add_argument("--final", action="store_true", help="write the README screenshots")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    report: dict = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, vp, touch in (("desktop", {"width": 1440, "height": 900}, False), ("phone", {"width": 390, "height": 844}, True)):
            ctx = browser.new_context(viewport=vp, has_touch=touch, is_mobile=touch, device_scale_factor=2 if touch else 1)
            page = ctx.new_page()
            page.on("console", lambda m, n=name: errors.append(f"{n} console {m.type}: {m.text}") if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e, n=name: errors.append(f"{n} pageerror: {e}"))
            page.goto(args.url.rstrip("/") + "/?run=bench", wait_until="load")   # the benchmark run has visible rejects
            page.wait_for_function("window.__forge && window.__forge.sim.n > 0")
            report[f"{name}_hscroll"] = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            page.click("#skip")                                   # the END state of the replay, never a mid-run frame
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(1500)
            # cut at card boundaries below the sticky bar: the run chooser through the stat tiles on desktop,
            # the whole stage card on a phone
            top_sel, bottom_sel = ("#runs", "#stats") if not touch else ("#stage", "#stage")
            page.evaluate(f"window.scrollTo(0, document.querySelector('{top_sel}').getBoundingClientRect().top + scrollY - {70 if not touch else 66})")
            page.wait_for_timeout(500)
            r1 = page.evaluate(f"document.querySelector('{bottom_sel}').getBoundingClientRect().bottom")
            clip_top = 64
            page.screenshot(path=str(out / f"{name}-streaming.png"),
                            clip={"x": 0, "y": clip_top, "width": vp["width"], "height": min(vp["height"], r1 + 10) - clip_top})
            # open the inspector on a clear, correct rejection: a refusal caught at the quality gate
            page.evaluate("document.querySelector('#f-reason').value='refusal_boilerplate'; document.querySelector('#f-reason').dispatchEvent(new Event('change'))")
            page.wait_for_timeout(200)
            page.locator("#logsp .row").first.scroll_into_view_if_needed()
            if touch:
                page.locator("#logsp .row").first.tap()
            else:
                page.locator("#logsp .row").first.click()
            page.wait_for_timeout(900)
            page.evaluate("document.querySelector('#i-body').scrollTop = 0")
            page.wait_for_timeout(200)
            # cut at the bottom of the last check row that is fully visible, not through a row
            cut = page.evaluate("""() => { const lim = innerHeight - 4; let b = 0;
                document.querySelectorAll('#i-body .chk').forEach(e => { const r = e.getBoundingClientRect(); if (r.bottom <= lim && r.bottom > b) b = r.bottom; });
                return b || innerHeight; }""")
            page.screenshot(path=str(out / f"{name}-inspector.png"), clip={"x": 0, "y": 0, "width": vp["width"], "height": min(vp["height"], cut + 8)})
            if args.final:  # the downstream experiment view
                page.click("#i-close")
                page.wait_for_timeout(700)
                page.evaluate("window.scrollTo(0, document.querySelector('#ft-legend').getBoundingClientRect().top + scrollY - 76)")
                page.wait_for_timeout(600)
                # cut under the last card row that fits entirely in the viewport
                cut = page.evaluate("""() => { const lim = innerHeight - 4; let b = 0;
                    document.querySelectorAll('#ft-grid > *').forEach(e => { const r = e.getBoundingClientRect(); if (r.bottom <= lim && r.bottom > b) b = r.bottom; });
                    return b || innerHeight; }""")
                page.screenshot(path=str(out / f"{name}-finetune.png"), clip={"x": 0, "y": 0, "width": vp["width"], "height": min(vp["height"], cut + 12)})
            small = page.evaluate(
                "[...document.querySelectorAll('button, a, select, input, label.btn')].filter(e => e.offsetParent && "
                "(e.getBoundingClientRect().height < 43.5) && !e.classList.contains('sr')).map(e => e.id || e.textContent.trim().slice(0,20))")
            report[f"{name}_small_targets"] = small[:12]
            ctx.close()
        browser.close()
    report["console_errors"] = errors
    print(json.dumps(report, indent=2))
    if args.final:
        import shutil

        dst = ROOT / "docs" / "assets"
        dst.mkdir(parents=True, exist_ok=True)
        for n in ("desktop-streaming", "phone-streaming", "phone-inspector", "desktop-finetune"):
            shutil.copy(out / f"{n}.png", dst / f"{n}.png")


if __name__ == "__main__":
    main()
