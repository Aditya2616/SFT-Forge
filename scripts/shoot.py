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
            page.evaluate("document.querySelector('#stage').scrollIntoView({block: 'center'})")
            page.wait_for_timeout(4200)
            page.screenshot(path=str(out / f"{name}-streaming.png"))
            report[f"{name}_hscroll"] = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            page.click("#skip")
            page.wait_for_timeout(600)
            page.screenshot(path=str(out / f"{name}-done.png"), full_page=True)
            # open the inspector on the first rejected row in the log
            # open a clear, correct rejection: a refusal caught at the quality gate
            page.evaluate("document.querySelector('#f-reason').value='refusal_boilerplate'; document.querySelector('#f-reason').dispatchEvent(new Event('change'))")
            page.wait_for_timeout(200)
            page.locator("#logsp .row").first.scroll_into_view_if_needed()
            if touch:
                page.locator("#logsp .row").first.tap()
            else:
                page.locator("#logsp .row").first.click()
            page.wait_for_timeout(700)
            page.screenshot(path=str(out / f"{name}-inspector.png"))
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
        for n in ("desktop-streaming", "phone-streaming", "desktop-inspector", "phone-inspector"):
            shutil.copy(out / f"{n}.png", dst / f"{n}.png")


if __name__ == "__main__":
    main()
