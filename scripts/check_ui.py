"""Exercise the benchmark run view and live mode (paste, run, inspect) at desktop and phone size.

Usage: python scripts/check_ui.py [--url http://127.0.0.1:8104/] [--live]
--live also runs the paste flow (needs the FastAPI server, not a static file server).
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
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out", default=str(ROOT / ".shots"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    report: dict = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, vp, touch in (("desktop", {"width": 1440, "height": 900}, False), ("phone", {"width": 390, "height": 844}, True)):
            ctx = browser.new_context(viewport=vp, has_touch=touch, is_mobile=touch)
            page = ctx.new_page()
            page.on("console", lambda m, n=name: errors.append(f"{n} {m.type}: {m.text}") if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e, n=name: errors.append(f"{n} pageerror: {e}"))
            page.on("requestfailed", lambda r, n=name: errors.append(f"{n} requestfailed: {r.url}"))
            page.goto(args.url, wait_until="load")
            page.wait_for_function("window.__forge && window.__forge.sim.n > 0")
            (page.locator("#runs button[data-id=bench]").tap if touch else page.locator("#runs button[data-id=bench]").click)()
            page.wait_for_timeout(1500)
            page.click("#skip")
            page.wait_for_timeout(600)
            report[f"{name}_bench_badge"] = page.inner_text("#badge")
            report[f"{name}_bench_table_rows"] = page.locator("#bench-table tr").count()
            report[f"{name}_hscroll"] = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            page.locator("#bench").scroll_into_view_if_needed()
            page.screenshot(path=str(out / f"{name}-bench.png"))
            if args.live:
                page.fill("#paste", "\n".join([
                    '{"instruction": "Why is the sky blue?", "output": "Air scatters short blue wavelengths of sunlight more than red ones."}',
                    '{"instruction": "Why is the sky blue?", "output": "Air scatters short blue wavelengths of sunlight more than red ones."}',
                    '{"instruction": "Contact?", "output": "Mail me at jane.doe@example.com or use key AKIAIOSFODNN7EXAMPLE"}',
                    "not json",
                ]))
                page.locator("#go").scroll_into_view_if_needed()
                (page.locator("#go").tap if touch else page.locator("#go").click)()
                page.wait_for_function("document.querySelector('#badge').textContent.toLowerCase().includes('live')", timeout=120000)
                page.wait_for_timeout(1500)
                page.click("#skip")
                page.wait_for_timeout(800)
                report[f"{name}_live_badge"] = page.inner_text("#badge")
                report[f"{name}_live_stats"] = page.inner_text("#stats")[:200]
                page.screenshot(path=str(out / f"{name}-live.png"))
            ctx.close()
        browser.close()
    report["console_errors"] = errors
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
