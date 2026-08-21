#!/usr/bin/env python3
"""Regenerate the README screenshots against a running instance.

    uvicorn api:app --app-dir src --port 8000
    python scripts/make_screenshots.py

Playwright is needed for this and for nothing else, so it stays out of
requirements.txt: pip install playwright && python -m playwright install chromium

The questions are in German because the app is, and because a screenshot of an
English query against a German statute would be showing something the project
does not do.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
BASE = "http://localhost:8000"

# Each shot names the question and how much of the page to keep. Cropping to a
# height rather than saving the full page keeps the images readable inline in a
# README, where a 3,000 pixel column renders as an unreadable strip.
SHOTS = [
    ("answer", "Wie hoch darf die Mietkaution sein und wie darf ich sie zahlen?", 1180),
    ("sources", "Wie hoch darf die Mietkaution sein und wie darf ich sie zahlen?", None),
    ("layperson", "Bei mir ist seit zwei Wochen die Heizung kaputt. Muss ich die volle Miete zahlen?", 1180),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 1000}, device_scale_factor=2)

        for name, question, height in SHOTS:
            page.goto(BASE, wait_until="networkidle")
            page.fill("#q", question)
            page.click("button[type=submit]")

            # Generation runs on the local model, so this is tens of seconds, not
            # the fraction of a second a default timeout allows for.
            page.wait_for_selector("#out .answer", state="visible", timeout=180_000)
            page.wait_for_timeout(600)

            if name == "sources":
                sources = page.locator("#out h2 + .card")
                sources.scroll_into_view_if_needed()
                page.wait_for_timeout(300)
                sources.screenshot(path=str(OUT / f"{name}.png"))
            else:
                box = {"x": 0, "y": 0, "width": 1100, "height": height}
                page.screenshot(path=str(OUT / f"{name}.png"), clip=box)

            print(f"{name}.png")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
