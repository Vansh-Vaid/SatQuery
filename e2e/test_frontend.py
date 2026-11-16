"""Browser smoke test: the built UI loads and a user can run an analysis end to end.

Starts the real server, drives headless Chromium with Playwright and fails on any uncaught JavaScript
error. Run separately from the unit tests (CI runs it before building the image):

  pip install playwright && python -m playwright install chromium
  python -m pytest -q e2e
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    env = {**os.environ, "SATQUERY_PERSIST_RUNS": "0", "SATQUERY_RATE_LIMIT": "0"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "satquery.api.app:app", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/api/health", timeout=2)
                break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError("server exited:\n" + proc.stdout.read().decode(errors="replace"))
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start within 60 s")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture()
def page(server):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1440, "height": 900})
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        yield pg
        browser.close()


def test_landing_page_loads(page, server):
    page.goto(server + "/")
    assert page.title().startswith("SatQuery AI")
    page.get_by_role("link", name="SatQuery AI home").wait_for()
    assert page.errors == []


@pytest.mark.parametrize("demo, query", [
    ("single_s2", "Highlight the water body referred to in the query."),
])
def test_console_runs_an_analysis(page, server, demo, query):
    page.goto(f"{server}/console?demo={demo}&q={query.replace(' ', '+')}")
    deck = page.get_by_role("textbox", name="Query")
    deck.wait_for()
    # the deep link fills the query once the demo list has loaded (asynchronously)
    playwright.expect(deck).to_have_value(query, timeout=30_000)
    run = page.get_by_role("button", name="Analyse →")
    playwright.expect(run).to_be_enabled(timeout=30_000)  # demo files fetched and validated
    run.click()
    answer = page.locator(".verdict-answer")
    answer.wait_for(timeout=90_000)
    assert answer.inner_text().strip()
    playwright.expect(page.get_by_label("Agentic execution trace")).to_be_visible()
    assert page.errors == []
