# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Playwright-backed ScaleWoB automation that runs a local Chrome browser.

This module provides a drop-in replacement for ``scalewob.automation.ScaleWoBAutomation``
that controls a real Chromium browser via Playwright. It is useful for local development
and for environments without access to the hosted ScaleWoB service.

The Playwright Sync API cannot run inside an asyncio event loop (verl's agent loop is
async). To avoid that, all Playwright operations are dispatched to a single dedicated
thread that owns the browser context.

Usage:
    import scalewob.automation
    from verl.experimental.scalewob.playwright_backend import PlaywrightScaleWoBAutomation
    scalewob.automation.ScaleWoBAutomation = PlaywrightScaleWoBAutomation

Or set ``browser_config.backend = "playwright"`` (after the accompanying ``browser.py``
change) and pass ``browser_config.chrome_executable`` / ``browser_config.window_size``.
"""

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image
from playwright.sync_api import sync_playwright


def _is_url(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith(("http://", "https://", "file://", "about:"))


class PlaywrightScaleWoBAutomation:
    """Playwright-based browser automation compatible with ``ScaleWoBBrowser``.

    The ``env_id`` passed to the constructor is interpreted as a URL to open.
    If it is not a URL, the browser navigates to ``about:blank`` and optionally
    renders the task description as a simple page.

    All public methods are synchronous, but they dispatch to a single internal worker
    thread so that the Playwright Sync API never runs inside the caller's asyncio loop.
    """

    def __init__(
        self,
        env_id: str | None = None,
        base_url: str | None = None,
        platform: str = "mobile",
        headless: bool = True,
        screenshot_quality: str = "low",
        chrome_executable: str | None = None,
        window_size: tuple[int, int] = (390, 844),
        mobile_user_agent: str | None = None,
    ):
        self._env_id = env_id or "about:blank"
        self._base_url = base_url
        self._platform = platform
        self._headless = headless
        self._screenshot_quality = screenshot_quality
        self._chrome_executable = chrome_executable
        self._window_size = window_size
        self._mobile_user_agent = mobile_user_agent or (
            "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36"
        )

        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._closed = False

        # Internal state owned by the worker thread.
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None
        self._last_screenshot: Image.Image | None = None
        self._task_id: Any = 0
        self._params: dict[str, Any] | None = None
        self.tasks: list[dict[str, Any]] = []

    def _run(self, fn, *args, **kwargs):
        """Dispatch ``fn`` to the worker thread and return its result."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Playwright automation has been closed")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scalewob-playwright")
        return self._executor.submit(fn, *args, **kwargs).result()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._run(self._start)

    def _start(self) -> None:
        if self._playwright is not None:
            return
        self._playwright = sync_playwright().start()
        launch_args = ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        if self._screenshot_quality == "low":
            launch_args.append("--force-device-scale-factor=1")
        launch_kwargs = {
            "headless": self._headless,
            "args": launch_args,
        }
        if self._chrome_executable:
            launch_kwargs["executable_path"] = self._chrome_executable
        self._browser = self._playwright.chromium.launch(**launch_kwargs)
        self._context = self._browser.new_context(
            viewport={"width": self._window_size[0], "height": self._window_size[1]},
            user_agent=self._mobile_user_agent,
            device_scale_factor=1,
            has_touch=True,
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(5000)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.submit(self._close).result()
            executor.shutdown(wait=True)

    def _close(self) -> None:
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Task / evaluation
    # ------------------------------------------------------------------
    def start_evaluation(self) -> None:
        """Open the environment URL and record the initial screenshot."""
        self._run(self._start_evaluation)

    def _start_evaluation(self) -> None:
        if self._page is None:
            self._start()
        url = self._env_id if _is_url(self._env_id) else "about:blank"
        if url == "about:blank":
            # Render a minimal placeholder page so the screenshot is not empty.
            self._page.set_content(
                "<html><body style='margin:0;padding:20px;font-family:sans-serif;'>"
                f"<h1>Task environment</h1><p>env_id: {self._env_id}</p>"
                "</body></html>"
            )
        else:
            self._page.goto(url, wait_until="domcontentloaded")
        # Give a short moment for the page to settle.
        time.sleep(0.1)
        self._last_screenshot = self._take_screenshot_pil()

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------
    def take_screenshot(self, format: str = "pil") -> Image.Image | bytes:
        """Return the current page screenshot as a PIL image or raw bytes."""
        return self._run(self._take_screenshot, format)

    def _take_screenshot(self, format: str) -> Image.Image | bytes:
        if self._page is None:
            self._start()
        raw = self._page.screenshot(type="png")
        if format == "pil":
            self._last_screenshot = Image.open(io.BytesIO(raw)).convert("RGB")
            return self._last_screenshot
        return raw

    def _take_screenshot_pil(self) -> Image.Image:
        raw = self._page.screenshot(type="png")
        return Image.open(io.BytesIO(raw)).convert("RGB")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def click(self, x: int, y: int) -> None:
        self._run(self._click, x, y)

    def _click(self, x: int, y: int) -> None:
        self._page.mouse.click(x, y)
        self._wait_for_stable()

    def long_press(self, x: int, y: int) -> None:
        self._run(self._long_press, x, y)

    def _long_press(self, x: int, y: int) -> None:
        self._page.mouse.move(x, y)
        self._page.mouse.down()
        time.sleep(0.5)
        self._page.mouse.up()
        self._wait_for_stable()

    def type(self, text: str) -> None:
        self._run(self._type, text)

    def _type(self, text: str) -> None:
        # Try to type into the active element. If none, click the body first.
        if self._page.evaluate("document.activeElement === document.body"):
            self._page.click("body")
        self._page.keyboard.type(text)
        self._wait_for_stable()

    def press_enter(self) -> None:
        self._run(self._press_enter)

    def _press_enter(self) -> None:
        self._page.keyboard.press("Enter")
        self._wait_for_stable()

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._run(self._drag, x1, y1, x2, y2)

    def _drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._page.mouse.move(x1, y1)
        self._page.mouse.down()
        self._page.mouse.move(x2, y2, steps=10)
        self._page.mouse.up()
        self._wait_for_stable()

    def _wait_for_stable(self) -> None:
        """Small pause to let the page settle after an action."""
        time.sleep(0.1)

    # ------------------------------------------------------------------
    # Finish / reward
    # ------------------------------------------------------------------
    def finish_evaluation(self, task_id: Any = 0, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Evaluate a task finish and return a reward.

        The reward logic is intentionally simple for local smoke runs:

        - If ``params`` contains ``success_selector`` or ``success_text``, the page is
          checked for that selector/text. Success yields reward 1.0, otherwise 0.0.
        - If neither is provided, the agent is assumed to have succeeded and the reward
          is 1.0.

        Users should override this for real task evaluation.
        """
        return self._run(self._finish_evaluation, task_id, params)

    def _finish_evaluation(self, task_id: Any, params: dict[str, Any] | None) -> dict[str, Any]:
        self._task_id = task_id
        self._params = params or {}
        reward = 1.0
        success = True
        success_selector = self._params.get("success_selector")
        success_text = self._params.get("success_text")

        if success_selector is not None:
            try:
                self._page.wait_for_selector(success_selector, timeout=2000)
                success = True
            except Exception:
                success = False
                reward = 0.0

        if success and success_text is not None:
            page_text = self._page.inner_text("body").lower()
            if success_text.lower() not in page_text:
                success = False
                reward = 0.0

        return {
            "success": success,
            "reward": reward,
            "task_id": task_id,
            "params": params,
        }

    # ------------------------------------------------------------------
    # Task metadata helpers
    # ------------------------------------------------------------------
    def set_tasks(self, tasks: list[dict[str, Any]]) -> None:
        """Provide task metadata. This is normally supplied by the ScaleWoB server."""
        self.tasks = list(tasks)

    def add_task(self, task_id: Any, description: str, params: dict[str, Any] | None = None) -> None:
        """Convenience helper to add a single task."""
        self.tasks.append(
            {
                "task_id": task_id,
                "description": description,
                "params": params or {},
            }
        )
