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

import concurrent.futures
import io
import logging
import os
import queue
import threading
import time
from typing import Any

import psutil
from PIL import Image
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# page.evaluate() (used by finish_evaluation's window.evaluateTask call and by
# _type's activeElement check) does not honor page.set_default_timeout() -- a
# hung env page can block it forever. Bound how long close() waits for the
# worker thread before giving up and force-killing the browser process instead.
_CLOSE_TIMEOUT_SECONDS = 5.0


class _DaemonSingleThreadExecutor:
    """Runs submitted calls on a single dedicated daemon thread.

    This exists instead of ``concurrent.futures.ThreadPoolExecutor`` because that class's
    worker threads are never daemon threads: if a submitted call hangs forever (e.g. a
    Playwright ``page.evaluate()`` stuck on an unresponsive page), the interpreter's
    ``concurrent.futures.thread._python_exit`` atexit hook will try to ``join()`` that thread
    and block process shutdown forever, even after we've given up on the call and moved on.
    A daemon thread lets a permanently wedged call be abandoned safely: the thread leaks until
    the process exits, but it never blocks that exit.
    """

    def __init__(self, thread_name: str):
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name=thread_name, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            fn, args, kwargs, future = self._queue.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - propagate to the caller via the future
                future.set_exception(exc)
            else:
                future.set_result(result)

    def submit(self, fn, *args, **kwargs) -> "concurrent.futures.Future":
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((fn, args, kwargs, future))
        return future


def _is_url(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith(("http://", "https://", "file://", "about:"))


def _resolve_env_url(env_id: str | None, base_url: str | None) -> str:
    """Resolve an ``env_id`` (a bare env name or an already-complete URL) to a URL.

    Training data passes bare env names (e.g. ``"BestBuy"``, ``"12306"``); only ad-hoc/manual
    usage passes a full URL directly. Bare names are joined with ``base_url`` to reach the
    env's ``index.html``, matching the URL contract served by ``scripts/serve_scalewob.py``.
    """
    if _is_url(env_id):
        return env_id
    if not env_id or not base_url:
        return "about:blank"
    return f"{base_url.rstrip('/')}/{env_id}/index.html"


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
        default_timeout_seconds: float = 10.0,
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
        self._default_timeout_seconds = default_timeout_seconds

        self._executor: _DaemonSingleThreadExecutor | None = None
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

        # PID of the Playwright driver process (node subprocess that owns the browser).
        # Tracked so a wedged worker thread (e.g. stuck inside page.evaluate(), which does
        # not honor set_default_timeout) can still be recovered by killing the underlying
        # OS process tree from outside that thread -- Python cannot forcibly stop a running
        # thread, so this is the only reliable way to unblock it.
        self._driver_pid: int | None = None

    def _run(self, fn, *args, **kwargs):
        """Dispatch ``fn`` to the worker thread and return its result."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Playwright automation has been closed")
            if self._executor is None:
                self._executor = _DaemonSingleThreadExecutor(thread_name="scalewob-playwright")
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
        # The driver subprocess (playwright's node process) parents the actual
        # chromium process tree; capture its PID now while we know it's alive so
        # close()/kill() can find and terminate that tree later even if the
        # worker thread that owns self._browser is itself wedged. This reaches into
        # a private attribute because the sync API exposes no public accessor for
        # it; fail soft (kill-on-timeout just becomes a no-op) if it ever changes.
        try:
            driver_proc = self._playwright._impl_obj._connection._transport._proc
            self._driver_pid = driver_proc.pid
        except AttributeError:
            logger.warning("could not determine Playwright driver PID; force-kill-on-timeout will be unavailable")
            self._driver_pid = None
        self._context = self._browser.new_context(
            viewport={"width": self._window_size[0], "height": self._window_size[1]},
            user_agent=self._mobile_user_agent,
            device_scale_factor=1,
            has_touch=True,
        )
        self._page = self._context.new_page()
        # Playwright's per-call default timeout (covers most actions/navigations, notably
        # NOT page.evaluate() -- see close()/_kill_process_tree() for that case). Kept in sync
        # with ScaleWoBBrowserConfig.browser_operation_timeout_seconds, the outer timeout that
        # wraps every call from browser.py, so a screenshot/click/etc. can't spuriously fire
        # Playwright's own timeout before the caller's configured one ever gets a chance.
        self._page.set_default_timeout(self._default_timeout_seconds * 1000)

    def close(self) -> None:
        """Close the browser, tearing down the OS process tree if the worker thread is wedged.

        ``page.evaluate()`` (used by ``finish_evaluation`` and ``_type``) does not honor
        ``page.set_default_timeout()``, so a hung env page can block the single worker thread
        forever. In that case a plain ``close()`` call queued behind it would also hang forever
        (Python cannot forcibly stop a running thread), so we bound the wait and fall back to
        killing the browser's OS process tree directly, which unblocks the wedged call too.
        Once that happens the worker thread is abandoned for good (it may still be parked
        inside Playwright's now-dead dispatcher fiber) -- safe because it's a daemon thread,
        so it can never block process shutdown.

        This method must never raise: it runs from the agent loop's ``finally:`` block, and an
        exception escaping there fails every other in-flight rollout on the same worker via
        ``asyncio.gather()`` -- one trajectory's teardown problem should never take down the
        whole training job.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is None:
            return
        future = executor.submit(self._close)
        try:
            future.result(timeout=_CLOSE_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "PlaywrightScaleWoBAutomation.close() timed out after %.1fs; force-killing browser process tree",
                _CLOSE_TIMEOUT_SECONDS,
            )
            try:
                self._kill_process_tree()
            except Exception:
                logger.warning("PlaywrightScaleWoBAutomation._kill_process_tree() raised", exc_info=True)
        except Exception:
            logger.warning("PlaywrightScaleWoBAutomation._close() raised", exc_info=True)

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

    def _kill_process_tree(self) -> None:
        """Force-kill the Playwright driver process and any chromium descendants.

        Safe to call from any thread, including while the dedicated worker thread is stuck
        inside a hung Playwright call -- SIGKILL-ing the underlying process immediately fails
        that pending call rather than waiting for it to return.

        The driver process can exit on its own between any two of the calls below (e.g. the
        wedged call finally unblocks and the process tears itself down); every psutil call that
        touches it -- not just the initial ``Process()`` construction -- can therefore raise
        ``NoSuchProcess``, so each one is guarded individually rather than relying on a single
        surrounding try/except.
        """
        if self._driver_pid is None:
            return
        try:
            driver = psutil.Process(self._driver_pid)
            procs = [driver, *driver.children(recursive=True)]
        except psutil.NoSuchProcess:
            return
        for proc in procs:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            psutil.wait_procs(procs, timeout=_CLOSE_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("psutil.wait_procs() raised while force-killing browser process tree", exc_info=True)

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
        url = _resolve_env_url(self._env_id, self._base_url)
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
        """Evaluate a task finish by calling the environment's own ``window.evaluateTask``.

        Falls back to the ``success_selector``/``success_text`` params-based check (and, absent
        those, an unconditional success) only if the page does not expose ``evaluateTask`` at all
        (e.g. a manually-opened URL that isn't a ScaleWoB env bundle).
        """
        return self._run(self._finish_evaluation, task_id, params)

    def _finish_evaluation(self, task_id: Any, params: dict[str, Any] | None) -> dict[str, Any]:
        self._task_id = task_id
        self._params = params or {}

        has_evaluate_task = self._page.evaluate("typeof window.evaluateTask === 'function'")
        if has_evaluate_task:
            numeric_task_id: Any
            try:
                numeric_task_id = int(task_id)
            except (TypeError, ValueError):
                numeric_task_id = task_id
            eval_params = {"taskId": numeric_task_id, **self._params}
            result = self._page.evaluate("(params) => window.evaluateTask(params)", eval_params)
            success = bool(result.get("success"))
            return {
                "success": success,
                "reward": 1.0 if success else 0.0,
                "task_id": task_id,
                "params": params,
                "message": result.get("message"),
                "score": result.get("score"),
            }

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
