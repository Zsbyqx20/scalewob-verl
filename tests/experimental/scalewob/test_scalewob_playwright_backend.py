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

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from verl.experimental.scalewob.playwright_backend import PlaywrightScaleWoBAutomation, _resolve_env_url

_CHROME_EXECUTABLE = os.environ.get("CHROME_EXECUTABLE", "/workspace/chrome/chrome-linux64/chrome")
_HAS_CHROME = os.path.isfile(_CHROME_EXECUTABLE)


def test_resolve_env_url_joins_bare_env_id_with_base_url():
    url = _resolve_env_url("BestBuy", "http://127.0.0.1:8000/scalewob-env")
    assert url == "http://127.0.0.1:8000/scalewob-env/BestBuy/index.html"


def test_resolve_env_url_strips_trailing_slash_on_base_url():
    url = _resolve_env_url("BestBuy", "http://127.0.0.1:8000/scalewob-env/")
    assert url == "http://127.0.0.1:8000/scalewob-env/BestBuy/index.html"


def test_resolve_env_url_passes_through_full_urls_unchanged():
    url = _resolve_env_url("http://example.com/custom-env", "http://127.0.0.1:8000/scalewob-env")
    assert url == "http://example.com/custom-env"


def test_resolve_env_url_falls_back_to_blank_without_base_url():
    assert _resolve_env_url("BestBuy", None) == "about:blank"
    assert _resolve_env_url(None, "http://127.0.0.1:8000/scalewob-env") == "about:blank"


class _FakePage:
    def __init__(self, evaluate_results):
        self._evaluate_results = list(evaluate_results)
        self.evaluate_calls = []

    def evaluate(self, script, arg=None):
        self.evaluate_calls.append((script, arg))
        return self._evaluate_results.pop(0)

    def wait_for_selector(self, selector, timeout=None):
        raise TimeoutError("no such selector")

    def inner_text(self, selector):
        return ""


def _automation_with_fake_page(evaluate_results):
    automation = PlaywrightScaleWoBAutomation()
    automation._page = _FakePage(evaluate_results)
    return automation


def test_finish_evaluation_uses_window_evaluate_task_when_present():
    automation = _automation_with_fake_page(
        [True, {"success": True, "score": 100, "message": "Address added and set as default."}]
    )

    result = automation._finish_evaluation(task_id="1", params={"street": "999 Innovation Dr"})

    assert result == {
        "success": True,
        "reward": 1.0,
        "task_id": "1",
        "params": {"street": "999 Innovation Dr"},
        "message": "Address added and set as default.",
        "score": 100,
    }
    eval_script, eval_arg = automation._page.evaluate_calls[1]
    assert eval_arg == {"taskId": 1, "street": "999 Innovation Dr"}


def test_finish_evaluation_reports_failure_from_window_evaluate_task():
    automation = _automation_with_fake_page([True, {"success": False, "message": "Address not found."}])

    result = automation._finish_evaluation(task_id="1", params={"street": "999 Innovation Dr"})

    assert result["success"] is False
    assert result["reward"] == 0.0
    assert result["message"] == "Address not found."


def test_finish_evaluation_falls_back_to_params_when_no_evaluate_task():
    automation = _automation_with_fake_page([False])

    result = automation._finish_evaluation(task_id="1", params=None)

    assert result == {"success": True, "reward": 1.0, "task_id": "1", "params": None}


@pytest.mark.skipif(not _HAS_CHROME, reason=f"chrome executable not found at {_CHROME_EXECUTABLE}")
def test_close_recovers_when_worker_thread_is_wedged_in_evaluate():
    """page.evaluate() (used by finish_evaluation/_type) ignores set_default_timeout, so a
    stuck env page can wedge the single worker thread forever. close() must still return by
    force-killing the underlying Chrome process instead of waiting on that thread.
    """
    automation = PlaywrightScaleWoBAutomation(chrome_executable=_CHROME_EXECUTABLE, headless=True)
    automation.start()
    automation.start_evaluation()

    # Queue a call that hangs forever on the same single-worker executor `close()` would
    # otherwise queue behind, reproducing the reported production hang.
    wedge_future = automation._executor.submit(automation._page.evaluate, "new Promise(() => {})")
    time.sleep(0.5)
    assert not wedge_future.done()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as ex:
        close_future = ex.submit(automation.close)
        close_future.result(timeout=10)
    elapsed = time.time() - t0

    assert elapsed < 10
    assert automation._closed is True


@pytest.mark.skipif(not _HAS_CHROME, reason=f"chrome executable not found at {_CHROME_EXECUTABLE}")
def test_finish_evaluation_hang_is_bounded_by_browser_operation_timeout():
    """Reproduces the reported production hang end-to-end through ScaleWoBBrowser: a page
    whose window.evaluateTask never resolves must not wedge the caller past the configured
    browser_operation_timeout_seconds, and the browser must still be closeable afterwards.
    """
    from verl.experimental.scalewob.browser import ScaleWoBBrowser, ScaleWoBBrowserConfig

    browser = ScaleWoBBrowser(
        ScaleWoBBrowserConfig(
            backend="playwright",
            chrome_executable=_CHROME_EXECUTABLE,
            browser_operation_timeout_seconds=2.0,
        )
    )
    browser._env_id = "about:blank"
    automation = browser._ensure_automation()
    automation.start()
    automation.start_evaluation()
    automation._run(
        automation._page.set_content,
        "<html><body><script>window.evaluateTask = () => new Promise(() => {});</script></body></html>",
    )

    t0 = time.time()
    result = browser.step({"action": "finish", "status": "finished"})
    elapsed = time.time() - t0

    assert elapsed < 5
    assert result["info"].get("invalid_action") is True
    assert "timed out" in result["info"]["error"]

    with ThreadPoolExecutor(max_workers=1) as ex:
        close_future = ex.submit(browser.close)
        close_future.result(timeout=10)
