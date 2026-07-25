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

from verl.experimental.scalewob.playwright_backend import PlaywrightScaleWoBAutomation, _resolve_env_url


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
