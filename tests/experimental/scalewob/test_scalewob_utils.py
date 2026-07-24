import time

from PIL import Image

from verl.experimental.scalewob.actions import parse_action
from verl.experimental.scalewob.browser import (
    ScaleWoBBrowser,
    ScaleWoBBrowserConfig,
    ScaleWoBBrowserOperationTimeoutError,
)
from verl.experimental.scalewob.prompt import build_prompt_messages
from verl.experimental.scalewob.vision import screenshot_hash


def test_action_parser_accepts_aliases_and_clamps_coordinates():
    parsed = parse_action('```json\n{"action": "click", "x": -3, "y": 1204}\n```')

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "tap", "x": 0, "y": 1000}


def test_sft_action_parser_accepts_inline_click():
    parsed = parse_action("Thought: Tap the OK button.\nAction: `device.click(100, 400)`")

    assert parsed.is_action_valid == 1
    assert parsed.thought == "Tap the OK button."
    assert parsed.raw_action == "device.click(100, 400)"
    assert parsed.action == {"action": "tap", "x": 100, "y": 400}


def test_sft_action_parser_accepts_fenced_swipe():
    parsed = parse_action("Thought: Scroll down.\nAction:\n```python\ndevice.swipe((100, 800), (100, 300))\n```")

    assert parsed.is_action_valid == 1
    assert parsed.raw_action == "device.swipe((100, 800), (100, 300))"
    assert parsed.action == {"action": "swipe", "x1": 100, "y1": 800, "x2": 100, "y2": 300}


def test_sft_action_parser_accepts_drag_alias():
    parsed = parse_action("Thought: Drag upward.\nAction: `device.drag((100, 800), (100, 300))`")

    assert parsed.is_action_valid == 1
    assert parsed.raw_action == "device.drag((100, 800), (100, 300))"
    assert parsed.action == {"action": "swipe", "x1": 100, "y1": 800, "x2": 100, "y2": 300}


def test_sft_action_parser_accepts_four_coordinate_motion():
    parsed = parse_action("Thought: Drag upward.\nAction: `device.drag(100, 800, 100, 300)`")

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "swipe", "x1": 100, "y1": 800, "x2": 100, "y2": 300}


def test_sft_action_parser_accepts_type():
    parsed = parse_action('Thought: Enter the value.\nAction: `device.type("hello")`')

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "input_text", "text": "hello"}


def test_sft_action_parser_accepts_long_click():
    parsed = parse_action("Thought: Long press the item.\nAction: `device.long_click(100, 400)`")

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "long_press", "x": 100, "y": 400}


def test_sft_action_parser_accepts_enter():
    parsed = parse_action("Thought: Submit the search.\nAction: `device.enter()`")

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "press_enter"}


def test_sft_action_parser_accepts_end_task():
    parsed = parse_action("Thought: The task is complete.\nAction: `device.end_task('finished')`")

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "finish", "status": "finished"}


def test_sft_action_parser_accepts_end_task_params():
    parsed = parse_action("Thought: The task is complete.\nAction: `device.end_task('finished', {'order_id': '123'})`")

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "finish", "status": "finished", "params": {"order_id": "123"}}


def test_sft_action_parser_ignores_non_object_end_task_params():
    parsed = parse_action(
        "Thought: Done.\nAction: `device.end_task('finished', \"Task completed successfully.\")`"
    )

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "finish", "status": "finished"}


def test_sft_action_parser_accepts_end_task_keywords():
    parsed = parse_action(
        "Thought: The task is complete.\nAction: `device.end_task(status='finished', params={'order_id': '123'})`"
    )

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "finish", "status": "finished", "params": {"order_id": "123"}}


def test_sft_action_parser_ignores_non_object_end_task_keyword_params():
    parsed = parse_action(
        "Thought: Done.\nAction: `device.end_task(status='finished', params='Task completed successfully.')`"
    )

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "finish", "status": "finished"}


def test_json_action_parser_accepts_end_task_alias():
    parsed = parse_action('{"action": "end_task", "status": "finished", "params": {"order_id": "123"}}')

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "finish", "status": "finished", "params": {"order_id": "123"}}


def test_bad_json_returns_fallback_wait():
    parsed = parse_action("{not-json")

    assert parsed.is_action_valid == 0
    assert parsed.action == {"action": "wait"}
    assert parsed.error is not None


def test_malformed_sft_response_returns_invalid_wait():
    parsed = parse_action("Thought: I should tap the button.\nAction: `device.click(`")

    assert parsed.is_action_valid == 0
    assert parsed.action == {"action": "wait"}
    assert parsed.error is not None


def test_unsupported_sft_device_method_returns_invalid_wait():
    parsed = parse_action("Thought: Go back.\nAction: `device.back()`")

    assert parsed.is_action_valid == 0
    assert parsed.action == {"action": "wait"}
    assert parsed.error == "unsupported_device_method: back"


def test_sft_action_parser_rejects_keyword_arguments():
    parsed = parse_action("Thought: Tap the target.\nAction: `device.click(x=100, y=400)`")

    assert parsed.is_action_valid == 0
    assert parsed.action == {"action": "wait"}
    assert parsed.error == "device_action_keywords_not_supported"


def test_sft_action_parser_rejects_non_literal_arguments():
    parsed = parse_action("Thought: Tap the target.\nAction: `device.click(1 + 2, 400)`")

    assert parsed.is_action_valid == 0
    assert parsed.action == {"action": "wait"}
    assert parsed.error == "action_argument_not_literal"


def test_sft_action_parser_rejects_multiple_statements():
    parsed = parse_action(
        "Thought: Tap the target.\nAction:\n```python\ndevice.click(100, 400)\ndevice.wait()\n```"
    )

    assert parsed.is_action_valid == 0
    assert parsed.action == {"action": "wait"}
    assert parsed.raw_action == "device.click(100, 400)\ndevice.wait()"
    assert parsed.error == "device_action_must_be_single_expression"


def test_prompt_builder_emits_exactly_one_image_placeholder():
    image = Image.new("RGB", (8, 8), "white")
    messages = build_prompt_messages(
        task_description="Press OK",
        screenshot=image,
        action_history=[{"thought": "Find OK.", "raw_action": "device.click(500, 500)"}],
        action_history_len=4,
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert all(message["role"] != "system" for message in messages)
    text_items = [item for item in messages[0]["content"] if item.get("type") == "text"]
    assert len(text_items) == 1
    prompt_text = text_items[0]["text"]
    for marker in (
        "# Core Objective",
        "## Device Control APIs",
        "## Task to Complete",
        "## Action History and Notes",
        "## Your Response",
    ):
        assert marker in prompt_text
    assert "Step 1 Thought: Find OK." in prompt_text
    assert "Step 1 Action: device.click(500, 500)" in prompt_text
    assert "[Current Screen]" not in prompt_text
    assert (
        "## Screenshot Status\nCurrent screenshot captured successfully.\n\n## Your Response"
        in prompt_text
    )

    image_items = [
        item
        for message in messages
        for item in (message["content"] if isinstance(message["content"], list) else [])
        if item.get("type") == "image"
    ]
    assert len(image_items) == 1
    assert image_items[0]["image"] is image
    assert messages[0]["content"].index(text_items[0]) < messages[0]["content"].index(image_items[0])


def test_prompt_builder_advertises_sft_scalewob_actions():
    image = Image.new("RGB", (8, 8), "white")
    messages = build_prompt_messages(
        task_description="Press OK",
        screenshot=image,
        action_history=[],
        action_history_len=4,
    )

    prompt_text = messages[0]["content"][0]["text"]
    for supported_method in (
        "device.click",
        "device.long_click",
        "device.type",
        "device.enter",
        "device.swipe",
        "device.wait",
        "device.end_task",
    ):
        assert supported_method in prompt_text
    assert (
        "- `device.swipe((x1, y1), (x2, y2))`: Swipe from the start point to the end point. "
        "Use this for scrolling and gesture movement on phone."
    ) in prompt_text
    assert "`device.end_task(status, params)`" in prompt_text
    assert "If the subtask requires to provide additional params on completion" in prompt_text


def test_prompt_builder_includes_task_params_schema():
    image = Image.new("RGB", (8, 8), "white")
    messages = build_prompt_messages(
        task_description="Submit the form",
        screenshot=image,
        action_history=[],
        action_history_len=4,
        task_params_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    )

    prompt_text = messages[0]["content"][0]["text"]
    assert "When calling `device.end_task(...)`, the `params` object must satisfy this JSON schema:" in prompt_text
    assert "'order_id'" in prompt_text


def test_prompt_builder_formats_empty_history():
    image = Image.new("RGB", (8, 8), "white")
    messages = build_prompt_messages(
        task_description="Press OK",
        screenshot=image,
        action_history=[],
        action_history_len=4,
    )

    prompt_text = messages[0]["content"][0]["text"]
    assert "(No previous actions)" in prompt_text


def test_screenshot_hash_is_stable_for_identical_resized_images():
    image_a = Image.new("RGB", (16, 16), "blue")
    image_b = Image.new("RGB", (16, 16), "blue")

    assert screenshot_hash(image_a) == screenshot_hash(image_b)


class _DummyAutomation:
    def __init__(self):
        self.closed = False

    def type(self, text: str, append: bool = False):
        raise RuntimeError("Active element is 'body', not an input field")

    def close(self):
        self.closed = True


class _FinishAutomation:
    def __init__(self):
        self.finish_calls = []
        self.tasks = [{"task_id": 3, "params": None}]

    def finish_evaluation(self, task_id: int = 0, params=None):
        self.finish_calls.append({"task_id": task_id, "params": params})
        return {"success": True}

    def close(self):
        pass


class _CoordinateAutomation:
    def __init__(self):
        self.clicks = []
        self.long_presses = []
        self.drags = []

    def click(self, x: int, y: int):
        self.clicks.append((x, y))

    def long_press(self, x: int, y: int):
        self.long_presses.append((x, y))

    def drag(self, x1: int, y1: int, x2: int, y2: int):
        self.drags.append((x1, y1, x2, y2))

    def close(self):
        pass


class _TaskAutomation:
    def __init__(self):
        self.tasks = [
            {
                "task_id": 3,
                "description": "Place the order",
                "params": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            }
        ]

    def close(self):
        pass


def test_browser_step_turns_execution_errors_into_invalid_steps(monkeypatch):
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig(post_action_wait_seconds=0))
    dummy = _DummyAutomation()
    browser._automation = dummy
    browser._env_id = "12306"

    result = browser.step({"action": "input_text", "text": "hello"})

    assert result["reward"] == 0.0
    assert result["done"] is False
    assert result["info"]["invalid_action"] is True
    assert "Active element" in result["info"]["error"]
    assert dummy.closed is False


def test_browser_step_forwards_finish_params_to_scalewob():
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig())
    dummy = _FinishAutomation()
    browser._automation = dummy
    browser._env_id = "12306"
    browser._task_id = 7

    result = browser.step({"action": "finish", "status": "finished", "params": {"order_id": "123"}})

    assert result["reward"] == 1.0
    assert result["done"] is True
    assert dummy.finish_calls == [{"task_id": 7, "params": {"order_id": "123"}}]


def test_browser_finish_infeasible_status_short_circuits_zero_reward():
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig())
    dummy = _FinishAutomation()
    browser._automation = dummy
    browser._env_id = "12306"
    browser._task_id = 7

    result = browser.step({"action": "finish", "status": "infeasible"})

    assert result["reward"] == 0.0
    assert result["done"] is True
    assert dummy.finish_calls == []


def test_browser_finish_failed_status_short_circuits_zero_reward():
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig())
    dummy = _FinishAutomation()
    browser._automation = dummy
    browser._env_id = "12306"
    browser._task_id = 7

    result = browser.step({"action": "finish", "status": "failed"})

    assert result["reward"] == 0.0
    assert result["done"] is True
    assert dummy.finish_calls == []


def test_browser_finish_uses_canonical_task_id_when_available():
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig())
    dummy = _FinishAutomation()
    browser._automation = dummy
    browser._env_id = "12306"
    browser._task_id = "3"
    browser._canonical_task_id = 3

    result = browser.step({"action": "finish", "status": "finished", "params": {"order_id": "123"}})

    assert result["reward"] == 1.0
    assert result["done"] is True
    assert dummy.finish_calls == [{"task_id": 3, "params": {"order_id": "123"}}]


def test_browser_get_task_metadata_matches_string_task_id():
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig())
    browser._automation = _TaskAutomation()
    browser._env_id = "shop"
    browser._task_id = "3"

    metadata = browser.get_task_metadata()

    assert metadata is not None
    assert metadata["task_id"] == 3
    assert metadata["params"]["required"] == ["order_id"]


def test_browser_maps_normalized_click_to_screenshot_pixels():
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig(post_action_wait_seconds=0))
    dummy = _CoordinateAutomation()
    browser._automation = dummy
    browser._env_id = "12306"
    browser.last_screenshot_size = (390, 844)

    result = browser.step({"action": "tap", "x": 500, "y": 500})

    assert dummy.clicks == [(194, 422)]
    assert result["info"]["normalized_action"] == {"action": "tap", "x": 500, "y": 500}
    assert result["info"]["executed_action"] == {"action": "tap", "x": 194, "y": 422}


def test_browser_maps_max_normalized_coordinate_to_bottom_right_pixel():
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig(post_action_wait_seconds=0))
    dummy = _CoordinateAutomation()
    browser._automation = dummy
    browser._env_id = "12306"
    browser.last_screenshot_size = (390, 844)

    result = browser.step({"action": "tap", "x": 1000, "y": 1000})

    assert dummy.clicks == [(389, 843)]
    assert result["info"]["executed_action"] == {"action": "tap", "x": 389, "y": 843}


def test_browser_maps_swipe_coordinates_to_screenshot_pixels():
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig(post_action_wait_seconds=0))
    dummy = _CoordinateAutomation()
    browser._automation = dummy
    browser._env_id = "12306"
    browser.last_screenshot_size = (390, 844)

    result = browser.step({"action": "swipe", "x1": 0, "y1": 250, "x2": 1000, "y2": 750})

    assert dummy.drags == [(0, 211, 389, 632)]
    assert result["info"]["normalized_action"] == {"action": "swipe", "x1": 0, "y1": 250, "x2": 1000, "y2": 750}
    assert result["info"]["executed_action"] == {"action": "swipe", "x1": 0, "y1": 211, "x2": 389, "y2": 632}


def test_browser_wait_action_sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr("verl.experimental.scalewob.browser.time.sleep", lambda seconds: calls.append(seconds))
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig(wait_action_seconds=1.25))
    browser._automation = _CoordinateAutomation()
    browser._env_id = "12306"

    result = browser.step({"action": "wait"})

    assert result["info"]["executed_action"] == {"action": "wait"}
    assert calls == [1.25]


def test_browser_operation_timeout_returns_invalid_action():
    class _BlockingAutomation:
        def click(self, x: int, y: int):
            time.sleep(0.05)

    browser = ScaleWoBBrowser(
        ScaleWoBBrowserConfig(post_action_wait_seconds=0, browser_operation_timeout_seconds=0.001)
    )
    browser._automation = _BlockingAutomation()
    browser._env_id = "12306"
    browser.last_screenshot_size = (390, 844)

    result = browser.step({"action": "tap", "x": 500, "y": 500})

    assert result["info"]["invalid_action"] is True
    assert result["info"]["executed_action"] == {"action": "tap", "x": 194, "y": 422}
    assert "timed out" in result["info"]["error"]


def test_browser_call_with_timeout_raises_timeout_error():
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig(browser_operation_timeout_seconds=0.001))

    def blocking():
        time.sleep(0.05)

    try:
        browser._call_with_timeout(blocking)
    except ScaleWoBBrowserOperationTimeoutError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected timeout")


def test_browser_reset_recreates_automation_after_failed_attempt():
    class _ResetAutomation:
        instances = []

        def __init__(self, should_fail: bool):
            self.should_fail = should_fail
            self.closed = False
            self.starts = 0
            _ResetAutomation.instances.append(self)

        def start(self):
            self.starts += 1
            if self.should_fail:
                raise RuntimeError("first start failed")

        def start_evaluation(self):
            pass

        def close(self):
            self.closed = True

    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig(reset_retries=2, reset_retry_delay_seconds=0))
    created = 0

    def fake_ensure_automation():
        nonlocal created
        if browser._automation is None:
            browser._automation = _ResetAutomation(should_fail=created == 0)
            created += 1
        return browser._automation

    browser._ensure_automation = fake_ensure_automation

    result = browser.reset({"env_id": "shop", "task_id": 3})

    assert result == {"env_id": "shop", "task_id": 3}
    assert len(_ResetAutomation.instances) == 2
    assert _ResetAutomation.instances[0].closed is True
    assert _ResetAutomation.instances[0].starts == 1
    assert _ResetAutomation.instances[1].starts == 1
