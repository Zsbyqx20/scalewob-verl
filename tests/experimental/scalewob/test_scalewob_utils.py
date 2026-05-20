from PIL import Image

from verl.experimental.scalewob.actions import parse_action
from verl.experimental.scalewob.browser import ScaleWoBBrowser, ScaleWoBBrowserConfig
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

    def finish_evaluation(self, task_id: int = 0, params=None):
        self.finish_calls.append({"task_id": task_id, "params": params})
        return {"success": True}

    def close(self):
        pass


def test_browser_step_turns_execution_errors_into_invalid_steps(monkeypatch):
    browser = ScaleWoBBrowser(ScaleWoBBrowserConfig())
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
