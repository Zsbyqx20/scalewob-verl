from PIL import Image

from verl.experimental.scalewob.actions import parse_action
from verl.experimental.scalewob.browser import ScaleWoBBrowser, ScaleWoBBrowserConfig
from verl.experimental.scalewob.prompt import build_prompt_messages
from verl.experimental.scalewob.vision import screenshot_hash


def test_action_parser_accepts_aliases_and_clamps_coordinates():
    parsed = parse_action('```json\n{"action": "click", "x": -3, "y": 1204}\n```')

    assert parsed.is_action_valid == 1
    assert parsed.action == {"action": "tap", "x": 0, "y": 1000}


def test_bad_json_returns_fallback_wait():
    parsed = parse_action("{not-json")

    assert parsed.is_action_valid == 0
    assert parsed.action == {"action": "wait"}
    assert parsed.error is not None


def test_prompt_builder_emits_exactly_one_image_placeholder():
    image = Image.new("RGB", (8, 8), "white")
    messages = build_prompt_messages(
        task_description="Press OK",
        screenshot=image,
        action_history=[{"action": "tap"}],
        action_history_len=4,
    )

    image_items = [
        item
        for message in messages
        for item in (message["content"] if isinstance(message["content"], list) else [])
        if item.get("type") == "image"
    ]
    assert len(image_items) == 1
    assert image_items[0]["image"] is image


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
