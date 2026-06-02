from __future__ import annotations

import asyncio
from typing import Any

import pytest
from omegaconf import OmegaConf
from PIL import Image

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.scalewob.agent_loop import ScaleWoBAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeTokenizer:
    padding_side = "right"

    def __init__(self, decoded: dict[tuple[int, ...], str]):
        self._decoded = decoded

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del add_generation_prompt, tokenize, kwargs
        return [100 + idx for idx, _ in enumerate(messages)]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return self._decoded[tuple(token_ids)]


class _FakeServerManager:
    def __init__(self, outputs: list[TokenOutput]):
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> TokenOutput:
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "sampling_params": sampling_params,
                "image_data": image_data,
                "video_data": video_data,
            }
        )
        return self.outputs.pop(0)


class _ScriptedBrowser:
    instances: list[_ScriptedBrowser] = []
    step_results: list[dict[str, Any]] = []
    fail_on_reset: Exception | None = None
    fail_on_screenshot: Exception | None = None
    screenshots: list[Image.Image] = []
    tasks: list[dict[str, Any]] = []

    def __init__(self, config):
        self.config = config
        self.events: list[Any] = []
        self.actions: list[dict[str, Any]] = []
        self.closed = False
        _ScriptedBrowser.instances.append(self)

    def reset(self, scalewob_info: dict[str, Any]) -> dict[str, Any]:
        self.events.append(("reset", dict(scalewob_info)))
        if _ScriptedBrowser.fail_on_reset is not None:
            raise _ScriptedBrowser.fail_on_reset
        return {"env_id": scalewob_info.get("env_id"), "task_id": scalewob_info.get("task_id", 0)}

    def get_task_metadata(self, task_id=None) -> dict[str, Any] | None:
        target_task_id = task_id if task_id is not None else self.events[0][1].get("task_id", 0)
        for task in _ScriptedBrowser.tasks:
            if str(task.get("task_id")) == str(target_task_id):
                return dict(task)
        return None

    def screenshot(self) -> Image.Image:
        self.events.append("screenshot")
        if _ScriptedBrowser.fail_on_screenshot is not None:
            raise _ScriptedBrowser.fail_on_screenshot
        if _ScriptedBrowser.screenshots:
            return _ScriptedBrowser.screenshots.pop(0)
        return Image.new("RGB", (16, 16), "white")

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        self.events.append(("step", action))
        self.actions.append(action)
        return _ScriptedBrowser.step_results.pop(0)

    def close(self) -> None:
        self.events.append("close")
        self.closed = True


def _install_fake_browser(
    monkeypatch,
    step_results: list[dict[str, Any]],
    *,
    fail_on_reset=None,
    fail_on_screenshot=None,
    screenshots=None,
    tasks=None,
):
    _ScriptedBrowser.instances = []
    _ScriptedBrowser.step_results = list(step_results)
    _ScriptedBrowser.fail_on_reset = fail_on_reset
    _ScriptedBrowser.fail_on_screenshot = fail_on_screenshot
    _ScriptedBrowser.screenshots = list(screenshots or [])
    _ScriptedBrowser.tasks = list(tasks or [])
    monkeypatch.setattr("verl.experimental.scalewob.agent_loop.ScaleWoBBrowser", _ScriptedBrowser)


def _make_loop(server_manager: _FakeServerManager, tokenizer: _FakeTokenizer) -> ScaleWoBAgentLoop:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": 16,
                    "scalewob": {
                        "max_env_steps": 3,
                        "target_image_hw": [16, 16],
                        "anchor_hash_hw": [8, 8],
                        "action_history_len": 4,
                        "max_stale_steps": 2,
                    },
                },
                "model": {},
            },
            "data": {"apply_chat_template_kwargs": {}, "tool_config_path": None},
        }
    )
    return ScaleWoBAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
    )


def _patch_multimodal_methods(monkeypatch, loop: ScaleWoBAgentLoop, captured_messages: list[list[dict[str, Any]]]):
    async def fake_process_vision_info(messages: list[dict[str, Any]]) -> dict[str, Any]:
        image = next(item["image"] for item in messages[0]["content"] if item.get("type") == "image")
        return {"images": [image]}

    async def fake_apply_chat_template(messages: list[dict[str, Any]], images: list[Any]) -> list[int]:
        del images
        captured_messages.append(messages)
        return [10, len(captured_messages)]

    monkeypatch.setattr(loop, "process_vision_info", fake_process_vision_info)
    monkeypatch.setattr(loop, "apply_chat_template", fake_apply_chat_template)


def test_scalewob_agent_loop_runs_multi_step_rollout(monkeypatch):
    _install_fake_browser(
        monkeypatch,
        [
            {"observation": None, "reward": 0.0, "done": False, "info": {}},
            {"observation": None, "reward": 0.5, "done": True, "info": {"final_reward": 1.0}},
        ],
    )
    server_manager = _FakeServerManager(
        [
            TokenOutput(token_ids=[11], log_probs=[-0.1], num_preempted=1),
            TokenOutput(token_ids=[22], log_probs=[-0.2], num_preempted=0),
        ]
    )
    tokenizer = _FakeTokenizer(
        {
            (11,): "Thought: Tap the target.\nAction: `device.click(100, 400)`",
            (22,): "Thought: Done.\nAction: `device.end_task('finished', {'order_id': '123'})`",
        }
    )
    loop = _make_loop(server_manager, tokenizer)
    captured_messages: list[list[dict[str, Any]]] = []
    _patch_multimodal_methods(monkeypatch, loop, captured_messages)

    output = asyncio.run(
        loop.run(
            sampling_params={"temperature": 0.0},
            extra_info={"index": 7, "description": "Place the order", "scalewob": {"env_id": "shop", "task_id": 3}},
            uid="sample-7",
        )
    )

    browser = _ScriptedBrowser.instances[0]
    assert browser.closed is True
    assert browser.events[0] == ("reset", {"env_id": "shop", "task_id": 3})
    assert browser.events.count("screenshot") == 2
    assert browser.actions == [
        {"action": "tap", "x": 100, "y": 400},
        {"action": "finish", "status": "finished", "params": {"order_id": "123"}},
    ]
    assert len(captured_messages) == 2
    second_prompt_text = captured_messages[1][0]["content"][0]["text"]
    assert "Step 1 Action: device.click(100, 400)" in second_prompt_text

    assert output.reward_score == 1.0
    assert output.extra_fields["finished"] is True
    assert output.extra_fields["rollout_error"] is None
    assert output.extra_fields["rewards"] == [0.0, 0.5]
    assert len(output.step_outputs) == 2
    assert [step.reward_score for step in output.step_outputs] == [1.0, 1.0]
    assert [step.extra_fields["step_id"] for step in output.step_outputs] == [0, 1]
    assert [step.extra_fields["traj_uid"] for step in output.step_outputs] == [output.extra_fields["traj_uid"]] * 2

    required_fields = {
        "step_id",
        "traj_uid",
        "anchor_obs",
        "raw_action",
        "normalized_action",
        "rewards",
        "is_action_valid",
    }
    for step in output.step_outputs:
        assert required_fields <= step.extra_fields.keys()
    assert output.step_outputs[0].extra_fields["raw_action"] == "device.click(100, 400)"
    assert output.step_outputs[0].extra_fields["normalized_action"] == {"action": "tap", "x": 100, "y": 400}
    assert output.step_outputs[0].extra_fields["executed_action"] is None
    assert output.step_outputs[1].extra_fields["is_action_valid"] == 1
    assert server_manager.calls[0]["image_data"] is not None


def test_scalewob_agent_loop_injects_task_params_schema(monkeypatch):
    _install_fake_browser(
        monkeypatch,
        [{"observation": None, "reward": 1.0, "done": True, "info": {"final_reward": 1.0}}],
        tasks=[
            {
                "task_id": 3,
                "description": "Place the order",
                "params": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            }
        ],
    )
    server_manager = _FakeServerManager([TokenOutput(token_ids=[22], log_probs=[-0.2], num_preempted=0)])
    tokenizer = _FakeTokenizer({(22,): "Thought: Done.\nAction: `device.end_task('finished')`"})
    loop = _make_loop(server_manager, tokenizer)
    captured_messages: list[list[dict[str, Any]]] = []
    _patch_multimodal_methods(monkeypatch, loop, captured_messages)

    asyncio.run(
        loop.run(
            sampling_params={"temperature": 0.0},
            extra_info={"description": "Place the order", "scalewob": {"env_id": "shop", "task_id": "3"}},
        )
    )

    prompt_text = captured_messages[0][0]["content"][0]["text"]
    assert "When calling `device.end_task(...)`, the `params` object must satisfy this JSON schema:" in prompt_text
    assert "'order_id'" in prompt_text


def test_scalewob_agent_loop_emits_inactive_step_on_reset_failure(monkeypatch):
    _install_fake_browser(
        monkeypatch,
        [],
        fail_on_reset=RuntimeError("Failed to fetch tasks: window.getTasks is not a function"),
    )
    loop = _make_loop(_FakeServerManager([]), _FakeTokenizer({}))
    captured_messages: list[list[dict[str, Any]]] = []
    _patch_multimodal_methods(monkeypatch, loop, captured_messages)

    output = asyncio.run(loop.run(sampling_params={}, extra_info={"scalewob": {"env_id": "shop"}}))

    assert _ScriptedBrowser.instances[0].closed is True
    assert _ScriptedBrowser.instances[0].events[-1] == "close"
    assert output.reward_score == 0.0
    assert output.extra_fields["finished"] is False
    assert output.extra_fields["rollout_error"] is None
    assert output.extra_fields["rewards"] == [0.0]
    assert len(output.step_outputs) == 1
    step = output.step_outputs[0]
    assert step.response_mask == [0]
    assert step.extra_fields["active_masks"] == 0
    assert step.extra_fields["is_action_valid"] == 0
    assert step.extra_fields["normalized_action"] == {"action": "browser_error", "phase": "reset"}
    assert step.extra_fields["rollout_error"] == "browser_reset_failed"
    assert "window.getTasks is not a function" in step.extra_fields["action_exec_error"]


def test_scalewob_agent_loop_closes_browser_on_screenshot_failure(monkeypatch):
    _install_fake_browser(
        monkeypatch,
        [],
        fail_on_screenshot=RuntimeError("screenshot failed"),
    )
    loop = _make_loop(_FakeServerManager([]), _FakeTokenizer({}))
    captured_messages: list[list[dict[str, Any]]] = []
    _patch_multimodal_methods(monkeypatch, loop, captured_messages)

    with pytest.raises(RuntimeError, match="screenshot failed"):
        asyncio.run(loop.run(sampling_params={}, extra_info={"scalewob": {"env_id": "shop"}}))

    assert _ScriptedBrowser.instances[0].closed is True
    assert _ScriptedBrowser.instances[0].events[-1] == "close"


def test_scalewob_agent_loop_continues_after_invalid_finish_action(monkeypatch):
    _install_fake_browser(
        monkeypatch,
        [
            {
                "observation": None,
                "reward": 0.0,
                "done": False,
                "info": {"invalid_action": True, "error": "finish rejected"},
            },
            {"observation": None, "reward": 0.0, "done": False, "info": {}},
            {"observation": None, "reward": 1.0, "done": True, "info": {"final_reward": 1.0}},
        ],
    )
    server_manager = _FakeServerManager(
        [
            TokenOutput(token_ids=[31], log_probs=[0.0]),
            TokenOutput(token_ids=[32], log_probs=[0.0]),
            TokenOutput(token_ids=[33], log_probs=[0.0]),
        ]
    )
    tokenizer = _FakeTokenizer(
        {
            (31,): "Thought: Try to finish.\nAction: `device.end_task('finished')`",
            (32,): "Thought: Continue.\nAction: `device.click(100, 400)`",
            (33,): "Thought: Finish now.\nAction: `device.end_task('finished')`",
        }
    )
    loop = _make_loop(server_manager, tokenizer)
    captured_messages: list[list[dict[str, Any]]] = []
    _patch_multimodal_methods(monkeypatch, loop, captured_messages)

    output = asyncio.run(loop.run(sampling_params={}, extra_info={"scalewob": {"env_id": "shop"}}))

    assert len(output.step_outputs) == 3
    assert output.extra_fields["finished"] is True
    assert output.step_outputs[0].extra_fields["is_action_valid"] == 0
    assert output.step_outputs[0].extra_fields["action_exec_error"] == "finish rejected"
    assert _ScriptedBrowser.instances[0].actions == [
        {"action": "finish", "status": "finished"},
        {"action": "tap", "x": 100, "y": 400},
        {"action": "finish", "status": "finished"},
    ]


def test_scalewob_agent_loop_stops_after_repeated_identical_screenshots(monkeypatch):
    _install_fake_browser(
        monkeypatch,
        [
            {
                "observation": None,
                "reward": 0.0,
                "done": False,
                "info": {"executed_action": {"action": "tap", "x": 1, "y": 1}},
            },
            {
                "observation": None,
                "reward": 0.0,
                "done": False,
                "info": {"executed_action": {"action": "tap", "x": 1, "y": 1}},
            },
            {
                "observation": None,
                "reward": 0.0,
                "done": False,
                "info": {"executed_action": {"action": "tap", "x": 1, "y": 1}},
            },
        ],
        screenshots=[Image.new("RGB", (16, 16), "white") for _ in range(3)],
    )
    server_manager = _FakeServerManager(
        [
            TokenOutput(token_ids=[41], log_probs=[0.0]),
            TokenOutput(token_ids=[42], log_probs=[0.0]),
            TokenOutput(token_ids=[43], log_probs=[0.0]),
        ]
    )
    tokenizer = _FakeTokenizer(
        {
            (41,): "Thought: Tap.\nAction: `device.click(100, 400)`",
            (42,): "Thought: Tap again.\nAction: `device.click(100, 400)`",
            (43,): "Thought: Tap again.\nAction: `device.click(100, 400)`",
        }
    )
    loop = _make_loop(server_manager, tokenizer)
    captured_messages: list[list[dict[str, Any]]] = []
    _patch_multimodal_methods(monkeypatch, loop, captured_messages)

    output = asyncio.run(loop.run(sampling_params={}, extra_info={"scalewob": {"env_id": "shop"}}))

    assert len(output.step_outputs) == 3
    assert output.extra_fields["rollout_error"] == "browser_stale_state"
    final_step = output.step_outputs[-1]
    assert final_step.extra_fields["rollout_error"] == "browser_stale_state"
    assert final_step.extra_fields["active_masks"] == 0
    assert final_step.extra_fields["is_action_valid"] == 0
    assert final_step.extra_fields["stale_steps"] == 2
    assert final_step.extra_fields["previous_anchor_obs"] == final_step.extra_fields["current_anchor_obs"]
    assert final_step.extra_fields["executed_action"] == {"action": "tap", "x": 1, "y": 1}


def test_scalewob_agent_loop_stops_on_browser_operation_timeout(monkeypatch):
    _install_fake_browser(
        monkeypatch,
        [
            {
                "observation": None,
                "reward": 0.0,
                "done": False,
                "info": {
                    "invalid_action": True,
                    "error": "click timed out after 0.001s",
                    "executed_action": {"action": "tap", "x": 1, "y": 1},
                },
            },
            {"observation": None, "reward": 1.0, "done": True, "info": {}},
        ],
    )
    server_manager = _FakeServerManager(
        [
            TokenOutput(token_ids=[51], log_probs=[0.0]),
            TokenOutput(token_ids=[52], log_probs=[0.0]),
        ]
    )
    tokenizer = _FakeTokenizer(
        {
            (51,): "Thought: Tap.\nAction: `device.click(100, 400)`",
            (52,): "Thought: Finish.\nAction: `device.end_task('finished')`",
        }
    )
    loop = _make_loop(server_manager, tokenizer)
    captured_messages: list[list[dict[str, Any]]] = []
    _patch_multimodal_methods(monkeypatch, loop, captured_messages)

    output = asyncio.run(loop.run(sampling_params={}, extra_info={"scalewob": {"env_id": "shop"}}))

    assert len(output.step_outputs) == 1
    assert output.extra_fields["rollout_error"] == "browser_operation_timeout"
    assert output.step_outputs[0].extra_fields["rollout_error"] == "browser_operation_timeout"
    assert output.step_outputs[0].extra_fields["action_exec_error"] == "click timed out after 0.001s"
