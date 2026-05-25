import numpy as np
import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopStepOutput,
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)
from verl.experimental.scalewob.debug import log_scalewob_batch_debug, resolve_scalewob_debug_config
from verl.protocol import DataProto


def _internal(
    step_id: int,
    reward_score: float,
    index: int = 5,
    traj_uid: str = "traj-a",
    extra_fields: dict | None = None,
) -> _InternalAgentLoopOutput:
    prompt_ids = torch.tensor([[101, 102]], dtype=torch.long)
    response_ids = torch.tensor([[step_id + 11, 0, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    position_ids = torch.arange(5).unsqueeze(0)
    return _InternalAgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        position_ids=position_ids,
        response_mask=response_mask,
        attention_mask=attention_mask,
        response_logprobs=None,
        routed_experts=None,
        multi_modal_inputs={"image_grid_thw": torch.ones(1, 3, dtype=torch.long), "images_seqlens": torch.ones(1)},
        multi_modal_data=None,
        reward_score=reward_score,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "uid": f"uid-{step_id}",
            "index": index,
            "traj_uid": traj_uid,
            "step_id": step_id,
            "anchor_obs": f"hash-{step_id}",
            "active_masks": 1,
            "rewards": 0.25,
            "is_action_valid": 1,
            "thought": "Tap the target.",
            "raw_action": "device.click(100, 400)",
            "normalized_action": {"action": "tap", "x": 100, "y": 400},
            "data_source": "scalewob",
            "raw_prompt": [{"role": "user", "content": "x"}],
            **(extra_fields or {}),
        },
    )


def test_flattened_agent_loop_outputs_are_flattened_to_dataproto():
    dummy_worker = type("_DummyWorker", (), {"reward_loop_worker_handles": None})()

    out = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[[_internal(0, 1.0), _internal(1, 1.0)], [_internal(0, 0.0, index=6, traj_uid="traj-b")]],
        input_non_tensor_batch={
            "index": np.array([5, 6], dtype=object),
            "uid": np.array(["sample-5", "sample-6"], dtype=object),
            "agent_name": np.array(["scalewob_agent", "scalewob_agent"], dtype=object),
        },
    )

    assert len(out) == 3
    assert out.meta_info["rollout_flattened_steps"] is True
    assert out.non_tensor_batch["index"].tolist() == [5, 5, 6]
    assert out.non_tensor_batch["step_id"].tolist() == [0, 1, 0]
    assert out.non_tensor_batch["traj_uid"].tolist() == ["traj-a", "traj-a", "traj-b"]
    assert out.non_tensor_batch["anchor_obs"].tolist() == ["hash-0", "hash-1", "hash-0"]
    assert out.non_tensor_batch["rewards"].tolist() == [0.25, 0.25, 0.25]
    assert out.non_tensor_batch["is_action_valid"].tolist() == [1, 1, 1]
    assert out.non_tensor_batch["active_masks"].tolist() == [1, 1, 1]
    assert out.non_tensor_batch["step_id"].dtype == np.int64
    assert out.non_tensor_batch["rewards"].dtype == np.float32
    assert out.non_tensor_batch["is_action_valid"].dtype == np.int64
    assert out.non_tensor_batch["active_masks"].dtype == np.int64
    assert out.non_tensor_batch["raw_action"].tolist() == [
        "device.click(100, 400)",
        "device.click(100, 400)",
        "device.click(100, 400)",
    ]
    assert out.non_tensor_batch["action_exec_error"].tolist() == [None, None, None]
    assert out.non_tensor_batch["action_parse_error"].tolist() == [None, None, None]
    assert out.non_tensor_batch["rollout_error"].tolist() == [None, None, None]
    assert out.non_tensor_batch["screenshot"].tolist() == [None, None, None]
    assert "rm_scores" in out.batch
    assert len(out.non_tensor_batch["multi_modal_inputs"]) == 3


def test_agent_loop_step_output_type_exists_for_scale_wob_rows():
    step = AgentLoopStepOutput(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        metrics=AgentLoopMetrics(),
        extra_fields={"step_id": 0},
    )
    assert step.extra_fields["step_id"] == 0


def test_rollout_error_field_survives_concat_when_only_one_worker_has_failure():
    dummy_worker = type("_DummyWorker", (), {"reward_loop_worker_handles": None})()
    failed_chunk = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[[_internal(0, 0.0, extra_fields={"rollout_error": "browser_reset_failed"})]],
        input_non_tensor_batch={
            "index": np.array([5], dtype=object),
            "uid": np.array(["sample-5"], dtype=object),
            "agent_name": np.array(["scalewob_agent"], dtype=object),
        },
    )
    normal_chunk = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[[_internal(0, 1.0, index=6, traj_uid="traj-b")]],
        input_non_tensor_batch={
            "index": np.array([6], dtype=object),
            "uid": np.array(["sample-6"], dtype=object),
            "agent_name": np.array(["scalewob_agent"], dtype=object),
        },
    )

    out = DataProto.concat([failed_chunk, normal_chunk])

    assert len(out) == 2
    assert out.non_tensor_batch["rollout_error"].tolist() == ["browser_reset_failed", None]


def test_scalewob_debug_writes_sampled_jsonl_and_screenshots_from_flattened_batch(tmp_path):
    dummy_worker = type("_DummyWorker", (), {"reward_loop_worker_handles": None})()
    screenshot_0 = Image.new("RGB", (8, 8), "red")
    screenshot_1 = Image.new("RGB", (8, 8), "blue")
    out = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[
            [
                _internal(0, 1.0, extra_fields={"screenshot": screenshot_0}),
                _internal(1, 1.0, extra_fields={"screenshot": screenshot_1}),
            ]
        ],
        input_non_tensor_batch={
            "index": np.array([5], dtype=object),
            "uid": np.array(["sample-5"], dtype=object),
            "agent_name": np.array(["scalewob_agent"], dtype=object),
        },
    )
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "scalewob": {
                    "debug": {
                        "enabled": True,
                        "log_every_n_steps": 1,
                        "save_jsonl": True,
                        "save_screenshots": True,
                        "output_dir": str(tmp_path),
                    }
                }
            }
        }
    }

    log_scalewob_batch_debug(out, config, global_step=3)

    debug_config = resolve_scalewob_debug_config(config)
    assert debug_config["enabled"] is True
    assert debug_config["save_screenshots"] is True
    jsonl_files = list((tmp_path / "global_step=3").glob("trajectory_*.jsonl"))
    assert len(jsonl_files) == 1
    lines = jsonl_files[0].read_text().strip().splitlines()
    assert len(lines) == 2
    assert '"raw_action": "device.click(100, 400)"' in lines[0]
    assert '"screenshot_path": "screenshots/' in lines[0]
    screenshots = sorted((tmp_path / "global_step=3" / "screenshots").glob("trajectory_*.png"))
    assert len(screenshots) == 2
