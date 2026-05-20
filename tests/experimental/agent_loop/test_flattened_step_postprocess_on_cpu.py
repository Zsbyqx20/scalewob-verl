import numpy as np
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopStepOutput,
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)


def _internal(step_id: int, reward_score: float, index: int = 5, traj_uid: str = "traj-a") -> _InternalAgentLoopOutput:
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
            "data_source": "scalewob",
            "raw_prompt": [{"role": "user", "content": "x"}],
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
