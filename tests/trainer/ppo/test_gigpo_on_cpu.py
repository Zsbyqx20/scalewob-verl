import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_step_discounted_returns
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage


def test_compute_step_discounted_returns_mixed_batch_order():
    rewards = np.array([1.0, 10.0, 2.0, 20.0], dtype=np.float32)
    traj_uid = np.array(["a", "b", "a", "b"], dtype=object)

    returns = compute_step_discounted_returns(rewards, traj_uid, gamma=0.5)

    torch.testing.assert_close(returns, torch.tensor([2.0, 20.0, 2.0, 20.0]))


def test_compute_step_discounted_returns_accepts_object_rewards():
    rewards = np.array([1.0, 10.0, 2.0, 20.0], dtype=object)
    traj_uid = np.array(["a", "b", "a", "b"], dtype=object)

    returns = compute_step_discounted_returns(rewards, traj_uid, gamma=0.5)

    torch.testing.assert_close(returns, torch.tensor([2.0, 20.0, 2.0, 20.0]))


def test_gigpo_advantages_are_token_shaped_masked_and_penalize_invalid_actions():
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    token_level_rewards = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    data = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "token_level_rewards": token_level_rewards,
        },
        non_tensors={
            "index": np.array([7, 7, 7, 7], dtype=object),
            "anchor_obs": np.array(["same", "same", "same", "same"], dtype=object),
            "traj_uid": np.array(["t0", "t1", "t2", "t3"], dtype=object),
            "active_masks": np.array([1, 1, 1, 1], dtype=object),
            "rewards": np.array([1.0, 1.0, 1.0, 1.0], dtype=object),
            "is_action_valid": np.array([1, 0, 1, 0], dtype=object),
        },
    )

    out = compute_advantage(
        data,
        adv_estimator="gigpo",
        gamma=1.0,
        config={
            "gigpo": {
                "step_advantage_w": 1.0,
                "mode": "mean_norm",
                "invalid_action_penalty_coef": 1.0,
            }
        },
    )

    assert out.batch["advantages"].shape == response_mask.shape
    assert out.batch["returns"].shape == response_mask.shape
    assert torch.all(out.batch["advantages"][response_mask == 0] == 0)
    assert out.batch["advantages"][1, 0] < out.batch["advantages"][0, 0]


def test_flattened_batch_padding_for_dp_zeroes_padding_loss_masks():
    data = DataProto.from_dict(
        tensors={
            "response_mask": torch.ones(3, 2),
            "advantages": torch.ones(3, 2),
            "returns": torch.ones(3, 2),
        },
        non_tensors={
            "active_masks": np.array([1, 1, 1], dtype=np.int64),
            "traj_uid": np.array(["a", "b", "c"], dtype=object),
        },
        meta_info={"rollout_flattened_steps": True},
    )
    trainer = type("_DummyTrainer", (), {"_get_dp_size": lambda self, worker_group, role: 4})()

    padded, pad_size = RayPPOTrainer._pad_flattened_batch_for_dp(
        trainer,
        data,
        worker_group=None,
        role="actor",
        zero_padding_loss=True,
    )

    assert pad_size == 1
    assert len(padded) == 4
    assert torch.all(padded.batch["response_mask"][-1] == 0)
    assert torch.all(padded.batch["advantages"][-1] == 0)
    assert torch.all(padded.batch["returns"][-1] == 0)
    assert padded.non_tensor_batch["active_masks"].tolist() == [1, 1, 1, 0]


def test_get_dp_size_falls_back_across_dispatch_mesh_names():
    class _WorkerGroup:
        world_size = 4

        def __init__(self):
            self._dispatch_info = {}

        def _query_dispatch_info(self, mesh_name):
            if mesh_name != "actor":
                raise AssertionError(f"{mesh_name} is not registered")
            return [0, 1, 2, 3]

    trainer = type("_DummyTrainer", (), {})()

    dp_size = RayPPOTrainer._get_dp_size(trainer, _WorkerGroup(), ("ref", "actor"))
    assert dp_size == 4
