import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_step_discounted_returns
from verl.trainer.ppo.ray_trainer import compute_advantage


def test_compute_step_discounted_returns_mixed_batch_order():
    rewards = np.array([1.0, 10.0, 2.0, 20.0], dtype=np.float32)
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
            "active_masks": np.array([1, 1, 1, 1], dtype=np.int64),
            "rewards": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "is_action_valid": np.array([1, 0, 1, 0], dtype=np.int64),
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
