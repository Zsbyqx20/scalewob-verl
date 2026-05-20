#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-data/scalewob/train.parquet}

python3 -m verl.trainer.main_ppo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TRAIN_FILE}" \
  data.train_batch_size=16 \
  data.max_prompt_length=2048 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.limit_images=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=scalewob_agent \
  actor_rollout_ref.rollout.scalewob.max_env_steps=10 \
  actor_rollout_ref.rollout.scalewob.target_image_hw="[1024,474]" \
  algorithm.adv_estimator=gigpo \
  algorithm.gigpo.invalid_action_penalty_coef=0.1 \
  reward_model.enable=false \
  critic.enable=false \
  trainer.project_name=scalewob_gigpo \
  trainer.experiment_name=qwen2_5_vl_scalewob_gigpo \
  "$@"
