#!/usr/bin/env bash
set -euo pipefail
set -x

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

ENGINE=${ENGINE:-vllm}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-VL-8B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/scalewob/train.parquet"}
VAL_FILE=${VAL_FILE:-"${TRAIN_FILE}"}

if [[ "${MODEL_PATH}" == *"A3B"* || "${MODEL_PATH}" == *"Moe"* || "${MODEL_PATH}" == *"MoE"* ]]; then
  DEFAULT_FSDP_WRAP_POLICY="[Qwen3VLMoeTextDecoderLayer]"
else
  DEFAULT_FSDP_WRAP_POLICY="[Qwen3VLTextDecoderLayer]"
fi
FSDP_WRAP_POLICY=${FSDP_WRAP_POLICY:-"${DEFAULT_FSDP_WRAP_POLICY}"}

PROJECT_NAME=${PROJECT_NAME:-scalewob_gigpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_scalewob_gigpo}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

GEN_TP=${GEN_TP:-1}
SP_SIZE=${SP_SIZE:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-10}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
MIXED_PRECISION_DTYPE=${MIXED_PRECISION_DTYPE:-bf16}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-256}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-20000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-5}
LOGGER=${LOGGER:-'["console"]'}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gigpo \
  algorithm.use_kl_in_reward=False \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=mean_std_norm \
  algorithm.gigpo.enable_similarity=false \
  algorithm.gigpo.invalid_action_penalty_coef=0.1 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${PROMPT_LENGTH}" \
  data.max_response_length="${RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.image_key=images \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  actor_rollout_ref.actor.fsdp_config.dtype="${MODEL_DTYPE}" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype="${MIXED_PRECISION_DTYPE}" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype="${MIXED_PRECISION_DTYPE}" \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype="${MIXED_PRECISION_DTYPE}" \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="${FSDP_WRAP_POLICY}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  actor_rollout_ref.ref.fsdp_config.dtype="${MODEL_DTYPE}" \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype="${MIXED_PRECISION_DTYPE}" \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.reduce_dtype="${MIXED_PRECISION_DTYPE}" \
  +actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype="${MIXED_PRECISION_DTYPE}" \
  actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
  +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="${FSDP_WRAP_POLICY}" \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.limit_images=1 \
  actor_rollout_ref.rollout.skip_tokenizer_init=False \
  actor_rollout_ref.rollout.dtype="${MODEL_DTYPE}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.agent.default_agent_loop=scalewob_agent \
  actor_rollout_ref.rollout.scalewob.max_env_steps="${MAX_ENV_STEPS}" \
  actor_rollout_ref.rollout.scalewob.target_image_hw="[1024,474]" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  critic.enable=False \
  reward.reward_model.enable=False \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_mode=auto \
  trainer.val_before_train=True \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "$@"
