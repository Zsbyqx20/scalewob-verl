# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import os
from typing import Any
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopStepOutput,
    register,
)
from verl.experimental.scalewob.actions import parse_action
from verl.experimental.scalewob.browser import ScaleWoBBrowser, ScaleWoBBrowserConfig
from verl.experimental.scalewob.debug import resolve_scalewob_debug_config
from verl.experimental.scalewob.prompt import build_prompt_messages
from verl.experimental.scalewob.vision import resize_screenshot, screenshot_hash
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _compact_error(exc: Exception, max_length: int = 2000) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if len(message) <= max_length:
        return message
    return message[: max_length - 3] + "..."


@register("scalewob_agent")
class ScaleWoBAgentLoop(AgentLoopBase):
    """ScaleWoB browser rollout loop that emits one training row per browser action step."""

    def __init__(self, *args, scalewob: dict[str, Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        rollout_scalewob = getattr(self.rollout_config, "scalewob", None)
        if rollout_scalewob is not None:
            rollout_scalewob = dict(rollout_scalewob)
        self.browser_config = ScaleWoBBrowserConfig.from_mapping(scalewob or rollout_scalewob)
        self.debug_config = resolve_scalewob_debug_config(self.config, scalewob or rollout_scalewob)
        self.response_length = self.rollout_config.response_length

    def _fallback_token_id(self) -> int:
        for attr in ("pad_token_id", "eos_token_id"):
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                return int(token_id)
        return 0

    async def _build_inactive_step_output(
        self,
        *,
        phase: str,
        error: Exception,
        description: str,
        scalewob_info: dict[str, Any],
        extra_info: dict[str, Any],
        stable_index: Any,
        traj_uid: str,
        kwargs: dict[str, Any],
        metrics: dict[str, Any],
    ) -> AgentLoopStepOutput:
        height, width = self.browser_config.target_image_hw
        screenshot = Image.new("RGB", (int(width), int(height)), "white")
        anchor_obs = f"{phase}_failed"
        messages = build_prompt_messages(
            task_description=description,
            screenshot=screenshot,
            action_history=[],
            action_history_len=self.browser_config.action_history_len,
        )
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        if images is None or len(images) != 1:
            image_count = 0 if images is None else len(images)
            raise ValueError(f"ScaleWoB error prompt must contain exactly one image, got {image_count}")
        prompt_ids = await self.apply_chat_template(messages, images=images)
        error_text = _compact_error(error)
        extra_fields = {
            "uid": kwargs.get("uid"),
            "index": stable_index,
            "traj_uid": traj_uid,
            "step_id": 0,
            "anchor_obs": anchor_obs,
            "active_masks": 0,
            "rewards": 0.0,
            "is_action_valid": 0,
            "data_source": kwargs.get("data_source", "scalewob"),
            "extra_info": extra_info,
            "turn_scores": [],
            "tool_rewards": [],
            "thought": "",
            "raw_action": "",
            "normalized_action": {"action": "browser_error", "phase": phase},
            "action_exec_error": error_text,
            "rollout_error": f"browser_{phase}_failed",
        }
        if self.debug_config["enabled"]:
            extra_fields.update(
                {
                    "env_id": scalewob_info.get("env_id")
                    or scalewob_info.get("environment_id")
                    or scalewob_info.get("env"),
                    "task_id": scalewob_info.get("task_id"),
                    "task_description": description,
                    "final_reward": 0.0,
                }
            )
            if self.debug_config["include_prompt"]:
                extra_fields["prompt_messages"] = messages
            if self.debug_config["save_screenshots"]:
                extra_fields["screenshot"] = screenshot.copy()
        return AgentLoopStepOutput(
            prompt_ids=prompt_ids,
            response_ids=[self._fallback_token_id()],
            response_mask=[0],
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            reward_score=0.0,
            num_turns=1,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields=extra_fields,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {}) or {}
        scalewob_info = dict(extra_info.get("scalewob", {}) or {})
        description = scalewob_info.get("description") or extra_info.get("description", "")
        stable_index = extra_info.get("index", kwargs.get("index", scalewob_info.get("task_id", uuid4().hex)))
        traj_uid = uuid4().hex

        browser = ScaleWoBBrowser(self.browser_config)
        step_outputs: list[AgentLoopStepOutput] = []
        action_history: list[dict[str, Any]] = []
        env_rewards: list[float] = []
        final_reward = 0.0
        finished = False
        rollout_error: str | None = None
        previous_anchor_obs: str | None = None
        stale_steps = 0
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": 0}
        task_metadata: dict[str, Any] | None = None

        try:
            reset_ok = False
            try:
                browser.reset(scalewob_info)
                reset_ok = True
                task_metadata = browser.get_task_metadata()
            except Exception as exc:
                logger.warning("ScaleWoB browser reset failed; emitting inactive rollout step: %s", _compact_error(exc))
                step_outputs.append(
                    await self._build_inactive_step_output(
                        phase="reset",
                        error=exc,
                        description=description,
                        scalewob_info=scalewob_info,
                        extra_info=extra_info,
                        stable_index=stable_index,
                        traj_uid=traj_uid,
                        kwargs=kwargs,
                        metrics=metrics,
                    )
                )
                env_rewards.append(0.0)
            for step_id in range(self.browser_config.max_env_steps if reset_ok else 0):
                screenshot = resize_screenshot(browser.screenshot(), self.browser_config.target_image_hw)
                anchor_obs = screenshot_hash(screenshot, self.browser_config.anchor_hash_hw)
                task_params_schema = None
                if task_metadata is not None:
                    task_params_schema = task_metadata.get("params")
                messages = build_prompt_messages(
                    task_description=description,
                    screenshot=screenshot,
                    action_history=action_history,
                    action_history_len=self.browser_config.action_history_len,
                    task_params_schema=task_params_schema,
                )
                multi_modal_data = await self.process_vision_info(messages)
                images = multi_modal_data.get("images")
                if images is None or len(images) != 1:
                    image_count = 0 if images is None else len(images)
                    raise ValueError(f"ScaleWoB step prompt must contain exactly one image, got {image_count}")
                prompt_ids = await self.apply_chat_template(messages, images=images)

                with simple_timer("generate_sequences", metrics):
                    token_output: TokenOutput = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=images,
                    )
                metrics["num_preempted"] += token_output.num_preempted if token_output.num_preempted is not None else 0

                response_ids = token_output.token_ids[: self.response_length]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                parsed = parse_action(response_text, coord_scale=self.browser_config.coord_scale)
                step_result = browser.step(parsed.action)
                env_reward = float(step_result.get("reward", 0.0) or 0.0)
                done = bool(step_result.get("done", False))
                step_info = step_result.get("info", {}) or {}
                execution_valid = 0 if step_info.get("invalid_action") else 1
                default_final_reward = env_reward if done else final_reward
                final_reward = float(step_info.get("final_reward", default_final_reward))
                env_rewards.append(env_reward)
                normalized_action = step_info.get("normalized_action", parsed.action)
                executed_action = step_info.get("executed_action")
                action_name = str(parsed.action.get("action", "")).strip().lower()
                stale_previous_anchor_obs = previous_anchor_obs
                if action_name in {"wait", "pause", "finish"}:
                    stale_steps = 0
                elif previous_anchor_obs is not None and anchor_obs == previous_anchor_obs:
                    stale_steps += 1
                else:
                    stale_steps = 0
                is_stale_cutoff = stale_steps >= int(self.browser_config.max_stale_steps)
                if is_stale_cutoff:
                    rollout_error = "browser_stale_state"
                    if self.browser_config.stale_action_penalty:
                        execution_valid = 0
                        env_reward = 0.0
                        env_rewards[-1] = 0.0
                previous_anchor_obs = anchor_obs

                extra_fields = {
                    "uid": kwargs.get("uid"),
                    "index": stable_index,
                    "traj_uid": traj_uid,
                    "step_id": step_id,
                    "anchor_obs": anchor_obs,
                    "active_masks": 1,
                    "rewards": env_reward,
                    "is_action_valid": int(parsed.is_action_valid and execution_valid),
                    "data_source": kwargs.get("data_source", "scalewob"),
                    "extra_info": extra_info,
                    "turn_scores": [],
                    "tool_rewards": [],
                    "thought": parsed.thought,
                    "raw_action": parsed.raw_action,
                    "normalized_action": normalized_action,
                    "executed_action": executed_action,
                    "stale_steps": stale_steps,
                    "previous_anchor_obs": stale_previous_anchor_obs,
                    "current_anchor_obs": anchor_obs,
                }
                if parsed.error:
                    extra_fields["action_parse_error"] = parsed.error
                if step_info.get("error"):
                    extra_fields["action_exec_error"] = step_info["error"]
                    if "timed out" in str(step_info["error"]):
                        rollout_error = "browser_operation_timeout"
                        extra_fields["rollout_error"] = rollout_error
                if is_stale_cutoff:
                    extra_fields["rollout_error"] = rollout_error
                    extra_fields["active_masks"] = 0 if self.browser_config.stale_action_penalty else 1
                if self.debug_config["enabled"]:
                    extra_fields.update(
                        {
                            "env_id": scalewob_info.get("env_id")
                            or scalewob_info.get("environment_id")
                            or scalewob_info.get("env"),
                            "task_id": scalewob_info.get("task_id"),
                            "task_description": description,
                            "final_reward": final_reward,
                        }
                    )
                    if self.debug_config["include_response"]:
                        extra_fields["response_text"] = response_text
                    if self.debug_config["include_prompt"]:
                        extra_fields["prompt_messages"] = messages
                    if self.debug_config["save_screenshots"]:
                        extra_fields["screenshot"] = screenshot.copy()

                step_outputs.append(
                    AgentLoopStepOutput(
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        response_logprobs=(
                            token_output.log_probs[: self.response_length] if token_output.log_probs else None
                        ),
                        routed_experts=(
                            token_output.routed_experts[: len(prompt_ids) + self.response_length]
                            if token_output.routed_experts is not None
                            else None
                        ),
                        multi_modal_data=multi_modal_data,
                        reward_score=final_reward,
                        num_turns=2,
                        metrics=AgentLoopMetrics(**metrics),
                        extra_fields=extra_fields,
                    )
                )

                action_history.append(
                    {
                        "thought": parsed.thought,
                        "raw_action": parsed.raw_action,
                        "action": parsed.action,
                        "valid": parsed.is_action_valid,
                        "reward": env_reward,
                    }
                )
                finished = parsed.action["action"] == "finish" and done
                if is_stale_cutoff:
                    break
                if rollout_error == "browser_operation_timeout":
                    break
                if finished or done:
                    break
                if len(response_ids) >= self.response_length:
                    break
        finally:
            browser.close()

        for step_output in step_outputs:
            step_output.reward_score = final_reward
            if self.debug_config["enabled"]:
                step_output.extra_fields["final_reward"] = final_reward
        if not step_outputs:
            raise RuntimeError("ScaleWoB rollout produced no step outputs")

        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            reward_score=final_reward,
            num_turns=0,
            metrics=AgentLoopMetrics(**metrics),
            step_outputs=step_outputs,
            extra_fields={
                "traj_uid": traj_uid,
                "finished": finished,
                "rewards": env_rewards,
                "rollout_error": rollout_error,
            },
        )
