# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopStepOutput,
    register,
)
from verl.experimental.scalewob.actions import parse_action
from verl.experimental.scalewob.browser import ScaleWoBBrowser, ScaleWoBBrowserConfig
from verl.experimental.scalewob.prompt import build_prompt_messages
from verl.experimental.scalewob.vision import resize_screenshot, screenshot_hash
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("scalewob_agent")
class ScaleWoBAgentLoop(AgentLoopBase):
    """ScaleWoB browser rollout loop that emits one training row per browser action step."""

    def __init__(self, *args, scalewob: dict[str, Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        rollout_scalewob = getattr(self.rollout_config, "scalewob", None)
        if rollout_scalewob is not None:
            rollout_scalewob = dict(rollout_scalewob)
        self.browser_config = ScaleWoBBrowserConfig.from_mapping(scalewob or rollout_scalewob)
        self.response_length = self.rollout_config.response_length

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
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": 0}

        try:
            browser.reset(scalewob_info)
            for step_id in range(self.browser_config.max_env_steps):
                screenshot = resize_screenshot(browser.screenshot(), self.browser_config.target_image_hw)
                anchor_obs = screenshot_hash(screenshot, self.browser_config.anchor_hash_hw)
                messages = build_prompt_messages(
                    task_description=description,
                    screenshot=screenshot,
                    action_history=action_history,
                    action_history_len=self.browser_config.action_history_len,
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
                }
                if parsed.error:
                    extra_fields["action_parse_error"] = parsed.error
                if step_info.get("error"):
                    extra_fields["action_exec_error"] = step_info["error"]

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

                action_history.append({"action": parsed.action, "valid": parsed.is_action_valid, "reward": env_reward})
                finished = parsed.action["action"] == "finish"
                if finished or done:
                    break
                if len(response_ids) >= self.response_length:
                    break
        finally:
            browser.close()

        for step_output in step_outputs:
            step_output.reward_score = final_reward
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
            extra_fields={"traj_uid": traj_uid, "finished": finished, "rewards": env_rewards},
        )
