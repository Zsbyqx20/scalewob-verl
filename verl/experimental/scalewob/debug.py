# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
import logging
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_DEBUG_CONFIG = {
    "enabled": False,
    "log_every_n_steps": 10,
    "max_trajectories": 2,
    "max_steps_per_trajectory": 10,
    "include_prompt": False,
    "include_response": True,
    "save_jsonl": False,
    "save_screenshots": False,
    "output_dir": None,
}


def _to_plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _get(value: Any, key: str, default: Any = None) -> Any:
    value = _to_plain(value)
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _nested_get(value: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    current = value
    for key in keys:
        current = _get(current, key, default)
        if current is default:
            return default
    return current


def resolve_scalewob_debug_config(config: Any, scalewob_config: Any = None) -> dict[str, Any]:
    scalewob = scalewob_config
    if scalewob is None:
        scalewob = _nested_get(config, ("actor_rollout_ref", "rollout", "scalewob"), None)
    if scalewob is None:
        scalewob = _nested_get(config, ("rollout", "scalewob"), None)

    debug = _get(scalewob, "debug", {}) or {}
    debug = _to_plain(debug) or {}
    resolved = dict(DEFAULT_DEBUG_CONFIG)
    resolved.update({key: value for key, value in dict(debug).items() if key in resolved})

    if resolved["output_dir"] is None:
        default_local_dir = _nested_get(config, ("trainer", "default_local_dir"), None)
        if default_local_dir:
            resolved["output_dir"] = os.path.join(str(default_local_dir), "scalewob_debug")
        else:
            resolved["output_dir"] = "./scalewob_debug"

    for key in ("log_every_n_steps", "max_trajectories", "max_steps_per_trajectory"):
        resolved[key] = max(1, int(resolved[key]))
    resolved["enabled"] = bool(resolved["enabled"])
    resolved["include_prompt"] = bool(resolved["include_prompt"])
    resolved["include_response"] = bool(resolved["include_response"])
    resolved["save_jsonl"] = bool(resolved["save_jsonl"])
    resolved["save_screenshots"] = bool(resolved["save_screenshots"])
    return resolved


def should_log_scalewob_debug(config: dict[str, Any], global_step: int | None = None) -> bool:
    if not config.get("enabled", False):
        return False
    if global_step is None:
        return True
    return int(global_step) % int(config["log_every_n_steps"]) == 0


def _as_numpy(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values, dtype=object)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _batch_records(data: DataProto, debug_config: dict[str, Any]) -> list[dict[str, Any]]:
    non_tensors = data.non_tensor_batch
    if "traj_uid" not in non_tensors or "step_id" not in non_tensors:
        return []

    batch_size = len(_as_numpy(non_tensors["traj_uid"]))
    records = []
    for i in range(batch_size):
        record = {
            "index": _as_numpy(non_tensors["index"])[i] if "index" in non_tensors else None,
            "uid": _as_numpy(non_tensors["uid"])[i] if "uid" in non_tensors else None,
            "traj_uid": _as_numpy(non_tensors["traj_uid"])[i],
            "step_id": _safe_int(_as_numpy(non_tensors["step_id"])[i]),
            "env_id": _as_numpy(non_tensors["env_id"])[i] if "env_id" in non_tensors else None,
            "task_id": _as_numpy(non_tensors["task_id"])[i] if "task_id" in non_tensors else None,
            "task_description": _as_numpy(non_tensors["task_description"])[i]
            if "task_description" in non_tensors
            else None,
            "anchor_obs": _as_numpy(non_tensors["anchor_obs"])[i] if "anchor_obs" in non_tensors else None,
            "thought": _as_numpy(non_tensors["thought"])[i] if "thought" in non_tensors else None,
            "raw_action": _as_numpy(non_tensors["raw_action"])[i] if "raw_action" in non_tensors else None,
            "normalized_action": _as_numpy(non_tensors["normalized_action"])[i]
            if "normalized_action" in non_tensors
            else None,
            "reward": _safe_float(_as_numpy(non_tensors["rewards"])[i]) if "rewards" in non_tensors else 0.0,
            "final_reward": _safe_float(_as_numpy(non_tensors["final_reward"])[i])
            if "final_reward" in non_tensors
            else None,
            "is_action_valid": _safe_int(_as_numpy(non_tensors["is_action_valid"])[i])
            if "is_action_valid" in non_tensors
            else 0,
            "action_parse_error": _as_numpy(non_tensors["action_parse_error"])[i]
            if "action_parse_error" in non_tensors
            else None,
            "action_exec_error": _as_numpy(non_tensors["action_exec_error"])[i]
            if "action_exec_error" in non_tensors
            else None,
        }
        if debug_config.get("include_prompt", False) and "prompt_messages" in non_tensors:
            record["prompt_messages"] = _as_numpy(non_tensors["prompt_messages"])[i]
        if debug_config.get("include_response", True) and "response_text" in non_tensors:
            record["response_text"] = _as_numpy(non_tensors["response_text"])[i]
        if debug_config.get("save_screenshots", False) and "screenshot" in non_tensors:
            record["screenshot"] = _as_numpy(non_tensors["screenshot"])[i]
        records.append(record)
    return records


def _group_records(records: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped = OrderedDict()
    for record in sorted(records, key=lambda item: (str(item["traj_uid"]), int(item["step_id"]))):
        grouped.setdefault(str(record["traj_uid"]), []).append(record)
    return grouped


def _debug_output_dir(debug_config: dict[str, Any], global_step: int | None) -> str:
    output_dir = str(debug_config["output_dir"])
    step_dir = f"global_step={global_step}" if global_step is not None else "global_step=unknown"
    return os.path.join(output_dir, step_dir)


def _write_screenshots(records: list[dict[str, Any]], debug_config: dict[str, Any], global_step: int | None) -> None:
    output_dir = _debug_output_dir(debug_config, global_step)
    screenshot_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    for record in records:
        screenshot = record.pop("screenshot", None)
        if not isinstance(screenshot, Image.Image):
            continue
        traj_uid = str(record["traj_uid"])
        step_id = _safe_int(record["step_id"])
        filename = f"trajectory_{traj_uid}_step_{step_id:04d}.png"
        path = os.path.join(screenshot_dir, filename)
        screenshot.save(path)
        record["screenshot_path"] = os.path.relpath(path, output_dir)


def _write_jsonl(records: list[dict[str, Any]], debug_config: dict[str, Any], global_step: int | None) -> None:
    output_dir = _debug_output_dir(debug_config, global_step)
    os.makedirs(output_dir, exist_ok=True)
    for record in records:
        traj_uid = str(record["traj_uid"])
        path = os.path.join(output_dir, f"trajectory_{traj_uid}.jsonl")
        with open(path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")


def log_scalewob_batch_debug(data: DataProto, config: Any, global_step: int | None = None) -> None:
    debug_config = resolve_scalewob_debug_config(config)
    if not should_log_scalewob_debug(debug_config, global_step):
        return

    records = _batch_records(data, debug_config)
    if not records:
        return

    grouped = _group_records(records)
    traj_uids = {str(record["traj_uid"]) for record in records}
    final_rewards = []
    for traj_records in grouped.values():
        final_reward = traj_records[-1]["final_reward"]
        if final_reward is not None:
            final_rewards.append(_safe_float(final_reward))
    invalid_count = sum(1 for record in records if not record["is_action_valid"])
    parse_error_count = sum(1 for record in records if record["action_parse_error"])
    exec_error_count = sum(1 for record in records if record["action_exec_error"])
    finish_count = sum(
        1
        for record in records
        if isinstance(record["normalized_action"], dict) and record["normalized_action"].get("action") == "finish"
    )

    logger.warning(
        "ScaleWOB debug summary step=%s trajectories=%d flattened_steps=%d final_reward_mean=%.4f "
        "avg_steps_per_trajectory=%.2f invalid_action_rate=%.4f parse_errors=%d exec_errors=%d finish_actions=%d",
        global_step,
        len(traj_uids),
        len(records),
        float(np.mean(final_rewards)) if final_rewards else 0.0,
        len(records) / max(1, len(traj_uids)),
        invalid_count / max(1, len(records)),
        parse_error_count,
        exec_error_count,
        finish_count,
    )

    sampled_records = []
    for traj_uid, traj_records in list(grouped.items())[: debug_config["max_trajectories"]]:
        logger.warning("ScaleWOB debug trajectory traj_uid=%s steps=%d", traj_uid, len(traj_records))
        for record in traj_records[: debug_config["max_steps_per_trajectory"]]:
            sampled_records.append(record)
            logger.warning(
                "ScaleWOB debug step traj_uid=%s step_id=%s reward=%.4f final_reward=%s valid=%s "
                "thought=%r raw_action=%r normalized_action=%r parse_error=%r exec_error=%r",
                traj_uid,
                record["step_id"],
                record["reward"],
                record["final_reward"],
                record["is_action_valid"],
                record["thought"],
                record["raw_action"],
                record["normalized_action"],
                record["action_parse_error"],
                record["action_exec_error"],
            )

    if debug_config.get("save_screenshots", False):
        try:
            _write_screenshots(sampled_records, debug_config, global_step)
        except OSError as exc:
            logger.warning("Failed to write ScaleWOB debug screenshot artifacts: %s", exc)

    if debug_config.get("save_jsonl", False):
        try:
            _write_jsonl(sampled_records, debug_config, global_step)
        except OSError as exc:
            logger.warning("Failed to write ScaleWOB debug JSONL artifacts: %s", exc)


def compute_gigpo_debug_metrics(data: DataProto) -> dict[str, float]:
    metrics = {}
    non_tensors = data.non_tensor_batch
    if "advantages" not in data.batch.keys() or "response_mask" not in data.batch.keys():
        return metrics

    response_mask = data.batch["response_mask"].bool()
    active_advantages = data.batch["advantages"][response_mask]
    if active_advantages.numel() > 0:
        metrics.update(
            {
                "scalewob_debug/gigpo/advantage_mean": float(active_advantages.mean().item()),
                "scalewob_debug/gigpo/advantage_std": float(active_advantages.std(unbiased=False).item()),
                "scalewob_debug/gigpo/advantage_min": float(active_advantages.min().item()),
                "scalewob_debug/gigpo/advantage_max": float(active_advantages.max().item()),
            }
        )

    if "rewards" in non_tensors:
        rewards = np.asarray(non_tensors["rewards"], dtype=np.float32)
        metrics["scalewob_debug/gigpo/reward_mean"] = float(np.mean(rewards)) if rewards.size else 0.0
        metrics["scalewob_debug/gigpo/reward_std"] = float(np.std(rewards)) if rewards.size else 0.0
    if "is_action_valid" in non_tensors:
        valid = np.asarray(non_tensors["is_action_valid"], dtype=np.float32)
        invalid = 1.0 - valid
        metrics["scalewob_debug/gigpo/invalid_action_count"] = float(np.sum(invalid))
        metrics["scalewob_debug/gigpo/invalid_action_rate"] = float(np.mean(invalid)) if invalid.size else 0.0
    if "active_masks" in non_tensors:
        active = np.asarray(non_tensors["active_masks"], dtype=np.float32)
        metrics["scalewob_debug/gigpo/active_row_count"] = float(np.sum(active))
    if "index" in non_tensors:
        metrics["scalewob_debug/gigpo/unique_index_count"] = float(len(set(map(str, non_tensors["index"]))))
    if "traj_uid" in non_tensors:
        metrics["scalewob_debug/gigpo/unique_traj_uid_count"] = float(len(set(map(str, non_tensors["traj_uid"]))))
    if "index" in non_tensors and "anchor_obs" in non_tensors:
        groups = set(zip(map(str, non_tensors["index"]), map(str, non_tensors["anchor_obs"]), strict=True))
        metrics["scalewob_debug/gigpo/unique_step_group_count"] = float(len(groups))
    return metrics
