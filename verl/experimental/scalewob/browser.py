# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image


class ScaleWoBBrowserOperationTimeoutError(TimeoutError):
    """Raised when a ScaleWoB browser operation exceeds the configured timeout."""


@dataclass
class ScaleWoBBrowserConfig:
    base_url: str = "https://niumascript.com/scalewob-env"
    platform: str = "mobile"
    headless: bool = True
    screenshot_quality: str = "low"
    target_image_hw: tuple[int, int] = (1024, 474)
    coord_space: str = "normalized"
    coord_scale: int = 1000
    post_action_wait_seconds: float = 0.3
    wait_action_seconds: float = 1.0
    max_stale_steps: int = 3
    stale_action_penalty: bool = True
    browser_operation_timeout_seconds: float = 10.0
    reset_retries: int = 3
    reset_retry_delay_seconds: float = 1.0
    max_env_steps: int = 10
    action_history_len: int = 4
    anchor_hash_hw: tuple[int, int] = (64, 64)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "ScaleWoBBrowserConfig":
        if not mapping:
            return cls()
        kwargs = dict(mapping)
        for key in ("target_image_hw", "anchor_hash_hw"):
            if key in kwargs and isinstance(kwargs[key], list):
                kwargs[key] = tuple(kwargs[key])
        return cls(**{k: v for k, v in kwargs.items() if k in cls.__dataclass_fields__})


class ScaleWoBBrowser:
    """Thin wrapper around scalewob.automation.ScaleWoBAutomation."""

    def __init__(self, config: ScaleWoBBrowserConfig):
        self.config = config
        self._automation = None
        self._env_id: str | None = None
        self._task_id: Any = 0
        self._canonical_task_id: Any = None
        self.last_screenshot_size: tuple[int, int] | None = None

    def _call_with_timeout(self, fn, *args, **kwargs):
        timeout = float(self.config.browser_operation_timeout_seconds)
        if timeout <= 0:
            return fn(*args, **kwargs)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise ScaleWoBBrowserOperationTimeoutError(
                f"{getattr(fn, '__name__', type(fn).__name__)} timed out after {timeout:g}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _post_action_wait(self) -> None:
        delay = float(self.config.post_action_wait_seconds)
        if delay > 0:
            time.sleep(delay)

    def _normalized_to_pixel(self, value: Any, limit: int) -> int:
        if self.config.coord_space != "normalized":
            raise ValueError(f"unsupported_coord_space: {self.config.coord_space}")
        scale = float(self.config.coord_scale)
        if scale <= 0:
            raise ValueError("coord_scale_must_be_positive")
        pixel = round(float(value) / scale * (limit - 1))
        return int(max(0, min(limit - 1, pixel)))

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        executed = dict(action)
        action_name = str(action.get("action", "")).strip().lower()
        if action_name in {"tap", "click", "long_press", "long_click"}:
            if self.last_screenshot_size is None:
                raise RuntimeError("coordinate action requires a prior screenshot")
            width, height = self.last_screenshot_size
            executed["x"] = self._normalized_to_pixel(action.get("x", 0), width)
            executed["y"] = self._normalized_to_pixel(action.get("y", 0), height)
        elif action_name in {"swipe", "drag", "scroll"}:
            if self.last_screenshot_size is None:
                raise RuntimeError("coordinate action requires a prior screenshot")
            width, height = self.last_screenshot_size
            executed["x1"] = self._normalized_to_pixel(action.get("x1", action.get("start_x", 0)), width)
            executed["y1"] = self._normalized_to_pixel(action.get("y1", action.get("start_y", 0)), height)
            executed["x2"] = self._normalized_to_pixel(action.get("x2", action.get("end_x", 0)), width)
            executed["y2"] = self._normalized_to_pixel(action.get("y2", action.get("end_y", 0)), height)
        return executed

    def _ensure_automation(self):
        if self._automation is not None:
            return self._automation
        if not self._env_id:
            raise ValueError("ScaleWoB env_id is not set")
        try:
            from scalewob.automation import ScaleWoBAutomation
        except ImportError as exc:
            raise ImportError("ScaleWoB requires the `scalewob` package to be installed") from exc
        self._automation = ScaleWoBAutomation(
            env_id=self._env_id,
            base_url=self.config.base_url,
            platform=self.config.platform,
            headless=self.config.headless,
            screenshot_quality=self.config.screenshot_quality,
        )
        return self._automation

    def _match_task_id(self, candidate: Any, target: Any) -> bool:
        if candidate == target:
            return True
        if str(candidate) == str(target):
            return True
        try:
            return int(candidate) == int(target)
        except (TypeError, ValueError):
            return False

    def get_task_metadata(self, task_id: Any | None = None) -> dict[str, Any] | None:
        automation = self._ensure_automation()
        tasks = getattr(automation, "tasks", None)
        if not tasks:
            return None

        target_task_id = self._task_id if task_id is None else task_id
        for task in tasks:
            if self._match_task_id(task.get("task_id"), target_task_id):
                if task_id is None:
                    self._canonical_task_id = task.get("task_id")
                return dict(task)
        return None

    def reset(self, scalewob_info: dict[str, Any]) -> dict[str, Any]:
        env_id = scalewob_info.get("env_id") or scalewob_info.get("environment_id") or scalewob_info.get("env")
        self._env_id = str(env_id) if env_id is not None and env_id != "" else self._env_id
        if not self._env_id:
            raise ValueError("ScaleWoB reset requires scalewob_info.env_id")
        self._task_id = scalewob_info.get("task_id", 0)
        self._canonical_task_id = None
        self.last_screenshot_size = None
        last_error = None
        for attempt in range(self.config.reset_retries):
            try:
                automation = self._ensure_automation()
                self._call_with_timeout(automation.start)
                self._call_with_timeout(automation.start_evaluation)
                return {"env_id": self._env_id, "task_id": self._task_id}
            except Exception as exc:
                last_error = exc
                self.close()
                if attempt + 1 < self.config.reset_retries and self.config.reset_retry_delay_seconds > 0:
                    time.sleep(float(self.config.reset_retry_delay_seconds))
        raise RuntimeError(f"ScaleWoB reset failed after {self.config.reset_retries} retries") from last_error

    def screenshot(self) -> Image.Image:
        automation = self._ensure_automation()
        raw = self._call_with_timeout(automation.take_screenshot, format="pil")
        if isinstance(raw, Image.Image):
            image = raw.convert("RGB")
            self.last_screenshot_size = image.size
            return image
        if isinstance(raw, bytes):
            image = Image.open(BytesIO(raw)).convert("RGB")
            self.last_screenshot_size = image.size
            return image
        raise TypeError(f"Unsupported screenshot type: {type(raw)}")

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        automation = self._ensure_automation()
        normalized_action = dict(action)
        executed_action: dict[str, Any] | None = None

        def ok_step(*, reward: float = 0.0, done: bool = False, info: dict[str, Any] | None = None) -> dict[str, Any]:
            step_info = dict(info or {})
            step_info.setdefault("normalized_action", normalized_action)
            step_info.setdefault("executed_action", executed_action)
            return {
                "observation": None,
                "reward": reward,
                "done": done,
                "info": step_info,
            }

        def invalid_step(
            error: str,
            *,
            normalized_action: dict[str, Any] | None = None,
            executed_action: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return ok_step(
                info={
                    "error": error,
                    "invalid_action": True,
                    "normalized_action": normalized_action or dict(action),
                    "executed_action": executed_action,
                }
            )

        try:
            executed_action = self._execute_action(normalized_action)
        except Exception as exc:
            return invalid_step(str(exc), normalized_action=normalized_action, executed_action=None)
        action_name = str(executed_action.get("action", "")).strip().lower()

        if action_name in {"tap", "click"}:
            try:
                self._call_with_timeout(
                    automation.click,
                    int(executed_action.get("x", 0)),
                    int(executed_action.get("y", 0)),
                )
                self._post_action_wait()
                return ok_step()
            except Exception as exc:
                return invalid_step(str(exc), normalized_action=normalized_action, executed_action=executed_action)
        if action_name in {"long_press", "long_click"}:
            try:
                self._call_with_timeout(
                    automation.long_press,
                    int(executed_action.get("x", 0)),
                    int(executed_action.get("y", 0)),
                )
                self._post_action_wait()
                return ok_step()
            except Exception as exc:
                return invalid_step(str(exc), normalized_action=normalized_action, executed_action=executed_action)
        if action_name in {"input_text", "type", "text", "input"}:
            try:
                self._call_with_timeout(automation.type, str(executed_action.get("text", "")))
                self._post_action_wait()
                return ok_step()
            except Exception as exc:
                return invalid_step(str(exc), normalized_action=normalized_action, executed_action=executed_action)
        if action_name in {"press_enter", "enter"}:
            try:
                self._call_with_timeout(automation.press_enter)
                self._post_action_wait()
                return ok_step()
            except Exception as exc:
                return invalid_step(str(exc), normalized_action=normalized_action, executed_action=executed_action)
        if action_name in {"swipe", "drag", "scroll"}:
            try:
                self._call_with_timeout(
                    automation.drag,
                    int(executed_action.get("x1", executed_action.get("start_x", 0))),
                    int(executed_action.get("y1", executed_action.get("start_y", 0))),
                    int(executed_action.get("x2", executed_action.get("end_x", 0))),
                    int(executed_action.get("y2", executed_action.get("end_y", 0))),
                )
                self._post_action_wait()
                return ok_step()
            except Exception as exc:
                return invalid_step(str(exc), normalized_action=normalized_action, executed_action=executed_action)
        if action_name in {"wait", "pause"}:
            wait_seconds = float(self.config.wait_action_seconds)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            return ok_step()
        if action_name == "finish":
            try:
                params = executed_action.get("params")
                if params is not None and not isinstance(params, dict):
                    return invalid_step(
                        "finish_params_must_be_object",
                        normalized_action=normalized_action,
                        executed_action=executed_action,
                    )
                task_id = self._canonical_task_id if self._canonical_task_id is not None else self._task_id
                result = self._call_with_timeout(automation.finish_evaluation, task_id=task_id, params=params)
                reward = 1.0 if result.get("success") else float(result.get("reward", 0.0) or 0.0)
                return ok_step(reward=reward, done=True, info=result)
            except Exception as exc:
                return invalid_step(str(exc), normalized_action=normalized_action, executed_action=executed_action)
        return invalid_step(
            f"unsupported_action: {action_name}",
            normalized_action=normalized_action,
            executed_action=executed_action,
        )

    def close(self) -> None:
        if self._automation is not None:
            close = getattr(self._automation, "close", None)
            if close is not None:
                close()
            self._automation = None
