# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image


@dataclass
class ScaleWoBBrowserConfig:
    base_url: str = "https://niumascript.com/scalewob-env"
    platform: str = "mobile"
    headless: bool = True
    screenshot_quality: str = "low"
    target_image_hw: tuple[int, int] = (1024, 474)
    coord_scale: int = 1000
    reset_retries: int = 3
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

    def reset(self, scalewob_info: dict[str, Any]) -> dict[str, Any]:
        env_id = scalewob_info.get("env_id") or scalewob_info.get("environment_id") or scalewob_info.get("env")
        self._env_id = str(env_id) if env_id is not None and env_id != "" else self._env_id
        if not self._env_id:
            raise ValueError("ScaleWoB reset requires scalewob_info.env_id")
        self._task_id = scalewob_info.get("task_id", 0)
        automation = self._ensure_automation()
        last_error = None
        for _ in range(self.config.reset_retries):
            try:
                automation.start()
                automation.start_evaluation()
                return {"env_id": self._env_id, "task_id": self._task_id}
            except Exception as exc:
                last_error = exc
                self.close()
        raise RuntimeError(f"ScaleWoB reset failed after {self.config.reset_retries} retries") from last_error

    def screenshot(self) -> Image.Image:
        automation = self._ensure_automation()
        raw = automation.take_screenshot(format="pil")
        if isinstance(raw, Image.Image):
            return raw.convert("RGB")
        if isinstance(raw, bytes):
            return Image.open(BytesIO(raw)).convert("RGB")
        raise TypeError(f"Unsupported screenshot type: {type(raw)}")

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        automation = self._ensure_automation()
        action_name = str(action.get("action", "")).strip().lower()

        def invalid_step(error: str) -> dict[str, Any]:
            return {
                "observation": None,
                "reward": 0.0,
                "done": False,
                "info": {"error": error, "invalid_action": True},
            }

        if action_name in {"tap", "click"}:
            try:
                automation.click(int(action.get("x", 0)), int(action.get("y", 0)))
                return {"observation": None, "reward": 0.0, "done": False, "info": {}}
            except Exception as exc:
                return invalid_step(str(exc))
        if action_name in {"input_text", "type", "text", "input"}:
            try:
                automation.type(str(action.get("text", "")))
                return {"observation": None, "reward": 0.0, "done": False, "info": {}}
            except Exception as exc:
                return invalid_step(str(exc))
        if action_name in {"swipe", "drag", "scroll"}:
            try:
                automation.drag(
                    int(action.get("x1", action.get("start_x", 0))),
                    int(action.get("y1", action.get("start_y", 0))),
                    int(action.get("x2", action.get("end_x", 0))),
                    int(action.get("y2", action.get("end_y", 0))),
                )
                return {"observation": None, "reward": 0.0, "done": False, "info": {}}
            except Exception as exc:
                return invalid_step(str(exc))
        if action_name in {"wait", "pause"}:
            return {"observation": None, "reward": 0.0, "done": False, "info": {}}
        if action_name == "finish":
            try:
                result = automation.finish_evaluation(task_id=self._task_id)
                reward = 1.0 if result.get("success") else float(result.get("reward", 0.0) or 0.0)
                return {"observation": None, "reward": reward, "done": True, "info": result}
            except Exception as exc:
                return invalid_step(str(exc))
        return invalid_step(f"unsupported_action: {action_name}")

    def close(self) -> None:
        if self._automation is not None:
            close = getattr(self._automation, "close", None)
            if close is not None:
                close()
            self._automation = None
