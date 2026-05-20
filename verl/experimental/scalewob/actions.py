# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
import re
from dataclasses import dataclass
from typing import Any

ACTION_ALIASES = {
    "click": "tap",
    "press": "tap",
    "type": "input_text",
    "text": "input_text",
    "input": "input_text",
    "scroll": "swipe",
    "drag": "swipe",
    "pause": "wait",
}
VALID_ACTIONS = {"tap", "input_text", "swipe", "wait", "finish"}


@dataclass(frozen=True)
class ParsedAction:
    action: dict[str, Any]
    is_action_valid: int
    error: str | None = None


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else stripped


def _fallback_wait(error: str) -> ParsedAction:
    return ParsedAction(action={"action": "wait"}, is_action_valid=0, error=error)


def _clamp_coord(value: Any, coord_scale: int) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return int(max(0, min(coord_scale, round(number))))


def parse_action(text: str, coord_scale: int = 1000) -> ParsedAction:
    """Parse a model JSON action and project aliases/coordinates into ScaleWoB's action space."""
    try:
        payload = json.loads(_strip_markdown_fence(text))
    except json.JSONDecodeError as exc:
        return _fallback_wait(f"invalid_json: {exc}")

    if not isinstance(payload, dict):
        return _fallback_wait("action_payload_not_object")

    raw_name = payload.get("action") or payload.get("type") or payload.get("name")
    if not isinstance(raw_name, str):
        return _fallback_wait("missing_action_name")
    action_name = ACTION_ALIASES.get(raw_name.strip().lower(), raw_name.strip().lower())
    if action_name not in VALID_ACTIONS:
        return _fallback_wait(f"unsupported_action: {raw_name}")

    if action_name == "tap":
        projected = {
            "action": "tap",
            "x": _clamp_coord(payload.get("x"), coord_scale),
            "y": _clamp_coord(payload.get("y"), coord_scale),
        }
    elif action_name == "swipe":
        projected = {
            "action": "swipe",
            "x1": _clamp_coord(payload.get("x1", payload.get("start_x")), coord_scale),
            "y1": _clamp_coord(payload.get("y1", payload.get("start_y")), coord_scale),
            "x2": _clamp_coord(payload.get("x2", payload.get("end_x")), coord_scale),
            "y2": _clamp_coord(payload.get("y2", payload.get("end_y")), coord_scale),
        }
    elif action_name == "input_text":
        projected = {"action": "input_text", "text": str(payload.get("text", ""))}
    else:
        projected = {"action": action_name}

    return ParsedAction(action=projected, is_action_valid=1)
