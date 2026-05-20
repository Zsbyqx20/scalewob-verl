# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

ACTION_ALIASES = {
    "click": "tap",
    "press": "tap",
    "long_click": "long_press",
    "long_press": "long_press",
    "type": "input_text",
    "text": "input_text",
    "input": "input_text",
    "enter": "press_enter",
    "press_enter": "press_enter",
    "scroll": "swipe",
    "drag": "swipe",
    "pause": "wait",
}
VALID_ACTIONS = {"tap", "long_press", "input_text", "press_enter", "swipe", "wait", "finish"}
SUPPORTED_DEVICE_METHODS = {"click", "long_click", "type", "enter", "swipe", "wait", "end_task"}


@dataclass(frozen=True)
class ParsedAction:
    action: dict[str, Any]
    is_action_valid: int
    error: str | None = None
    thought: str | None = None
    raw_action: str | None = None


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else stripped


def _fallback_wait(error: str, *, thought: str | None = None, raw_action: str | None = None) -> ParsedAction:
    return ParsedAction(
        action={"action": "wait"},
        is_action_valid=0,
        error=error,
        thought=thought,
        raw_action=raw_action,
    )


def _clamp_coord(value: Any, coord_scale: int) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return int(max(0, min(coord_scale, round(number))))


def _extract_thought(text: str) -> str | None:
    match = re.search(r"Thought:\s*(.*?)(?=\n\s*(?:Action:|```)|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    thought = match.group(1).strip()
    return thought or None


def _extract_sft_action_code(text: str) -> str | None:
    inline = re.search(r"^\s*Action:\s*`?([^`\n]+)`?\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if inline:
        return inline.group(1).strip()

    fenced = re.search(r"^\s*Action:\s*```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if fenced:
        return fenced.group(1).strip()

    fenced_after_action = re.search(
        r"^\s*Action:\s*\n\s*```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    if fenced_after_action:
        return fenced_after_action.group(1).strip()
    return None


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        raise ValueError("action_argument_not_literal") from None


def _coord_pair(node: ast.AST) -> tuple[Any, Any]:
    value = _literal(node)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("coordinate_pair_required")
    return value[0], value[1]


def _optional_finish_status(call: ast.Call) -> str | None:
    if not call.args:
        return None
    status = _literal(call.args[0])
    if status is None:
        return None
    return str(status)


def _optional_finish_params(call: ast.Call) -> dict[str, Any] | None:
    if len(call.args) < 2:
        return None
    params = _literal(call.args[1])
    if params is None:
        return None
    if not isinstance(params, dict):
        raise ValueError("device_end_task_params_must_be_object")
    return params


def _parse_device_action(code: str, coord_scale: int) -> dict[str, Any]:
    try:
        parsed = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"invalid_device_action_syntax: {exc}") from None

    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Expr):
        raise ValueError("device_action_must_be_single_expression")

    call = parsed.body[0].value
    if not isinstance(call, ast.Call):
        raise ValueError("device_action_not_call")
    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "device"
        and isinstance(func.attr, str)
    ):
        raise ValueError("device_action_must_call_device_method")

    method = func.attr
    if method not in SUPPORTED_DEVICE_METHODS:
        raise ValueError(f"unsupported_device_method: {method}")
    if call.keywords:
        raise ValueError("device_action_keywords_not_supported")

    if method == "click":
        if len(call.args) != 2:
            raise ValueError("device_click_requires_x_y")
        return {
            "action": "tap",
            "x": _clamp_coord(_literal(call.args[0]), coord_scale),
            "y": _clamp_coord(_literal(call.args[1]), coord_scale),
        }
    if method == "long_click":
        if len(call.args) != 2:
            raise ValueError("device_long_click_requires_x_y")
        return {
            "action": "long_press",
            "x": _clamp_coord(_literal(call.args[0]), coord_scale),
            "y": _clamp_coord(_literal(call.args[1]), coord_scale),
        }
    if method == "type":
        if len(call.args) != 1:
            raise ValueError("device_type_requires_content")
        return {"action": "input_text", "text": str(_literal(call.args[0]))}
    if method == "enter":
        if call.args:
            raise ValueError("device_enter_takes_no_arguments")
        return {"action": "press_enter"}
    if method == "swipe":
        if len(call.args) != 2:
            raise ValueError("device_swipe_requires_start_end")
        x1, y1 = _coord_pair(call.args[0])
        x2, y2 = _coord_pair(call.args[1])
        return {
            "action": "swipe",
            "x1": _clamp_coord(x1, coord_scale),
            "y1": _clamp_coord(y1, coord_scale),
            "x2": _clamp_coord(x2, coord_scale),
            "y2": _clamp_coord(y2, coord_scale),
        }
    if method == "wait":
        if call.args:
            raise ValueError("device_wait_takes_no_arguments")
        return {"action": "wait"}
    if method == "end_task":
        if len(call.args) > 2:
            raise ValueError("device_end_task_takes_status_and_optional_params")
        action = {"action": "finish"}
        status = _optional_finish_status(call)
        params = _optional_finish_params(call)
        if status is not None:
            action["status"] = status
        if params is not None:
            action["params"] = params
        return action

    raise ValueError(f"unsupported_device_method: {method}")


def _parse_json_action(text: str, coord_scale: int) -> ParsedAction:
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
    elif action_name == "long_press":
        projected = {
            "action": "long_press",
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
    elif action_name == "finish":
        projected = {"action": "finish"}
        if "status" in payload:
            projected["status"] = str(payload.get("status"))
        params = payload.get("params")
        if isinstance(params, dict):
            projected["params"] = params
    else:
        projected = {"action": action_name}

    return ParsedAction(action=projected, is_action_valid=1, raw_action=json.dumps(payload, ensure_ascii=False))


def parse_action(text: str, coord_scale: int = 1000) -> ParsedAction:
    """Parse SFT device-use output, with legacy JSON actions as a compatibility fallback."""
    thought = _extract_thought(text)
    raw_action = _extract_sft_action_code(text)
    if raw_action is not None:
        try:
            return ParsedAction(
                action=_parse_device_action(raw_action, coord_scale),
                is_action_valid=1,
                thought=thought,
                raw_action=raw_action,
            )
        except ValueError as exc:
            return _fallback_wait(str(exc), thought=thought, raw_action=raw_action)

    if re.search(r"\b(?:Thought|Action):|device\.", text, flags=re.IGNORECASE):
        return _fallback_wait("missing_device_action", thought=thought)

    parsed = _parse_json_action(text, coord_scale)
    if parsed.thought is None and thought is not None:
        return ParsedAction(
            action=parsed.action,
            is_action_valid=parsed.is_action_valid,
            error=parsed.error,
            thought=thought,
            raw_action=parsed.raw_action,
        )
    return parsed
