# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Any

from PIL import Image

SYSTEM_PROMPT = (
    "You are controlling a mobile browser. Respond with exactly one JSON object. "
    "Allowed actions are tap, input_text, swipe, wait, and finish. Coordinates use a 0-1000 normalized screen scale."
)


def _format_history(action_history: list[dict[str, Any]], max_items: int) -> str:
    if not action_history:
        return "none"
    recent = action_history[-max_items:]
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(recent))


def build_prompt_messages(
    *,
    task_description: str,
    screenshot: Image.Image,
    action_history: list[dict[str, Any]],
    action_history_len: int,
) -> list[dict[str, Any]]:
    """Build one-turn multimodal prompt with exactly one current screenshot."""
    text = (
        f"Task: {task_description}\n"
        f"Recent actions:\n{_format_history(action_history, action_history_len)}\n"
        "Return one JSON action for the current screenshot."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "image": screenshot},
            ],
        },
    ]
