# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
from collections.abc import Sequence

from PIL import Image


def resize_screenshot(image: Image.Image, target_hw: Sequence[int] | None) -> Image.Image:
    if not target_hw:
        return image
    height, width = int(target_hw[0]), int(target_hw[1])
    return image.resize((width, height), Image.Resampling.BILINEAR)


def screenshot_hash(image: Image.Image, hash_hw: Sequence[int] = (64, 64)) -> str:
    height, width = int(hash_hw[0]), int(hash_hw[1])
    compact = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return hashlib.sha256(compact.tobytes()).hexdigest()
