# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import csv
from pathlib import Path

import pandas as pd


def build_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for i, row in enumerate(reader):
            env_id = row.get("env_id") or row.get("environment_id") or row.get("env") or ""
            task_id = row.get("task_id") or row.get("id") or str(i)
            description = row.get("description") or row.get("task") or row.get("instruction") or ""
            split = row.get("split")
            rows.append(
                {
                    "data_source": "scalewob",
                    "prompt": [{"role": "user", "content": "ScaleWoB browser task."}],
                    "extra_info": {
                        "index": i,
                        "scalewob": {
                            "env_id": env_id,
                            "task_id": task_id,
                            "description": description,
                        },
                        **({"split": split} if split else {}),
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a ScaleWoB task CSV to VERL parquet format.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(build_rows(args.csv)).to_parquet(args.output)


if __name__ == "__main__":
    main()
