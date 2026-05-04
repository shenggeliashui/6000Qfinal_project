#!/usr/bin/env python3
"""
CGSS：调用与原文相同的 prepare_data 逻辑生成 opnqa_*_{train,val,test}.csv。

不修改 prepare_finetuning_data.py 的主流程语义；通过 subgroup_coder=identity
使「人口学子群体」标签与中文 CSV 中 group 列一致。

用法（在仓库根目录 subpop/ 下，先完成 refined + distribution CSV）::

    python scripts/data_generation/cgss/run_cgss_prepare_finetuning.py \\
        --train_ratio 0.9 --val_ratio 0.1 --test_ratio 0.0

或直接使用原文脚本::

    python scripts/data_generation/prepare_finetuning_data.py \\
        --dataset cgss2023 \\
        --steer_prompts_file_path data/cgss2023/metadata/steering_prompts_zh.json \\
        --steer_demographics_file_path data/cgss2023/metadata/demographics_cgss.csv \\
        --subgroup_coder identity
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subpop.survey.config import SteeringPromptType
from subpop.utils.random_utils import set_random_seed

_PREP = REPO_ROOT / "scripts/data_generation/prepare_finetuning_data.py"
_spec = importlib.util.spec_from_file_location("prepare_finetuning_data", _PREP)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
prepare_data = _mod.prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="CGSS: run prepare_data for finetuning CSVs")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--survey_csv",
        type=str,
        default="data/cgss2023/processed/cgss2023.csv",
    )
    parser.add_argument(
        "--steer_prompts",
        type=str,
        default="data/cgss2023/metadata/steering_prompts_zh.json",
    )
    parser.add_argument(
        "--steer_demographics",
        type=str,
        default="data/cgss2023/metadata/demographics_cgss.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cgss2023/processed",
    )
    parser.add_argument("--no_shuffle", action="store_true")
    args = parser.parse_args()

    survey_path = pathlib.Path(args.survey_csv)
    if not survey_path.is_absolute():
        survey_path = REPO_ROOT / survey_path
    steer_p = pathlib.Path(args.steer_prompts)
    if not steer_p.is_absolute():
        steer_p = REPO_ROOT / steer_p
    steer_d = pathlib.Path(args.steer_demographics)
    if not steer_d.is_absolute():
        steer_d = REPO_ROOT / steer_d
    out_dir = pathlib.Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    if not survey_path.exists():
        print(f"ERROR: survey CSV not found: {survey_path}", file=sys.stderr)
        sys.exit(1)
    if not steer_p.exists() or not steer_d.exists():
        print("ERROR: steering JSON or demographics CSV missing.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    identity = lambda _a, s: s

    for steer_type in SteeringPromptType:
        set_random_seed(args.seed)
        train_df, val_df, test_df = prepare_data(
            survey_file_path=survey_path,
            steering_prompts_file_path=steer_p,
            steering_demographics_file_path=steer_d,
            steering_prompt_type=steer_type,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            test_wave=[],
            subgroup_coder=identity,
        )
        if not args.no_shuffle:
            train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            val_df = val_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        train_df.to_csv(out_dir / f"opnqa_{steer_type.name}_train.csv", index=False)
        val_df.to_csv(out_dir / f"opnqa_{steer_type.name}_val.csv", index=False)
        test_df.to_csv(out_dir / f"opnqa_{steer_type.name}_test.csv", index=False)
        print(f"Wrote opnqa_{steer_type.name}_* under {out_dir}")


if __name__ == "__main__":
    main()
