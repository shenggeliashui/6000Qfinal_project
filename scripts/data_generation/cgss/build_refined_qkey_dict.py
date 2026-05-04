#!/usr/bin/env python3
"""
CGSS：从 Stata .dta 元数据生成与 SubPOP-eval 兼容的 refined_qkey_dict.json。

与原文 refine_question.py / GSS refined 流程对应，但独立脚本，不修改 SubPOP 原有脚本。

用法（在仓库根目录 subpop/ 下，模仿 README 中 SubPOP-Train / SubPOP-Eval）::

    # 训练集题项（默认输出 data/cgss-train/processed/refined_qkey_dict.json）
    python scripts/data_generation/cgss/build_refined_qkey_dict.py --split train --out_dataset cgss-train

    # 测试/评估集题项
    python scripts/data_generation/cgss/build_refined_qkey_dict.py --split eval --out_dataset cgss-eval

    # 合并 train+eval 题项到单一 refined（探索用）
    python scripts/data_generation/cgss/build_refined_qkey_dict.py --split all --out_dataset cgss2023

配置见 data/cgss2023/metadata/cgss_config.json：须含 questions_train、questions_eval（旧版仅 questions 仍支持）。
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import pyreadstat

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def questions_for_split(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    """Select question entries for train / eval / all (train+eval)."""
    qt = cfg.get("questions_train", [])
    qe = cfg.get("questions_eval", [])
    legacy = cfg.get("questions", [])
    if split == "all":
        merged = list(qt) + list(qe)
        return merged if merged else list(legacy)
    if split == "train":
        return list(qt) if qt else list(legacy)
    if split == "eval":
        return list(qe)
    raise ValueError(f"Unknown --split {split!r}, expected train|eval|all")


def _norm_code(k: Any) -> float:
    if k is None or (isinstance(k, float) and math.isnan(k)):
        return float("nan")
    try:
        return float(k)
    except (TypeError, ValueError):
        return float("nan")


def _build_option_ordinal(
    value_labels: Dict[Any, str],
    global_invalid: List[float],
    global_refusal: List[float],
    q_invalid: List[float],
    q_refusal: Optional[List[float]],
) -> Tuple[List[str], List[float], Dict[float, int]]:
    refusal_set = set(float(x) for x in (q_refusal if q_refusal is not None else global_refusal))
    invalid_set = set(float(x) for x in global_invalid) | set(float(x) for x in q_invalid)

    pairs: List[Tuple[float, str]] = []
    for raw_k, lab in value_labels.items():
        fk = _norm_code(raw_k)
        if fk != fk:  # nan
            continue
        if fk in invalid_set or fk in refusal_set:
            continue
        pairs.append((fk, str(lab).strip()))

    pairs.sort(key=lambda x: x[0])
    option_list = [p[1] for p in pairs]
    ordinal = [float(i + 1) for i in range(len(option_list))]
    code_to_idx = {p[0]: i for i, p in enumerate(pairs)}
    return option_list, ordinal, code_to_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="CGSS: build refined_qkey_dict.json from .dta meta")
    parser.add_argument(
        "--config",
        type=str,
        default="data/cgss2023/metadata/cgss_config.json",
        help="Path to cgss_config.json (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "eval", "all"],
        help="train / eval 对应 README 中 SubPOP-Train 与 SubPOP-Eval 的题项划分；all=合并",
    )
    parser.add_argument(
        "--out_dataset",
        type=str,
        default="cgss-train",
        help="输出目录名 data/{out_dataset}/processed/（如 cgss-train、cgss-eval）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="auto",
        help="refined_qkey_dict.json 路径；auto 则写入 data/{out_dataset}/processed/refined_qkey_dict.json",
    )
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path

    if args.output == "auto":
        out_path = REPO_ROOT / "data" / args.out_dataset / "processed" / "refined_qkey_dict.json"
    else:
        out_path = pathlib.Path(args.output)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    micro = pathlib.Path(cfg["microdata_path"])
    if not micro.is_absolute():
        micro = REPO_ROOT / micro
    if not micro.exists():
        print(f"ERROR: microdata not found: {micro}", file=sys.stderr)
        sys.exit(1)

    _, meta = pyreadstat.read_dta(str(micro), metadataonly=True)
    labels_map = meta.variable_value_labels or {}

    global_invalid = [float(x) for x in cfg.get("global_invalid_codes", [])]
    global_refusal = [float(x) for x in cfg.get("global_refusal_codes", [])]

    refined: Dict[str, Dict[str, Any]] = {}
    qlist = questions_for_split(cfg, args.split)
    if not qlist:
        print(f"ERROR: no questions for --split {args.split!r} (check questions_train / questions_eval).", file=sys.stderr)
        sys.exit(1)
    for q in qlist:
        var = q["var"]
        if str(var).startswith("__REPLACE"):
            print(f"SKIP placeholder question var: {var}", file=sys.stderr)
            continue
        vlabels = labels_map.get(var)
        if not vlabels:
            print(f"WARN: no value labels for variable {var}, skip", file=sys.stderr)
            continue
        q_invalid = [float(x) for x in q.get("invalid_codes", [])]
        q_refusal = q.get("refusal_codes", None)
        if q_refusal is not None:
            q_refusal = [float(x) for x in q_refusal]

        option_list, ordinal, _ = _build_option_ordinal(
            vlabels, global_invalid, global_refusal, q_invalid, q_refusal
        )
        if not option_list:
            print(f"WARN: no valid options after filtering for {var}, skip", file=sys.stderr)
            continue

        qtext = q.get("question_text")
        if qtext:
            body = str(qtext).strip()
        else:
            lab = (meta.column_names_to_labels or {}).get(var)
            body = str(lab).strip() if lab else var

        refined[var] = {
            "refined_qbody": body,
            "option_list": option_list,
            "ordinal": ordinal,
        }

    if not refined:
        print("ERROR: refined_qkey_dict is empty. Check cgss_config.json questions and .dta paths.", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} ({len(refined)} questions)")


if __name__ == "__main__":
    main()
