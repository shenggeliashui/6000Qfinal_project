#!/usr/bin/env python3
"""
CGSS：按子群体与抽样权重聚合题项分布，输出与 SubPOP generate_distribution 结果列一致的 CSV。

对应原文 scripts/data_generation/generate_distribution.py 中 GSS 分支的语义，
但独立实现，不修改原文脚本。

用法（在仓库根目录 subpop/ 下，与 build_refined 的 --split / --out_dataset 对齐）::

    python scripts/data_generation/cgss/generate_cgss_distribution.py --split train --out_dataset cgss-train
    python scripts/data_generation/cgss/generate_cgss_distribution.py --split eval --out_dataset cgss-eval
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pyreadstat

from subpop.utils.survey_utils import list_normalize


def questions_for_split(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
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
    raise ValueError(f"Unknown --split {split!r}")


def question_cfg_by_var(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for q in questions_for_split(cfg, "all"):
        out[q["var"]] = q
    return out


def _norm_code(k: Any) -> float:
    if k is None:
        return float("nan")
    try:
        if isinstance(k, float) and np.isnan(k):
            return float("nan")
    except TypeError:
        pass
    try:
        return float(k)
    except (TypeError, ValueError):
        return float("nan")


def _build_code_index(
    value_labels: Dict[Any, str],
    global_invalid: List[float],
    global_refusal: List[float],
    q_invalid: List[float],
    q_refusal: Optional[List[float]],
) -> Tuple[Dict[float, int], List[str], List[float], set]:
    refusal_set = set(float(x) for x in (q_refusal if q_refusal is not None else global_refusal))
    invalid_set = set(float(x) for x in global_invalid) | set(float(x) for x in q_invalid)

    pairs: List[Tuple[float, str]] = []
    for raw_k, lab in value_labels.items():
        fk = _norm_code(raw_k)
        if fk != fk:
            continue
        if fk in invalid_set or fk in refusal_set:
            continue
        pairs.append((fk, str(lab).strip()))
    pairs.sort(key=lambda x: x[0])
    option_list = [p[1] for p in pairs]
    ordinal = [float(i + 1) for i in range(len(option_list))]
    code_to_idx = {p[0]: i for i, p in enumerate(pairs)}
    return code_to_idx, option_list, ordinal, refusal_set


def generate_combined_pairs(
    loaded_pair: List[Tuple[str, str]], n_combination: int
) -> List[Union[Tuple[str, str], Tuple[List[str], List[str]]]]:
    attribute_to_groups: Dict[str, List[str]] = {}
    for attr, group in loaded_pair:
        attribute_to_groups.setdefault(attr, []).append(group)
    attribute_combinations = list(itertools.combinations(attribute_to_groups.keys(), n_combination))
    combined_pairs: List[Union[Tuple[str, str], Tuple[List[str], List[str]]]] = []
    for attributes in attribute_combinations:
        group_combinations = itertools.product(
            *[attribute_to_groups[attr] for attr in attributes]
        )
        for groups in group_combinations:
            combined_pairs.append((list(attributes), list(groups)))
    return combined_pairs


def _subpop_mask(df: pd.DataFrame, sub_specs: List[dict], attributes: Any, groups: Any) -> pd.Series:
    if isinstance(attributes, str):
        attributes = [attributes]
        groups = [groups]
    mask = pd.Series(True, index=df.index)
    for attr, grp in zip(attributes, groups):
        spec = next((s for s in sub_specs if s["attribute"] == attr), None)
        if spec is None:
            raise KeyError(f"No subpopulation config for attribute {attr}")
        codes = spec["value_map"].get(grp)
        if codes is None:
            raise KeyError(f"No value_map entry for ({attr}, {grp})")
        col = spec["column"]
        mask = mask & df[col].isin(list(codes))
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="CGSS: weighted distribution CSV")
    parser.add_argument("--config", type=str, default="data/cgss2023/metadata/cgss_config.json")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "eval", "all"],
        help="须与 build_refined_qkey_dict 使用的 split 一致",
    )
    parser.add_argument(
        "--out_dataset",
        type=str,
        default="cgss-train",
        help="输出 data/{out_dataset}/processed/{out_dataset}.csv",
    )
    parser.add_argument(
        "--refined",
        type=str,
        default="auto",
        help="refined_qkey_dict.json；auto=data/{out_dataset}/processed/refined_qkey_dict.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="auto",
        help="输出 CSV；auto=data/{out_dataset}/processed/{out_dataset}.csv",
    )
    parser.add_argument("--n_combination", type=int, default=1, help="Joint subpopulation order (1=单维)")
    args = parser.parse_args()

    cfg_path = REPO_ROOT / args.config if not pathlib.Path(args.config).is_absolute() else pathlib.Path(args.config)
    if args.refined == "auto":
        refined_path = REPO_ROOT / "data" / args.out_dataset / "processed" / "refined_qkey_dict.json"
    else:
        refined_path = REPO_ROOT / args.refined if not pathlib.Path(args.refined).is_absolute() else pathlib.Path(args.refined)
    if args.output == "auto":
        out_path = REPO_ROOT / "data" / args.out_dataset / "processed" / f"{args.out_dataset}.csv"
    else:
        out_path = REPO_ROOT / args.output if not pathlib.Path(args.output).is_absolute() else pathlib.Path(args.output)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(refined_path, "r", encoding="utf-8") as f:
        refined = json.load(f)

    micro = pathlib.Path(cfg["microdata_path"])
    if not micro.is_absolute():
        micro = REPO_ROOT / micro
    weight_col = cfg["weight_column"]
    if weight_col.startswith("__REPLACE"):
        print("ERROR: Set weight_column in cgss_config.json from the codebook.", file=sys.stderr)
        sys.exit(1)

    df, meta = pyreadstat.read_dta(str(micro))
    if weight_col not in df.columns:
        print(f"ERROR: weight column {weight_col} not in .dta", file=sys.stderr)
        sys.exit(1)

    labels_map = meta.variable_value_labels or {}
    global_invalid = [float(x) for x in cfg.get("global_invalid_codes", [])]
    global_refusal = [float(x) for x in cfg.get("global_refusal_codes", [])]
    sub_specs: List[dict] = cfg["subpopulations"]
    qcfg_by_var = question_cfg_by_var(cfg)

    loaded_pair: List[Tuple[str, str]] = []
    for spec in sub_specs:
        attr = spec["attribute"]
        for grp in spec["value_map"].keys():
            loaded_pair.append((attr, grp))
    attribute_group_pair = generate_combined_pairs(loaded_pair, args.n_combination)

    rows_out: List[Dict[str, Any]] = []
    w_all = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)

    for qkey, qmeta in refined.items():
        if qkey not in df.columns:
            print(f"WARN: question column {qkey} missing in .dta, skip", file=sys.stderr)
            continue
        vlabels = labels_map.get(qkey, {})
        q_cfg = qcfg_by_var.get(qkey)
        q_invalid = [float(x) for x in (q_cfg or {}).get("invalid_codes", [])]
        q_ref = (q_cfg or {}).get("refusal_codes", None)
        if q_ref is not None:
            q_ref = [float(x) for x in q_ref]

        code_to_idx, option_list, ordinal, refusal_set = _build_code_index(
            vlabels, global_invalid, global_refusal, q_invalid, q_ref
        )
        if not option_list:
            continue

        y = pd.to_numeric(df[qkey], errors="coerce").to_numpy(dtype=float)

        for ag in attribute_group_pair:
            if isinstance(ag[0], list):
                attributes, groups = ag[0], ag[1]
            else:
                attributes, groups = [ag[0]], [ag[1]]

            mask = _subpop_mask(df, sub_specs, attributes, groups)
            if not mask.any():
                continue

            idxs = np.where(mask.to_numpy())[0]
            resp_w = np.zeros(len(option_list), dtype=float)
            refuse_w = 0.0

            for i in idxs:
                wi = w_all[i]
                if not np.isfinite(wi) or wi <= 0:
                    continue
                yi = y[i]
                if not np.isfinite(yi):
                    continue
                yf = float(yi)
                if yf in refusal_set:
                    refuse_w += wi
                    continue
                if yf in set(global_invalid) | set(q_invalid):
                    continue
                j = code_to_idx.get(yf)
                if j is None:
                    continue
                resp_w[j] += wi

            total_resp = float(resp_w.sum())
            total_ref = float(refuse_w)
            denom = total_resp + total_ref
            if denom <= 0:
                continue
            refusal_rate = total_ref / denom
            try:
                resp_norm = list_normalize(list(resp_w))
            except ValueError:
                continue

            attr_str = str(attributes[0]) if len(attributes) == 1 else str(attributes)
            grp_str = str(groups[0]) if len(groups) == 1 else str(groups)

            rows_out.append(
                {
                    "qkey": qkey,
                    "attribute": attr_str,
                    "group": grp_str,
                    "responses": str(resp_norm),
                    "refusal_rate": refusal_rate,
                    "ordinal": str(ordinal),
                    "question": qmeta.get("refined_qbody", ""),
                    "options": str(option_list),
                }
            )

    if not rows_out:
        print("ERROR: no rows produced. Check config, refined JSON, and .dta.", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
