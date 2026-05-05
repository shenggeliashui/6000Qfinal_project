# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
论文式 WD **参照区间**（与 ``subpop.utils.survey_utils.ordinal_emd`` 一致）：

1. **上界（uniform）**：在每个题目上，用「全体合法选项上的均匀分布」与 **标注分布 output_dist**
   算 WD；表示「完全无知」时的误差量级。
2. **下界（bootstrap）**：在仅有 **聚合分布 p**（CSV 中的 ``output_dist``）时，用两次独立
   ``Multinomial(n, p)`` 的经验分布估计之间的 WD，再对 bootstrap 重复取平均；近似刻画 **同一总体下、
   有限样本随机波动** 带来的 WD 量级（与论文「两组受访者」在**仅有聚合表、无微观问卷表**时的蒙特卡洛类比）。
   若你有原始受访者层级数据，应在数据侧拆两组再聚合，本脚本不替代微观流程。

输入：与 ``eval_peft_test_hf`` 相同格式的 ``opnqa_*.csv``（含 ``output_dist``、``ordinal``）。

示例（仓库根目录）::

    python scripts/experiment/compute_wd_bounds.py \\
      --csv=subpop/train/datasets/cgss-eval/opnqa_QA_test.csv \\
      --n_per_side=500 \\
      --n_bootstrap=500 \\
      --seed=42 \\
      --dedupe \\
      --out_csv=./wd_bounds_per_row.csv
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fire
import numpy as np
import pandas as pd

from subpop.train.mcq_option_limit import MAX_MCQ_OPTIONS
from subpop.utils.survey_utils import ordinal_emd


def _parse_lists(row: pd.Series) -> tuple[list[float], list[float]]:
    od = row["output_dist"]
    om = row["ordinal"]
    if isinstance(od, str):
        od = ast.literal_eval(od)
    if isinstance(om, str):
        om = ast.literal_eval(om)
    g = [float(x) for x in od]
    o = [float(x) for x in om]
    if len(g) > MAX_MCQ_OPTIONS:
        g = g[:MAX_MCQ_OPTIONS]
        s = sum(g)
        if s > 0:
            g = [x / s for x in g]
    while len(g) < MAX_MCQ_OPTIONS:
        g.append(0.0)
    if len(o) > MAX_MCQ_OPTIONS:
        o = o[:MAX_MCQ_OPTIONS]
    pad_ord = max(o) if o else 0.0
    while len(o) < MAX_MCQ_OPTIONS:
        o.append(pad_ord)
    sg = sum(g)
    if sg > 0:
        g = [x / sg for x in g]
    return g, o


def _uniform_over_valid(ordinal: list[float]) -> list[float]:
    k = sum(1 for x in ordinal if x >= 0)
    if k == 0:
        return [1.0 / MAX_MCQ_OPTIONS] * MAX_MCQ_OPTIONS
    u = []
    for x in ordinal:
        if x >= 0:
            u.append(1.0 / k)
        else:
            u.append(0.0)
    return u


def _bootstrap_lower_wd(
    gold: list[float],
    ordinal: list[float],
    rng: np.random.Generator,
    n_per_side: int,
    n_bootstrap: int,
) -> float:
    p = np.array(gold, dtype=float)
    tot = p.sum()
    if tot <= 0:
        return float("nan")
    p = p / tot
    acc: list[float] = []
    for _ in range(n_bootstrap):
        c1 = rng.multinomial(n_per_side, p)
        c2 = rng.multinomial(n_per_side, p)
        if c1.sum() == 0 or c2.sum() == 0:
            continue
        p1 = (c1 / c1.sum()).tolist()
        p2 = (c2 / c2.sum()).tolist()
        wd = ordinal_emd(p1, p2, list(ordinal))
        if wd == wd and not np.isnan(wd):  # not nan
            acc.append(float(wd))
    return float(np.mean(acc)) if acc else float("nan")


def main(
    csv: str,
    n_per_side: int = 500,
    n_bootstrap: int = 500,
    seed: int = 42,
    dedupe: bool = True,
    max_rows: int = 0,
    out_csv: str = "",
):
    path = pathlib.Path(csv)
    if not path.is_file():
        raise FileNotFoundError(path.resolve())

    df = pd.read_csv(path, encoding="utf-8")
    for col in ("output_dist", "ordinal"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain column {col!r}")

    if dedupe and {"qkey", "attribute", "group"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["qkey", "attribute", "group"], keep="first")
        print(f"After dedupe by (qkey, attribute, group): {len(df)} rows")

    if max_rows > 0:
        df = df.iloc[: max_rows]

    rng = np.random.default_rng(seed)

    uppers: list[float] = []
    lowers: list[float] = []
    rows_out: list[dict] = []

    for idx, row in df.iterrows():
        gold, ordinal = _parse_lists(row)
        uni = _uniform_over_valid(ordinal)
        wd_up = ordinal_emd(uni, gold, ordinal)
        wd_lo = _bootstrap_lower_wd(gold, ordinal, rng, n_per_side, n_bootstrap)

        if wd_up == wd_up and not np.isnan(wd_up):
            uppers.append(float(wd_up))
        if wd_lo == wd_lo and not np.isnan(wd_lo):
            lowers.append(float(wd_lo))

        qkey = row.get("qkey", "")
        rec = {
            "qkey": qkey,
            "attribute": row.get("attribute", ""),
            "group": row.get("group", ""),
            "wd_upper_uniform": wd_up,
            "wd_lower_bootstrap_mean": wd_lo,
            "n_per_side": n_per_side,
            "n_bootstrap": n_bootstrap,
        }
        rows_out.append(rec)

    print(f"csv={path.resolve()} rows_processed={len(df)}")
    print(
        "WD upper (uniform vs gold): "
        f"mean={np.nanmean(uppers):.6f} median={np.nanmedian(uppers):.6f} "
        f"(nan/invalid excluded from summary counts)"
    )
    print(
        "WD lower (bootstrap multinomial, mean over reps): "
        f"mean={np.nanmean(lowers):.6f} median={np.nanmedian(lowers):.6f}"
    )
    print(
        "Interpret: model WD from eval_peft_test_hf should ideally lie **between** "
        "lower and upper means for a comparable test split (same ordinal_emd definition)."
    )

    if out_csv:
        outp = pathlib.Path(out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(outp, index=False)
        print(f"Wrote per-row bounds: {outp.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
