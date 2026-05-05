# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
配对对比 eval_peft_test_hf.py 产出的两个 CSV（基线 vs LoRA），汇总均值并可选作图。

前提：两次评估使用**同一** ``test_csv``、相同 ``--max_rows``（默认全表）、且脚本未改过 batch 顺序；
输出 CSV 按**行号**与测试集样本顺序一一对应。

用法（仓库根目录）::

    python scripts/experiment/compare_eval_csv.py \\
      --baseline_csv=./eval_baseline.csv \\
      --ft_csv=./eval_ft.csv \\
      --out_csv=./eval_compare_paired.csv \\
      --plot_path=./eval_compare.png
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fire
import pandas as pd


def main(
    baseline_csv: str,
    ft_csv: str,
    out_csv: str = "",
    plot_path: str = "",
) -> None:
    base = pathlib.Path(baseline_csv)
    ft_p = pathlib.Path(ft_csv)
    if not base.is_file():
        raise FileNotFoundError(base.resolve())
    if not ft_p.is_file():
        raise FileNotFoundError(ft_p.resolve())

    df_b = pd.read_csv(base)
    df_f = pd.read_csv(ft_p)
    for col in ("kl", "wd", "loss_used"):
        if col not in df_b.columns or col not in df_f.columns:
            raise ValueError(f"Both CSVs need columns kl, wd, loss_used; missing {col!r}")

    if len(df_b) != len(df_f):
        raise ValueError(
            f"Row count mismatch: baseline={len(df_b)} ft={len(df_f)}. "
            "Re-run both evals with the same test_csv and max_rows."
        )

    paired = pd.DataFrame(
        {
            "kl_baseline": df_b["kl"].astype(float),
            "kl_ft": df_f["kl"].astype(float),
            "wd_baseline": df_b["wd"].astype(float),
            "wd_ft": df_f["wd"].astype(float),
            "loss_baseline": df_b["loss_used"].astype(float),
            "loss_ft": df_f["loss_used"].astype(float),
        }
    )
    paired["kl_delta_base_minus_ft"] = paired["kl_baseline"] - paired["kl_ft"]
    paired["wd_delta_base_minus_ft"] = paired["wd_baseline"] - paired["wd_ft"]
    paired["loss_delta_base_minus_ft"] = paired["loss_baseline"] - paired["loss_ft"]
    paired.insert(0, "row_idx", range(len(paired)))

    n = len(paired)
    print(f"rows={n}")
    print("--- 全局均值（越低通常越好：越接近标注分布 output_dist）---")
    for name, bcol, fcol in (
        ("KL", "kl_baseline", "kl_ft"),
        ("WD", "wd_baseline", "wd_ft"),
        ("loss_used", "loss_baseline", "loss_ft"),
    ):
        mb = paired[bcol].mean()
        mf = paired[fcol].mean()
        print(f"  {name}: baseline_mean={mb:.6f}  ft_mean={mf:.6f}  (base-ft)_mean={mb - mf:.6f}")

    print("--- 配对差分 (baseline - ft)：正值表示微调后在该行指标更低更好 ---")
    print(f"  mean(kl_delta):  {paired['kl_delta_base_minus_ft'].mean():.6f}")
    print(f"  mean(wd_delta):  {paired['wd_delta_base_minus_ft'].mean():.6f}")
    imp_kl = (paired["kl_delta_base_minus_ft"] > 0).mean() * 100
    imp_wd = (paired["wd_delta_base_minus_ft"] > 0).mean() * 100
    print(f"  KL 改进比例(ft更好): {imp_kl:.1f}%  WD 改进比例: {imp_wd:.1f}%")

    if out_csv:
        outp = pathlib.Path(out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        paired.to_csv(outp, index=False)
        print(f"Wrote paired table: {outp.resolve()}")

    if plot_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            print("matplotlib 未安装，跳过作图。pip install matplotlib")
            raise SystemExit(1) from e

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax0, ax1 = axes
        ax0.hist(paired["kl_baseline"], bins=40, alpha=0.5, label="baseline KL")
        ax0.hist(paired["kl_ft"], bins=40, alpha=0.5, label="LoRA KL")
        ax0.set_xlabel("KL")
        ax0.set_title("KL per row")
        ax0.legend()

        ax1.hist(paired["wd_baseline"], bins=40, alpha=0.5, label="baseline WD")
        ax1.hist(paired["wd_ft"], bins=40, alpha=0.5, label="LoRA WD")
        ax1.set_xlabel("WD (ordinal EMD)")
        ax1.set_title("WD per row")
        ax1.legend()

        fig.tight_layout()
        pp = pathlib.Path(plot_path)
        pp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pp, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {pp.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
