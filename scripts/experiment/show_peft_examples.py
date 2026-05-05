# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
从测试 CSV（如 cgss-eval）中随机抽若干条，加载 LoRA 后打印：qkey、截断的 prompt、
标注分布 output_dist 的 Top-k、模型在选项 token 上的预测 Top-k（与 eval_peft_test_hf 同一套打分）。

仓库根目录：pip install -e . 后执行：

    python scripts/experiment/show_peft_examples.py \\
      --model_name=Qwen/Qwen2.5-0.5B \\
      --lora_path=./test20260504_xxxx \\
      --test_csv=subpop/train/datasets/cgss-eval/opnqa_QA_test.csv \\
      --n_examples=5 --seed=42 --is_chat=False
"""
from __future__ import annotations

import ast
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import fire
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from subpop.train.datasets.opinionqa_dataset import get_preprocessed_opinionqa_ce_or_wd_loss
from subpop.train.utils.train_utils import set_tokenizer_params

import eval_peft_test_hf as _eval_hf

_collate_opnqa_batch = _eval_hf._collate_opnqa_batch
_label_token_ids = _eval_hf._label_token_ids


def _topk_labels(probs: list[float], letters: list[str], k: int) -> list[tuple[str, float]]:
    pairs = list(zip(letters, probs))
    pairs.sort(key=lambda x: -x[1])
    return pairs[:k]


def main(
    model_name: str,
    lora_path: str,
    test_csv: str,
    is_chat: bool = False,
    n_examples: int = 5,
    seed: int = 42,
    topk: int = 5,
    max_prompt_chars: int = 600,
):
    test_path = pathlib.Path(test_csv)
    if not test_path.is_file():
        raise FileNotFoundError(test_path.resolve())
    lp = pathlib.Path(lora_path)
    if not lp.is_dir() or not (lp / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Invalid lora_path: {lp.resolve()}")

    df = pd.read_csv(test_path)
    n = len(df)
    if n == 0:
        print("test_csv is empty.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    set_tokenizer_params(tokenizer)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(lp), is_trainable=False)
    model.eval()
    device = next(model.parameters()).device

    label_ids = _label_token_ids(tokenizer)
    letters = [chr(ord("A") + i) for i in range(len(label_ids))]
    label_idx = torch.tensor(label_ids, dtype=torch.long, device=device)

    dataset = get_preprocessed_opinionqa_ce_or_wd_loss(
        None, tokenizer, str(test_path), is_chat, save=True
    )

    if len(dataset) != n:
        print(
            f"Warning: dataset rows ({len(dataset)}) != csv rows ({n}); align by min length."
        )
    n_align = min(len(dataset), n)

    rng = random.Random(seed)
    k = min(n_examples, n_align)
    indices = rng.sample(range(n_align), k=k)
    indices.sort()

    print(f"test_csv={test_path} rows={n} showing {k} random indices (seed={seed})")
    print("=" * 80)

    for rank, i in enumerate(indices):
        row = df.iloc[i]
        feat = dataset[i]
        batch = _collate_opnqa_batch(tokenizer, [feat])
        batch = {
            kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
            for kk, vv in batch.items()
        }
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits.float()
            probs = F.softmax(logits, dim=-1)
            bsz = 1
            tpos = batch["target_token_position"] - 1
            target_token_prob = probs[torch.arange(bsz, device=device), tpos, :][:, label_idx]
            target_token_prob = target_token_prob / target_token_prob.sum(dim=-1, keepdim=True)
            pred_probs = target_token_prob[0].detach().cpu().tolist()

        gold = ast.literal_eval(row["output_dist"])
        if len(gold) > len(letters):
            gold = gold[: len(letters)]
        gold_full = gold + [0.0] * (len(letters) - len(gold))
        s = sum(gold_full)
        if s > 0:
            gold_full = [x / s for x in gold_full]

        qkey = row.get("qkey", "")
        prompt = str(row.get("input_prompt", ""))
        prompt_show = prompt if len(prompt) <= max_prompt_chars else prompt[:max_prompt_chars] + "\n... [truncated]"

        print(f"\n### Example {rank + 1}  row_index={i}  qkey={qkey!r}")
        print("--- input_prompt (truncated) ---")
        print(prompt_show)
        print("--- gold output_dist (top-k) ---")
        for letter, p in _topk_labels(gold_full, letters, topk):
            print(f"  {letter}: {p:.4f}")
        print("--- model P(option token | prompt) (top-k) ---")
        for letter, p in _topk_labels(pred_probs, letters, topk):
            print(f"  {letter}: {p:.4f}")
        print("-" * 80)


if __name__ == "__main__":
    fire.Fire(main)
