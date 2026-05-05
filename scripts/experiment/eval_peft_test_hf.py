# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Colab / 本机：用 Transformers + PEFT 在测试 CSV 上算与训练一致的 KL（ce）或 WD（wd），无需 vLLM。

仓库根目录（含 pyproject 的 subpop/）下执行，且已 pip install -e .：

  # 微调后（LoRA）
  python scripts/experiment/eval_peft_test_hf.py \\
    --model_name=Qwen/Qwen2.5-0.5B \\
    --lora_path=./test20260504_xxxx \\
    --test_csv=subpop/train/datasets/cgss-train/opnqa_QA_test.csv \\
    --output_csv=./eval_ft.csv

  # 基线：同一基座、不加载 LoRA（与微调对比用）
  python scripts/experiment/eval_peft_test_hf.py \\
    --model_name=Qwen/Qwen2.5-0.5B \\
    --no_peft=True \\
    --test_csv=subpop/train/datasets/cgss-train/opnqa_QA_test.csv \\
    --output_csv=./eval_baseline.csv
"""
from __future__ import annotations

import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fire
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from subpop.train.datasets.opinionqa_dataset import get_preprocessed_opinionqa_ce_or_wd_loss
from subpop.train.mcq_option_limit import MAX_MCQ_OPTIONS
from subpop.train.utils.train_utils import ordinal_emd, set_tokenizer_params


def _label_token_ids(tokenizer):
    return [
        tokenizer.encode(" " + chr(ord("A") + i), add_special_tokens=False)[-1]
        for i in range(MAX_MCQ_OPTIONS)
    ]


def _collate_opnqa_batch(tokenizer, features: list[dict]) -> dict[str, torch.Tensor]:
    """左 padding 与训练侧一致，并修正 target_token_position。"""
    max_len = max(len(f["input_ids"]) for f in features)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    input_ids = []
    attention_mask = []
    target_pos = []
    for f in features:
        ids = f["input_ids"]
        pad_len = max_len - len(ids)
        if tokenizer.padding_side == "left":
            padded_ids = [pad_id] * pad_len + ids
            am = [0] * pad_len + [1] * len(ids)
            target_pos.append(int(f["target_token_position"]) + pad_len)
        else:
            padded_ids = ids + [pad_id] * pad_len
            am = [1] * len(ids) + [0] * pad_len
            target_pos.append(int(f["target_token_position"]))

        input_ids.append(padded_ids)
        attention_mask.append(am)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "target_token_position": torch.tensor(target_pos, dtype=torch.long),
        "response_distribution": torch.tensor(
            [f["response_distribution"] for f in features], dtype=torch.float32
        ),
        "ordinal_info": torch.tensor(
            [
                f["ordinal_info"]
                if "ordinal_info" in f and f["ordinal_info"] is not None
                else [0.0] * MAX_MCQ_OPTIONS
                for f in features
            ],
            dtype=torch.float32,
        ),
    }
    return batch


def main(
    model_name: str,
    test_csv: str,
    lora_path: str = "",
    no_peft: bool = False,
    loss_function_type: str = "ce",
    is_chat: bool = False,
    batch_size: int = 2,
    max_rows: int = 0,
    output_csv: str = "",
    num_workers: int = 0,
):
    """
    Args:
        model_name: HF 基座模型 ID 或本地路径（与训练时一致）。
        test_csv: 测试集 CSV（列与 prepare_finetuning_data 一致，含 input_prompt / output_dist / ordinal）。
        lora_path: 训练产出的 LoRA 目录；与 ``no_peft=True`` 二选一。
        no_peft: 为 True 时只评 **基座模型**（基线），不加载 LoRA。
        loss_function_type: ce（前向 KL）或 wd（ordinal EMD 聚合方式与 train_utils.evaluation 一致）。
        is_chat: 与训练时 --is_chat 一致。
        batch_size: 显存紧张时用 1。
        max_rows: >0 时只评前 N 条（试跑）。
        output_csv: 若非空，逐行写入 kl / wd / loss_used。
        num_workers: DataLoader worker 数，Colab 建议 0。
    """
    test_path = pathlib.Path(test_csv)
    if not test_path.is_file():
        raise FileNotFoundError(f"test_csv not found: {test_path.resolve()}")
    if not no_peft and not str(lora_path).strip():
        raise ValueError("Provide --lora_path=... for fine-tuned eval, or --no_peft=True for base-model baseline.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    set_tokenizer_params(tokenizer)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if no_peft:
        model = base
        print("--> Baseline: base model only (no PEFT)")
    else:
        lp = pathlib.Path(lora_path)
        if not lp.is_dir():
            raise FileNotFoundError(f"lora_path is not a directory: {lp.resolve()}")
        if not (lp / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"Missing adapter_config.json under {lp.resolve()}; "
                "use --no_peft=True for base-model baseline, or fix --lora_path"
            )
        model = PeftModel.from_pretrained(base, str(lp), is_trainable=False)
        print(f"--> Loaded LoRA from {lp.resolve()}")
    model.eval()

    device = next(model.parameters()).device
    label_to_token_id = _label_token_ids(tokenizer)
    label_idx = torch.tensor(label_to_token_id, dtype=torch.long, device=device)

    dataset = get_preprocessed_opinionqa_ce_or_wd_loss(
        None, tokenizer, str(test_path), is_chat, save=False
    )
    n = len(dataset)
    if n == 0:
        print("test dataset is empty (0 rows). Nothing to evaluate.")
        return
    if max_rows > 0:
        dataset = dataset.select(range(min(max_rows, n)))
        print(f"Using first {len(dataset)} rows (max_rows={max_rows})")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: _collate_opnqa_batch(tokenizer, b),
    )

    total_loss = 0.0
    total_kl = 0.0
    total_wd = 0.0
    n_batches = 0
    rows_out: list[dict] = []

    for batch in tqdm(loader, desc="eval test (HF+PEFT)"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits.float()
            bsz = logits.size(0)
            probs = F.softmax(logits, dim=-1)
            tpos = batch["target_token_position"] - 1
            target_token_prob = probs[torch.arange(bsz, device=device), tpos, :][:, label_idx]
            target_token_prob = target_token_prob / target_token_prob.sum(dim=-1, keepdim=True)

            resp_dist = batch["response_distribution"].float()
            kl_vec = (
                -torch.sum(resp_dist * torch.log(target_token_prob + 1e-8), dim=-1)
                + torch.sum(resp_dist * torch.log(resp_dist + 1e-8), dim=-1)
            )
            ordinal_info = batch["ordinal_info"].float()
            wd_list = []
            for i in range(bsz):
                wd_list.append(
                    ordinal_emd(resp_dist[i], target_token_prob[i], ordinal_info[i]).to(device)
                )
            wd_vec = torch.stack(wd_list)
            nz = wd_vec != 0
            wd_mean_batch = wd_vec[nz].mean() if nz.any() else wd_vec.mean()

            if loss_function_type == "ce":
                loss_b = kl_vec.mean()
            elif loss_function_type == "wd":
                loss_b = wd_mean_batch
            else:
                raise ValueError(f"Unknown loss_function_type: {loss_function_type}")

            total_loss += float(loss_b.detach().cpu())
            total_kl += float(kl_vec.mean().detach().cpu())
            total_wd += float(wd_mean_batch.detach().cpu())
            n_batches += 1

            if output_csv:
                for i in range(bsz):
                    rows_out.append(
                        {
                            "kl": float(kl_vec[i].detach().cpu()),
                            "wd": float(wd_vec[i].detach().cpu()),
                            "loss_used": float(
                                kl_vec[i].detach().cpu()
                                if loss_function_type == "ce"
                                else wd_vec[i].detach().cpu()
                            ),
                        }
                    )

    mean_loss = total_loss / max(n_batches, 1)
    mean_kl = total_kl / max(n_batches, 1)
    mean_wd = total_wd / max(n_batches, 1)
    print(f"test_csv={test_path}")
    print(f"rows={len(dataset)} batch_size={batch_size} batches={n_batches}")
    print(f"loss_function_type={loss_function_type} -> mean batch loss: {mean_loss:.6f}")
    print(f"mean batch KL (ce term): {mean_kl:.6f}")
    print(f"mean batch WD (ordinal EMD): {mean_wd:.6f}")

    if output_csv:
        out_p = pathlib.Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["kl", "wd", "loss_used"])
            w.writeheader()
            w.writerows(rows_out)
        print(f"Wrote per-row metrics: {out_p.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
