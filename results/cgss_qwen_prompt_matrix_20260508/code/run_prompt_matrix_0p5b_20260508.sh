#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/6000Q
export PATH=/root/miniconda3/bin:$PATH
source /etc/network_turbo || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
LOG="logs/prompt_matrix_0p5b_QA_BIO_PORTRAY_20260508.log"
exec >> "$LOG" 2>&1

echo "=== START 0.5B QA/BIO/PORTRAY $(date) ==="
run_one() {
  local prompt="$1"
  local model_path="/root/autodl-tmp/hf_models/Qwen2.5-0.5B"
  local nickname="qwen2.5-0.5b"
  local out_prefix="outputs/colab_qlora4bit_${nickname}_cgss_${prompt}_ce_"
  local base_csv="eval_baseline_0p5B_${prompt}.csv"
  echo ""
  echo "=== CASE model=0.5B prompt=${prompt} $(date) ==="
  if [ -d outputs/peft_checkpointing ]; then
    backup="outputs/peft_checkpointing_before_0p5B_${prompt}_$(date +%Y%m%d_%H%M%S)"
    mv outputs/peft_checkpointing "$backup"
    echo "Moved stale/resume checkpoint to $backup"
  fi
  echo "--- baseline eval ---"
  python scripts/experiment/eval_peft_test_hf.py \
    --model_name="$model_path" \
    --test_csv="subpop/train/datasets/cgss-eval/opnqa_${prompt}_test.csv" \
    --no_peft=True \
    --loss_function_type=ce \
    --is_chat=False \
    --batch_size=4 \
    --max_rows=0 \
    --num_workers=0 \
    --output_csv="$base_csv"
  echo "--- finetune ---"
  python scripts/experiment/run_finetune.py \
    --model_name="$model_path" \
    --model_nickname="${nickname}-base" \
    --dataset="opnqa_cgss_steering_dataset" \
    --dataset_path="cgss-train" \
    --steering_type="$prompt" \
    --loss_function_type="ce" \
    --enable_fsdp=False \
    --quantization="4bit" \
    --use_peft=True --peft_method="lora" \
    --lora_config.r=16 --lora_config.lora_alpha=64 --lora_config.lora_dropout=0.05 \
    --batch_size_training=1 --gradient_accumulation_steps=64 \
    --context_length=1024 \
    --which_scheduler="cosine" --warmup_ratio=0.03 \
    --lr=2e-4 --weight_decay=0.0 \
    --num_epochs=3 \
    --use_fast_kernels=True \
    --use_wandb=False \
    --output_dir="$out_prefix"
  local out_dir
  out_dir=$(ls -td ${out_prefix}* | head -n 1)
  echo "OUT_DIR=$out_dir"
  test -f "$out_dir/adapter_config.json"
  echo "--- ft eval ---"
  python scripts/experiment/eval_peft_test_hf.py \
    --model_name="$model_path" \
    --test_csv="subpop/train/datasets/cgss-eval/opnqa_${prompt}_test.csv" \
    --no_peft=False \
    --lora_path="$out_dir" \
    --loss_function_type=ce \
    --is_chat=False \
    --batch_size=4 \
    --max_rows=0 \
    --num_workers=0 \
    --output_csv="$out_dir/eval_ft.csv"
  echo "--- compare ---"
  python scripts/experiment/compare_eval_csv.py \
    --baseline_csv="$base_csv" \
    --ft_csv="$out_dir/eval_ft.csv" \
    --out_csv="$out_dir/eval_compare_paired.csv" \
    --plot_path="$out_dir/eval_compare.png"
  python - <<PY
import pandas as pd
base=pd.read_csv("$base_csv")
ft=pd.read_csv("$out_dir/eval_ft.csv")
print("RESULT", "0.5B", "$prompt", "baseline_kl", float(base.kl.mean()), "baseline_wd", float(base.wd.mean()), "ft_kl", float(ft.kl.mean()), "ft_wd", float(ft.wd.mean()), "out", "$out_dir")
PY
}
run_one QA
run_one BIO
run_one PORTRAY

echo "=== END 0.5B QA/BIO/PORTRAY $(date) ==="
