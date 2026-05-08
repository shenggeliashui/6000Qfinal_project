#!/usr/bin/env bash
set -euo pipefail

# Minimal template for reproducing one CGSS prompt run.
# Set MODEL_PATH, MODEL_NICKNAME, PROMPT, and BATCH_SIZE before running.
cd /root/autodl-tmp/6000Q
export PATH=/root/miniconda3/bin:$PATH
source /etc/network_turbo || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/hf_models/Qwen2.5-7B}
MODEL_NICKNAME=${MODEL_NICKNAME:-qwen2.5-7b}
PROMPT=${PROMPT:-QA}
BATCH_SIZE=${BATCH_SIZE:-2}
OUT_PREFIX="outputs/colab_qlora4bit_${MODEL_NICKNAME}_cgss_${PROMPT}_ce_"
BASE_CSV="eval_baseline_${MODEL_NICKNAME}_${PROMPT}.csv"

if [ -d outputs/peft_checkpointing ]; then
  mv outputs/peft_checkpointing "outputs/peft_checkpointing_before_${PROMPT}_$(date +%Y%m%d_%H%M%S)"
fi

python scripts/experiment/eval_peft_test_hf.py   --model_name="$MODEL_PATH"   --test_csv="subpop/train/datasets/cgss-eval/opnqa_${PROMPT}_test.csv"   --no_peft=True   --loss_function_type=ce   --is_chat=False   --batch_size="$BATCH_SIZE"   --max_rows=0   --num_workers=0   --output_csv="$BASE_CSV"

python scripts/experiment/run_finetune.py   --model_name="$MODEL_PATH"   --model_nickname="${MODEL_NICKNAME}-base"   --dataset="opnqa_cgss_steering_dataset"   --dataset_path="cgss-train"   --steering_type="$PROMPT"   --loss_function_type="ce"   --enable_fsdp=False   --quantization="4bit"   --use_peft=True --peft_method="lora"   --lora_config.r=16 --lora_config.lora_alpha=64 --lora_config.lora_dropout=0.05   --batch_size_training=1 --gradient_accumulation_steps=64   --context_length=1024   --which_scheduler="cosine" --warmup_ratio=0.03   --lr=2e-4 --weight_decay=0.0   --num_epochs=3   --use_fast_kernels=True   --use_wandb=False   --output_dir="$OUT_PREFIX"

OUT_DIR=$(ls -td ${OUT_PREFIX}* | head -n 1)
python scripts/experiment/eval_peft_test_hf.py   --model_name="$MODEL_PATH"   --test_csv="subpop/train/datasets/cgss-eval/opnqa_${PROMPT}_test.csv"   --no_peft=False   --lora_path="$OUT_DIR"   --loss_function_type=ce   --is_chat=False   --batch_size="$BATCH_SIZE"   --max_rows=0   --num_workers=0   --output_csv="$OUT_DIR/eval_ft.csv"

python scripts/experiment/compare_eval_csv.py   --baseline_csv="$BASE_CSV"   --ft_csv="$OUT_DIR/eval_ft.csv"   --out_csv="$OUT_DIR/eval_compare_paired.csv"   --plot_path="$OUT_DIR/eval_compare.png"
