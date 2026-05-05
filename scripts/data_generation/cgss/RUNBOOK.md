# CGSS 数据与微调速查（SubPOP-Train / SubPOP-Eval 式划分）

> 在仓库根目录 `subpop/`（含 `pyproject.toml`）下执行文中命令；续行以 Git Bash / Linux / macOS 为例（行尾 `\` 后勿空格），Windows CMD 将 `\` 改为 `^`。

---

## 一、与 README 的对应关系


| 概念               | 本仓库路径 / 设置                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------ |
| **SubPOP-Train** | `data/cgss-train/` + `questions_train`，划分 `train_ratio=0.9`，`val=0.1`，`test=0`             |
| **SubPOP-Eval**  | `data/cgss-eval/` + `questions_eval`，划分 `train=0`，`val=0`，`test=1.0`                       |
| **微观数据**         | `data/cgss2023/raw/CGSS2023.dta`（`cgss_config.json` 中 `microdata_path`）                    |
| **人口学 steering** | `data/cgss2023/metadata/demographics_cgss.csv` + `steering_prompts_zh.json`（Train/Eval 共用） |


---

## 二、生成 refined + 分布 CSV

```bash
python scripts/data_generation/cgss/build_refined_qkey_dict.py --split train --out_dataset cgss-train
python scripts/data_generation/cgss/generate_cgss_distribution.py --split train --out_dataset cgss-train

python scripts/data_generation/cgss/build_refined_qkey_dict.py --split eval --out_dataset cgss-eval
python scripts/data_generation/cgss/generate_cgss_distribution.py --split eval --out_dataset cgss-eval
```

**输出：**

- `data/cgss-train/processed/cgss-train.csv`
- `data/cgss-eval/processed/cgss-eval.csv`

---

## 三、生成微调用 `opnqa_*.csv`（`prepare_finetuning_data`）

### Train（与 README SubPOP-Train 比例一致）

```bash
python scripts/data_generation/prepare_finetuning_data.py --dataset cgss-train \
  --steer_prompts_file_path data/cgss2023/metadata/steering_prompts_zh.json \
  --steer_demographics_file_path data/cgss2023/metadata/demographics_cgss.csv \
  --train_ratio 0.9 --val_ratio 0.1 --test_ratio 0 --subgroup_coder identity
```

### Eval（与 README SubPOP-Eval 一致，全部进 test）

```bash
python scripts/data_generation/prepare_finetuning_data.py --dataset cgss-eval \
  --steer_prompts_file_path data/cgss2023/metadata/steering_prompts_zh.json \
  --steer_demographics_file_path data/cgss2023/metadata/demographics_cgss.csv \
  --train_ratio 0 --val_ratio 0 --test_ratio 1 --subgroup_coder identity
```

---

## 四、拷贝到训练目录（`subpop/train/datasets/cgss-train` / `cgss-eval`）

`prepare_finetuning_data.py` 先写入 `data/<dataset>/processed/opnqa_*.csv`。微调 `opnqa_cgss_steering_dataset` 从包路径读取（**当前工作目录 = 仓库根目录**）：

```text
subpop/train/datasets/{dataset_path}/opnqa_{steering_type}_{train,val,test}.csv
```

`{dataset_path}` 与训练时 `--dataset_path` 一致（CGSS 常用 `cgss-train` 或 `cgss-eval`），通常与第三节 `--dataset` 同名。将生成的 CSV 拷入对应子目录：

```bash
mkdir -p subpop/train/datasets/cgss-train subpop/train/datasets/cgss-eval
cp -f data/cgss-train/processed/opnqa_*.csv subpop/train/datasets/cgss-train/
cp -f data/cgss-eval/processed/opnqa_*.csv subpop/train/datasets/cgss-eval/
```

**说明：**

- `opnqa_*.csv` 含各 SteeringPromptType（QA、BIO、PORTRAY、ALL）；微调时与 `--steering_type` 一致的那份才会被用到。
- **Windows CMD**：`mkdir` 后执行  
`copy /Y data\cgss-train\processed\opnqa_*.csv subpop\train\datasets\cgss-train\`（eval 同理）。

---

## 五、`torchrun` 微调（CGSS）

### 前置条件

- 已完成第四节：`subpop/train/datasets/cgss-train/`（或 `cgss-eval/`，与 `--dataset_path` 一致）下存在 `opnqa_{steering_type}_{train,val,test}.csv`（常用 `opnqa_QA_*.csv`）。
- 已在仓库根目录执行 `pip install -e .` 并安装依赖：
  - 全量：`requirements.txt`（含 vLLM 等，解析较慢）
  - **仅数据 + 微调**：`pip install -r requirements-finetune.txt` 后再 `pip install -e .`
- 拉 gated 模型：`export HF_TOKEN=你的token`；可选 `export TOKENIZERS_PARALLELISM=true`

### 数据路径约定（`opnqa_cgss_steering_dataset`）

```text
subpop/train/datasets/{dataset_path}/opnqa_{steering_type}_train.csv
subpop/train/datasets/{dataset_path}/opnqa_{steering_type}_val.csv
subpop/train/datasets/{dataset_path}/opnqa_{steering_type}_test.csv
```

### 与 README 的差异（CGSS 三件套）

- `--dataset=opnqa_cgss_steering_dataset`（勿用 `opnqa_steering_dataset`，后者对应 `opnqa_500_*` 文件名）。
- `--dataset_path=cgss-train` 或 `cgss-eval`（与 `subpop/train/datasets/` 下子目录名一致）。
- `--steering_type=QA`（或 BIO / PORTRAY / ALL，须与 CSV 文件名一致）。

### 单卡示例（Colab / 单 GPU：`--nproc_per_node=1`）

将 `OUTPUT_DIR_HERE` 换为可写持久路径；**Qwen2.5 Base** 将 `MODEL_NAME_HERE` 换为如 `Qwen/Qwen2.5-7B`，并保持 `--is_chat=False`。

```bash
export HF_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=true

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
  scripts/experiment/run_finetune.py \
  --enable_fsdp \
  --low_cpu_fsdp \
  --fsdp_config.pure_bf16 \
  --fsdp_config.checkpoint_type=StateDictType.FULL_STATE_DICT \
  --use_peft=True \
  --use_fast_kernels \
  --peft_method=lora \
  --use_fp16 \
  --mixed_precision \
  --batch_size_training=4 \
  --val_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --dist_checkpoint_root_folder=./outputs/fsdp_dist \
  --dist_checkpoint_folder=fine-tuned \
  --batching_strategy=padding \
  --dataset=opnqa_cgss_steering_dataset \
  --dataset_path=cgss-train \
  --steering_type=QA \
  --output_dir=/content/drive/MyDrive/Colab/subpop-cgss-lora \
  --model_name=Qwen/Qwen2.5-7B \
  --model_nickname=cgss-lora \
  --lr=2e-4 \
  --num_epochs=1 \
  --max_train_step=200 \
  --max_eval_step=50 \
  --weight_decay=0 \
  --loss_function_type=ce \
  --which_scheduler=cosine \
  --warmup_ratio=0.1 \
  --gamma=0.85 \
  --attribute=None \
  --group=None \
  --lora_config.r=8 \
  --lora_config.lora_alpha=32 \
  --is_chat=False \
  --use_wandb=False \
  --num_workers_dataloader=0
```

### 说明

- `**--output_dir**`：程序会在其后**追加时间戳**子目录再写入 LoRA；父目录须可写。
- `**--num_epochs` / `--max_train_step` / `--max_eval_step`**：任一到达即停；试跑可减小 `max_train_step`。
- **OOM**：减小 `batch_size_training` / `val_batch_size`，或换更小模型 / 4bit（见 README 量化）。
- **cgss-eval**：将 `--dataset_path=cgss-eval`；eval 划分下 train CSV 可能为空，一般**训练仍用 cgss-train**，独立评估再切 `dataset_path`。
- **checkpoint 字段**：以 `fsdp_config.checkpoint_type=...` 为准（勿与 README 里旧名 `checkpoint_type` 混淆）。
- **W&B**：`--use_wandb=True` 并设置 `--name`、`--wandb_config.project`、`--wandb_config.entity`。

---

## 六、Colab 上测测试集（无 vLLM，HF+PEFT）

训练结束后 `--output_dir` 会带时间戳子目录，内含 LoRA（`adapter_config.json` 等）。在**仓库根目录**执行；**基线**与**微调**共用同一 `test_csv` / `model_name` / `loss_function_type` / `is_chat`，仅差是否加载 LoRA。

**`--test_csv` 默认示例**：使用 **`cgss-eval`** 下的 `opnqa_QA_test.csv`（SubPOP-Eval 划分，`train`/`val` 常为空、**test 为完整留出集**）。不要用 **`cgss-train`** 的 test 作为默认：`prepare_finetuning_data` 若 **`test_ratio=0`**，`cgss-train/opnqa_*_test.csv` 往往为空，会导致评估报错。

### 6.1 基座基线（不加载 LoRA）

```bash
python scripts/experiment/eval_peft_test_hf.py \
  --model_name=Qwen/Qwen2.5-0.5B \
  --test_csv=subpop/train/datasets/cgss-eval/opnqa_QA_test.csv \
  --no_peft=True \
  --loss_function_type=ce \
  --is_chat=False \
  --batch_size=2 \
  --max_rows=0 \
  --num_workers=0 \
  --output_csv=./eval_baseline.csv
```

- `**--no_peft=True**`：不要传 `--lora_path`。
- `**--max_rows=0**`：评全表；试跑可改为 `--max_rows=100`。
- `**--output_csv**`：逐行 `kl` / `wd` / `loss_used`，便于与 6.2 对比。

### 6.2 微调模型（加载 LoRA）

```bash
python scripts/experiment/eval_peft_test_hf.py \
  --model_name=Qwen/Qwen2.5-0.5B \
  --test_csv=subpop/train/datasets/cgss-eval/opnqa_QA_test.csv \
  --lora_path=./test20260504_xxxxxx \
  --loss_function_type=ce \
  --is_chat=False \
  --batch_size=2 \
  --max_rows=0 \
  --num_workers=0 \
  --output_csv=./eval_ft.csv
```

- `**--lora_path**`：填日志里 `PEFT modules are saved in ...` 对应目录（与训练 `--output_dir` **最终带时间戳**路径一致；将 `test20260504_xxxxxx` 换成实际目录名）。
- 其余参数与 **6.1** 对齐，便于公平对比。

### 6.3 参数说明与对比方式

- `**--test_csv`**：须含 `input_prompt` / `output_dist` / `ordinal`。**独立测试集评估优先用** `subpop/train/datasets/cgss-eval/opnqa_{steering_type}_test.csv`（与 **6.1 / 6.2** 示例一致）。仅在已从 `cgss-train` 划出非空 test 时，才可改用 `cgss-train/opnqa_*_test.csv`；若 `test_ratio=0` 导致 train 侧 test 为空，须重新 `prepare_finetuning_data` 划出 test，或继续使用 **`cgss-eval`**。
- `**--loss_function_type**`：与训练 `ce` / `wd` 一致；终端打印按 batch 平均的 loss / KL / WD。
- 与 `train_utils.evaluation` 同一套选项头（ `A`… `Z`）与 KL/WD 定义；**无需 vLLM**。
- **对比**：见 **6.5**（配对脚本）；指标含义见 **6.4**。**KL / WD 越低通常越好**（模型在选项头上的概率分布越接近 CSV 里的 `output_dist`）。

### 6.4 `eval_peft_test_hf.py` 输出怎么读

**终端最后几行**（每次跑完都会打印）：

| 打印项 | 含义 |
|--------|------|
| `test_csv=...` | 实际使用的测试表路径。 |
| `rows=... batch_size=... batches=...` | 样本条数、批大小、批次数。 |
| `loss_function_type=... -> mean batch loss` | 训练目标对应的 batch 平均「损失」：`ce` 时与下一行 KL 同量级（按 batch 内对样本平均后再对 batch 平均）；`wd` 时与 WD 流程一致。 |
| `mean batch KL (ce term)` | 选项分布上的 **KL（交叉熵 − 熵）** 的 batch 平均；**越小**表示模型预测分布越接近标注 `output_dist`。 |
| `mean batch WD (ordinal EMD)` | 把选项当作有序类时的 **Earth Mover’s Distance**（代码里 `ordinal_emd`）；**越小**表示序结构意义下越接近标注分布。 |
| `Wrote per-row metrics: ...` | 若传了 `--output_csv`，逐行结果已写入该文件。 |

**`--output_csv` 生成的 CSV**（三列，无表头以外的 key）：

| 列名 | 含义 |
|------|------|
| `kl` | 该样本（该行对应测试集顺序的一条）上的 KL。 |
| `wd` | 该样本上的 ordinal WD。 |
| `loss_used` | 与 `--loss_function_type` 一致：`ce` 时等于该行的 KL；`wd` 时等于该行的 WD。 |

**注意**：输出 CSV **不含 `qkey`**，与其它列的对应关系是 **行号 = 与 `test_csv` 经同一数据集顺序的第 i 条样本**。两次评估（基线 vs LoRA）必须用 **同一 `test_csv`**、同一 **`--max_rows`**（通常都用 `0` 评全表），否则无法按行配对。

### 6.5 基线 vs LoRA：配对对比与简单作图

在分别得到 `eval_baseline.csv` 与 `eval_ft.csv`（**6.1**、**6.2**）后，用仓库脚本按行对齐并汇总（需已安装 **pandas**；作图需 **`matplotlib`**）：

```bash
python scripts/experiment/compare_eval_csv.py \
  --baseline_csv=./eval_baseline.csv \
  --ft_csv=./eval_ft.csv \
  --out_csv=./eval_compare_paired.csv \
  --plot_path=./eval_compare.png
```

Colab 代码格示例：

```python
!python scripts/experiment/compare_eval_csv.py \
  --baseline_csv=./eval_baseline.csv \
  --ft_csv=./eval_ft.csv \
  --out_csv=./eval_compare_paired.csv \
  --plot_path=./eval_compare.png
```

脚本会打印：**全局均值**（基座 vs 微调）、**配对差分** `baseline − ft`（**KL/WD 上为正**通常表示 **微调后更小、更好**）、以及「有多少比例的样本上 KL/WD 变好」。`eval_compare_paired.csv` 含逐行 `kl_baseline` / `kl_ft` / `kl_delta_base_minus_ft` 等，便于再导入 R / Excel 做配对检验。

**可选 Notebook 内直接画图**（依赖 `pandas` + `matplotlib`，且已生成两个 eval CSV）：

```python
import pandas as pd
import matplotlib.pyplot as plt

b = pd.read_csv("eval_baseline.csv")
f = pd.read_csv("eval_ft.csv")
assert len(b) == len(f)

delta_kl = b["kl"] - f["kl"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(delta_kl, bins=40, color="steelblue", alpha=0.85)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("KL improvement (baseline − LoRA) per row")
ax.set_xlabel("positive → LoRA lower KL (usually better)")
fig.tight_layout()
plt.savefig("eval_delta_kl_hist.png", dpi=150)
plt.show()
print("mean KL baseline", b["kl"].mean(), "mean KL ft", f["kl"].mean())
```

**其它基线**：同一 CSV 上比较均匀分布、或论文/助教给定数字（均匀对照需自行实现，或与 README 公开 checkpoint 对齐）。

---

## 七、继续训练 / 断点续训（无 `--resume_from_checkpoint`）

本仓库 **没有** `--resume_from_checkpoint` 参数；`fire` 传入会被忽略。续训有两种方式：

### 7.1 自动续训（推荐）：依赖 `peft_checkpointing/`

训练时每个 epoch 会把 **可恢复状态** 写到：

```text
{dirname(最终 output_dir)}/peft_checkpointing/
```

例如 `--output_dir=./test`，程序会先变成 `./test20260504_162916`，则 `dirname` 为 `**.**`，目录为 `**./peft_checkpointing/**`（与带时间戳的 `./test20260504_162916/` 并列）。

其内含：`adapter_config.json`、LoRA 权重、`optimizer.pt`、`scheduler.pt`、`metadata.json`、`rng_state.pth` 等（`grad_scaler.pt` 仅在 `use_fp16=True` 时可能有）。

**续训做法**：在**同一工作目录**下再跑一次训练命令（`--output_dir` 前缀与上次相同，如仍用 `./test`）。若 `**./peft_checkpointing` 仍存在**，启动时会：

1. 从 `peft_checkpointing` **加载 LoRA**；
2. 若还存在 `optimizer.pt` + `scheduler.pt` + `metadata.json`，会 **恢复优化器、调度器、步数、epoch** 等。

**新开一轮、不要续训**：先删除或移走 `**./peft_checkpointing`**（否则会误加载旧任务）。

### 7.2 只指定「上一轮保存的 LoRA 目录」： `--from_peft_checkpoint`

若你手里主要是日志里的 `**PEFT modules are saved in ./test20260504_162916**` 这一类目录（含 `adapter_config.json`），希望 **从该适配器接着训**，可加：

```bash
--from_peft_checkpoint=./test20260504_162916
```

优先级：`**--from_peft_checkpoint` > 自动 `peft_checkpointing` > 全新 LoRA**。

- 仍可与 **7.1** 同时利用：若 `peft_checkpointing` 里还有 `optimizer.pt` 等，**优化器续训**仍从 `peft_checkpointing` 读（与当前 `finetuning.py` 逻辑一致）。
- 若只有适配器目录、没有 `peft_checkpointing`，则 **仅加载权重**，优化器从随机初始化开始（等价于「热启动」）。

### 7.3 Colab 示例（续训）

```bash
!python scripts/experiment/run_finetune.py \
  --enable_fsdp=False \
  --batch_size_training=1 \
  --dataset=opnqa_cgss_steering_dataset \
  --output_dir=./test \
  --model_name=Qwen/Qwen2.5-0.5B \
  --num_epochs=2 \
  --max_train_step=100 \
  --use_peft \
  --peft_method=lora \
  --use_fp16 \
  --use_wandb=False
```

- **自动续训**：不要加 `--from_peft_checkpoint`，保留上次生成的 `**./peft_checkpointing`**。
- **从某次 `test时间戳` 目录热启动**：加上  
`--from_peft_checkpoint=./test20260504_162916`  
（路径换成你机器上的实际目录）。

