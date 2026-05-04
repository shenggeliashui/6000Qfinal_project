# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    trust_remote_code: bool = False


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"

@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "recipes/quickstart/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""
    
@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class opnqa_steering_dataset:
    dataset: str = "opnqa_steering_dataset"
    file: str = "subpop/train/datasets/opinionqa_dataset.py:get_preprocessed_opinionqa_ce_or_wd_loss"
    train_split: str = "subpop/train/datasets/{dataset_path}/opnqa_500_{steering_type}_train.csv"
    valid_split: str = "subpop/train/datasets/{dataset_path}/opnqa_500_{steering_type}_val.csv"
    test_split:  str = "subpop/train/datasets/{dataset_path}/opnqa_500_{steering_type}_test.csv"

@dataclass
class opnqa_single_demographic_dataset:
    dataset: str = "opnqa_single_demographic_dataset"
    file: str = "subpop/train/datasets/opinionqa_dataset.py:get_preprocessed_opinionqa_ce_or_wd_loss"
    train_split: str = "subpop/train/datasets/{dataset_path}/opnqa_500_{attribute}_{group}_{steering_type}_train.csv"
    valid_split: str = "subpop/train/datasets/{dataset_path}/opnqa_500_{attribute}_{group}_{steering_type}_val.csv"
    test_split:  str = "subpop/train/datasets/{dataset_path}/opnqa_500_{attribute}_{group}_{steering_type}_test.csv"


@dataclass
class opnqa_cgss_steering_dataset:
    """CGSS 微调 CSV：位于 ``subpop/train/datasets/<dataset_path>/``（如 cgss-train、cgss-eval）。"""
    dataset: str = "opnqa_cgss_steering_dataset"
    file: str = "subpop/train/datasets/opinionqa_dataset.py:get_preprocessed_opinionqa_ce_or_wd_loss"
    train_split: str = "subpop/train/datasets/{dataset_path}/opnqa_{steering_type}_train.csv"
    valid_split: str = "subpop/train/datasets/{dataset_path}/opnqa_{steering_type}_val.csv"
    test_split: str = "subpop/train/datasets/{dataset_path}/opnqa_{steering_type}_test.csv"
