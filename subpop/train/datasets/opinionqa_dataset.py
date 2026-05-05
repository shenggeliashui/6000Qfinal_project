# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import ast
import json
from multiprocessing import Lock

import datasets
import pandas as pd

try:
    from subpop.train.mcq_option_limit import MAX_MCQ_OPTIONS
except ImportError:  # dynamic load / 未同步 mcq_option_limit.py 时仍可用
    MAX_MCQ_OPTIONS = 26


def _bos_prefix(tokenizer) -> str:
    """Qwen/Llama3 等 tokenizer 常将 bos_token 置为 None，不可与 str 直接拼接。"""
    t = getattr(tokenizer, "bos_token", None)
    return t if isinstance(t, str) and t else ""


def _eos_suffix(tokenizer) -> str:
    t = getattr(tokenizer, "eos_token", None)
    return t if isinstance(t, str) and t else ""


def get_preprocessed_opinionqa(dataset_config, tokenizer, split, save = True, debug = False):

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(
            _bos_prefix(tokenizer) + sample["input_prompt"],
            add_special_tokens=False
        )
        answer = tokenizer.encode(
            sample["output_prompt"].strip() + _eos_suffix(tokenizer),
            add_special_tokens=False
        ) # detail: adding strip(), because " A" is tokenized as ['<s>', '', ' A']
          # i.e., the whitespace is automatically included in the token list..
        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }
        return sample
    
    preprocessed_file_dir = split.split(".csv")[0] + "_preprocessed.json"

    if os.path.exists(preprocessed_file_dir): # if preprocessed file exists
        print("preprocessed file exists.")
        with open(split.split(".csv")[0] + "_preprocessed.json", 'r') as f:
            dataset_dict = json.load(f)
            dataset = datasets.Dataset.from_dict(dataset_dict)
    else:
        dataset = datasets.load_dataset(
            'csv', 
            data_files = split
        )['train']  # detail: not sure why, 
                    # but getting DatasetDict with 'train' key every time
        if debug:
            dataset = datasets.Dataset.from_dict(dataset[0:100]) # debug purpose, take 100 rows
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features), num_proc=32)

        if save:
            # save dataset to json format
            dataset_dict = dataset.to_dict()
            with open(split.split(".csv")[0] + "_preprocessed.json", 'w') as f:
                json.dump(dataset_dict, f)

    return dataset


def get_preprocessed_opinionqa_ce_or_wd_loss(
    dataset_config, tokenizer, split, chat_template, save = True,
):

    def tokenize_add_label(sample):
        resp_dist = ast.literal_eval(sample["output_dist"])
        if len(resp_dist) > MAX_MCQ_OPTIONS:
            resp_dist = resp_dist[:MAX_MCQ_OPTIONS]
            s = sum(resp_dist)
            if s > 0:
                resp_dist = [x / s for x in resp_dist]
        resp_dist = resp_dist + [0] * (MAX_MCQ_OPTIONS - len(resp_dist))
        ordinal_info = sample.get("ordinal", None)
        if ordinal_info is not None:
            ordinal_info = ast.literal_eval(ordinal_info)
            if len(ordinal_info) > MAX_MCQ_OPTIONS:
                ordinal_info = ordinal_info[:MAX_MCQ_OPTIONS]
            pad_ord = max(ordinal_info) if ordinal_info else 0
            ordinal_info = ordinal_info + [pad_ord] * (MAX_MCQ_OPTIONS - len(ordinal_info))

        if not chat_template: # using pretrained base model
            prompt = tokenizer.encode(
                _bos_prefix(tokenizer) + sample["input_prompt"],
                add_special_tokens=False
            )
            answer = tokenizer.encode(
                "Answer: A" + _eos_suffix(tokenizer), # "A" is just a placeholder
                add_special_tokens=False
            )[-2:] # [-2:] indicates the option and the eos_token

        else: # using chat model
            # currently only working for the qa steering format
            prompt_split = sample['input_prompt'].split("Answer:")[:-1]
            prompt_split = [x.strip() for x in prompt_split]

            messages = []
            messages.append({
                "role": "user",
                "content": prompt_split[0].strip()
            }) # steering question
            messages.append({
                "role": "assistant",
                "content": prompt_split[1].split("\n")[0].strip()
            }) # steering demographics
            messages.append({
                "role": "user",
                "content": prompt_split[1].replace(messages[1]["content"], "").strip()
            }) # survey question
            prompt = tokenizer.apply_chat_template(
                messages, tokenize = True,
                add_generation_prompt = True
            )
            answer = tokenizer.encode(
                "Answer: A" + _eos_suffix(tokenizer),
                add_special_tokens=False
            )[-2:]
                     
        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "target_token_position": len(prompt),
            "response_distribution": resp_dist
            }
        if ordinal_info is not None:
            sample["ordinal_info"] = ordinal_info
        return sample

    preprocessed_file_dir = (
        split.split(".csv")[0]
        + "_" + tokenizer.name_or_path.split("/")[-1]
        + "_preprocessed.json"
    ) # detail: preprocessing file is dependent on the tokenizer used

    if os.path.exists(preprocessed_file_dir): # if preprocessed file exists
        with open(preprocessed_file_dir, 'r', encoding="utf-8") as f:
            print("preprocessed file exists.")
            content = f.read().strip()
            dataset_dict = json.loads(content)
            dataset = datasets.Dataset.from_dict(dataset_dict)
    else: # if preprocessed file does not exist, preprocess the dataset
        dataset = datasets.load_dataset(
            'csv', 
            data_files = split
        )['train']  # detail: not sure why, 
                    # but getting DatasetDict with 'train' key every time
        dataset = dataset.map(
            tokenize_add_label,
            remove_columns=list(dataset.features),
            num_proc=32
        )

        if save:
            # save dataset to json format
            dataset_dict = dataset.to_dict()
            with Lock():
                with open(preprocessed_file_dir, 'w', encoding='utf-8') as f:
                    json.dump(dataset_dict, f, indent=4)
                    f.flush()

    return dataset