import ast
import argparse
import os
import pathlib
import random
import sys
from enum import Enum
from typing import Callable, List, Optional, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm

from subpop.survey.config import SteeringPromptType
from subpop.utils.survey_utils import generate_mcq, list_normalize
from subpop.utils.random_utils import set_random_seed


def code_subgroup(attribute: str, subgroup: str) -> str:
    """ Code subgroup for generation of steering prompt. """
    if attribute == 'CITIZEN':
        subgroup_coded = 'Yes' if subgroup == 'a US Citizen' else 'No'
    elif attribute == 'MARITAL':
        subgroup_coded = (
            'Never been married' if subgroup == 'Unmarried and have never been married'
            else subgroup
        )
    elif attribute == 'POLPARTY':
        subgroup_coded = 'Other' if subgroup == 'Something else' else subgroup
    else:
        subgroup_coded = subgroup
    return subgroup_coded


def _read_csv_with_encoding(path: pathlib.Path) -> "pd.DataFrame":
    """Windows 上部分 CSV 为系统默认编码（如 GBK）；统一尝试常见编码。"""
    path = pathlib.Path(path)
    last_err: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "cp936"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return pd.read_csv(path)


def prepare_data(
    survey_file_path: pathlib.Path,
    steering_prompts_file_path: pathlib.Path,
    steering_demographics_file_path: pathlib.Path,
    steering_prompt_type: SteeringPromptType = SteeringPromptType.QA,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    test_wave: Optional[List[int]] = [],
    subgroup_coder: Optional[Callable[[str, str], str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare data for training and evaluation of fine-tuned language model.
    Args:
        survey_file_path (pathlib.Path): Path to survey file
        steering_prompts_file_path (pathlib.Path): Path to steering prompts file
        steering_demographics_file_path (pathlib.Path): Path to steering demographics file
        steering_prompt_type (SteeringPromptType): Type of steering prompt
        train, val, test_ratio (float): Ratios for train, validation and test splits
        test_wave (List[int]): List of wave numbers dedicated for test
        subgroup_coder: Optional callable (attribute, subgroup) -> label used in survey CSV
            "group" column. Use ``lambda _a, s: s`` (or CLI ``--subgroup_coder identity``) when
            labels already match (e.g. CGSS Chinese metadata).
    Returns:
        Tuple of train, val, test DataFrames with input prompts, output tokens and output distribution
    Note:
        output_dist does not contain refusal option, instead saving normalized distribution without refusal option
    """

    assert train_ratio >= 0 and val_ratio >= 0 and test_ratio >= 0, "Ratios should be non-negative."
    assert sum([train_ratio, val_ratio, test_ratio]) == 1, "Ratios should sum up to 1."

    def _resolve_subgroup(attribute: str, subgroup: str) -> str:
        if subgroup_coder is not None:
            return subgroup_coder(attribute, subgroup)
        return code_subgroup(attribute, subgroup)

    full_survey_df: pd.DataFrame = _read_csv_with_encoding(survey_file_path)
    steering_prompts_df: pd.DataFrame = pd.read_json(steering_prompts_file_path, encoding="utf-8")
    steering_demographics_df: pd.DataFrame = _read_csv_with_encoding(steering_demographics_file_path)

    full_survey_df["responses"] = full_survey_df["responses"].apply(ast.literal_eval)
    full_survey_df["ordinal"] = full_survey_df["ordinal"].apply(ast.literal_eval)
    full_survey_df["options"] = full_survey_df["options"].apply(ast.literal_eval)

    steering_prompts_df["options"] = steering_prompts_df["options"].apply(ast.literal_eval)
    steering_demographics_df["group"] = steering_demographics_df["group"].apply(ast.literal_eval)
    steering_demographics_df = steering_demographics_df.explode("group")

    unique_qkeys: List[str] = full_survey_df["qkey"].unique()
    test_dedicated_qkeys: List[str] = [
        qkey for qkey in unique_qkeys if any(f"_W{wave}" in qkey for wave in test_wave)
    ] # qkeys belonging to particular waves that are dedicated for test
    remaining_qkeys: List[str] = [
        qkey for qkey in unique_qkeys if qkey not in test_dedicated_qkeys
    ] # qkeys that are not dedicated for test

    # shuffle keys and split into three datasets
    np.random.shuffle(remaining_qkeys)
    num_train_questions = int(round(train_ratio * len(remaining_qkeys)))
    num_val_questions = int(round(val_ratio * len(remaining_qkeys)))
    train_questions = remaining_qkeys[:num_train_questions]
    val_questions = remaining_qkeys[num_train_questions:num_train_questions+num_val_questions]
    test_questions = remaining_qkeys[num_train_questions+num_val_questions:] + test_dedicated_qkeys

    train_survey_df = full_survey_df[full_survey_df["qkey"].isin(train_questions)]
    val_survey_df = full_survey_df[full_survey_df["qkey"].isin(val_questions)]
    test_survey_df = full_survey_df[full_survey_df["qkey"].isin(test_questions)]
    
    train_data_list: List[pd.DataFrame] = []
    val_data_list: List[pd.DataFrame] = []
    test_data_list: List[pd.DataFrame] = []

    # iterate over each (attribute, subgroup) and (train, val, test) split
    # to generate three lists of dataframes, {train, val, test}_data_list
    for _, subgroup_row in tqdm(steering_demographics_df.iterrows()):
        attribute: str = subgroup_row["attribute"]
        subgroup: str = subgroup_row["group"]
        subgroup_coded = _resolve_subgroup(attribute, subgroup)

        for i, survey_df in enumerate([train_survey_df, val_survey_df, test_survey_df]):
            survey_subgroup_df: pd.DataFrame = survey_df[
                (survey_df["attribute"] == attribute)
                & (survey_df["group"] == subgroup_coded)
            ].reset_index(drop=True)
            data_df: pd.DataFrame = pd.DataFrame(
                columns=["qkey", "input_prompt", "output_token", "output_dist"]
            )

            for steering_prompt_type_str in steering_prompt_type.value:
                
                # steering prompt generation
                steering_prompt: str = steering_prompts_df[
                    steering_prompts_df.attribute == attribute
                ][steering_prompt_type_str].values[0]
                steering_options: list = steering_prompts_df[
                    steering_prompts_df.attribute == attribute
                ]["options"].values[0]

                assert (
                    subgroup in steering_options
                ), f"Subgroup {subgroup} not found in steering options"

                if steering_prompt_type_str == SteeringPromptType.QA.value[0]:
                    # QA steering prompt generation
                    steering_prompt = generate_mcq(
                        question_body=steering_prompt, options=steering_options
                    )
                    idx = steering_options.index(subgroup)
                    steering_prompt += f" {chr(ord('A') + idx)}. {subgroup}\n\n"
                    steering_prompt += "Answer the following question keeping in mind your previous answers.\n"

                else:
                    # BIO and PORTRAY steering prompt: does not require mcq generation
                    steering_prompt = ".\n".join(steering_prompt.split(". "))
                    steering_prompt += f" {subgroup}.\n\n"

                # survey prompt generation
                survey_prompt_series: pd.core.series.Series = survey_subgroup_df.apply(
                    lambda row: generate_mcq(
                        question_body=row.question,
                        options=row.options,
                        add_answer_forcing=True,
                    ),
                    axis=1,
                )

                # concatenation of steering and survey prompts
                if survey_prompt_series.empty:
                    continue
                data_df["input_prompt"] = steering_prompt + survey_prompt_series

                # augmentation of one-hot responses (ablation study for distribution modeling)
                response_dist_with_refusal: pd.core.series.Series = (
                    survey_subgroup_df.apply(
                        lambda row: row.responses + [row.refusal_rate], axis=1
                    )
                )
                response_samples: pd.core.series.Series = response_dist_with_refusal.apply(
                    lambda x: random.choices(range(len(x)), weights=x, k=100)
                )
                data_df["output_token"] = response_samples.apply(
                    lambda x: [f" {chr(ord('A') + i)}" for i in x]
                )
                data_df["output_dist"] = response_dist_with_refusal
                data_df["qkey"] = survey_subgroup_df["qkey"]
                data_df["attribute"] = attribute
                data_df["group"] = subgroup
                data_df["ordinal"] = survey_subgroup_df["ordinal"]

                if i == 0:
                    train_data_list.append(data_df.copy())
                elif i == 1:
                    val_data_list.append(data_df.copy())
                else:
                    test_data_list.append(data_df.copy())

    train_data_df = (
        pd.concat(train_data_list).reset_index(drop=True)
        if train_data_list else pd.DataFrame()
    )
    val_data_df = (
        pd.concat(val_data_list).reset_index(drop=True)
        if val_data_list else pd.DataFrame()
    )
    test_data_df = (
        pd.concat(test_data_list).reset_index(drop=True)
        if test_data_list else pd.DataFrame()
    )
    return train_data_df, val_data_df, test_data_df


def get_args_datagen():
    parser = argparse.ArgumentParser(
        description="Data Generation for Finetuning and Evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str, default="subpop-train",
        help="Dataset name",
    )
    parser.add_argument(
        "--steer_prompts_file_path",
        type=str, default=REPO_ROOT / "data" / "subpopulation_metadata" / "steering_prompts.json",
        help="Steer prompts file path",
    )
    parser.add_argument(
        "--steer_demographics_file_path",
        type=str, default=REPO_ROOT / "data" / "subpopulation_metadata" / "demographics_22.csv",
        help="Steer demographics file path",
    )
    parser.add_argument(
        "--train_ratio",
        type=float, default=0.9,
        help="Train split ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float, default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float, default=0.0,
        help="Test split ratio"
    )
    parser.add_argument(
        "--test_wave",
        type=int, nargs="+", default=[],
        help="Wave numbers dedicated for test. Used when wants to spare a particular wave for test."
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Flag to not shuffle data. Used when wants to keep the order of data."
    )
    parser.add_argument(
        "--subgroup_coder",
        type=str,
        default="default",
        choices=["default", "identity"],
        help="identity: use demographics subgroup label as-is in CSV (e.g. CGSS Chinese labels). "
        "default: use built-in code_subgroup mapping for SubPOP US labels.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args_datagen()

    dataset_name = args.dataset
    output_dir = REPO_ROOT / "data" / dataset_name / "processed"
    response_distribution_file_path = (
        REPO_ROOT / "data" / dataset_name / "processed" / f"{dataset_name}.csv"
    )
    steer_prompts_file_path = args.steer_prompts_file_path
    steer_demographics_file_path = args.steer_demographics_file_path

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    test_wave = args.test_wave

    seed = args.seed
    no_shuffle = args.no_shuffle

    if not pathlib.Path(output_dir).exists():
        os.makedirs(output_dir)

    subgroup_coder_fn = None
    if args.subgroup_coder == "identity":
        subgroup_coder_fn = lambda _a, s: s

    """For each steering prompt type, generate a train / validation / test split."""
    for steer_type in SteeringPromptType:
        set_random_seed(seed) # set the same seed for each steering prompt type.
        train_df, val_df, test_df = prepare_data(
            steering_prompt_type=steer_type,
            survey_file_path=response_distribution_file_path,
            steering_prompts_file_path=steer_prompts_file_path,
            steering_demographics_file_path=steer_demographics_file_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            test_wave=test_wave,
            subgroup_coder=subgroup_coder_fn,
        )
        # test_ratio=0 等情况下 test_df 可能无列，to_csv 会得到无法被 pandas/datasets 读取的空文件
        if len(train_df.columns) > 0:
            if len(val_df.columns) == 0:
                val_df = train_df.iloc[0:0].copy()
            if len(test_df.columns) == 0:
                test_df = train_df.iloc[0:0].copy()
        if not no_shuffle:
            train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_df.to_csv(
            os.path.join(output_dir, f"opnqa_{steer_type.name}_train.csv"), index=False, encoding="utf-8"
        )
        val_df.to_csv(
            os.path.join(output_dir, f"opnqa_{steer_type.name}_val.csv"), index=False, encoding="utf-8"
        )
        test_df.to_csv(
            os.path.join(output_dir, f"opnqa_{steer_type.name}_test.csv"), index=False, encoding="utf-8"
        )