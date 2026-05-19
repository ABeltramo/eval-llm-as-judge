"""Load and merge the SORRY-Bench prompt dataset with human judgment annotations."""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def _load_category_map() -> dict[str, str]:
    """Return a dict mapping category ID string -> short category name.

    Names are sourced from meta_info.py in the sorry-bench-202503 dataset repo
    (category_descriptions_short, 0-indexed, category IDs are 1-indexed strings).
    """
    path = hf_hub_download(
        repo_id="sorry-bench/sorry-bench-202406",
        filename="meta_info.py",
        repo_type="dataset",
    )
    ns: dict = {}
    with open(path) as f:
        exec(f.read(), ns)  # noqa: S102
    names: list[str] = ns["category_descriptions_short"]
    return {str(i + 1): name for i, name in enumerate(names)}


CATEGORY_NAMES: dict[str, str] = {}


def load_merged_dataset(split: str = "test", prompt_style: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns: question_id, question, category, model_id, human_score, response.

    Merges sorry-bench-202503 (prompts) with sorry-bench-human-judgment-202406
    (model responses + human binary labels) on question_id and prompt_style.

    Args:
        split: "train", "test", or "all". Controls which split of the human
               judgment dataset is loaded. Use "all" to load all annotations.
        prompt_style: which prompt variant to include. Defaults to None (all styles).
               Pass "base" to restrict to the original un-mutated instructions only.
    """
    global CATEGORY_NAMES
    if not CATEGORY_NAMES:
        CATEGORY_NAMES.update(_load_category_map())

    prompts = _load_prompts(prompt_style=prompt_style)
    judgments = _load_judgments(split, prompt_style=prompt_style)
    merged = judgments.merge(prompts, on=["question_id", "prompt_style"], how="inner")
    merged["category_name"] = merged["category"].map(CATEGORY_NAMES).fillna(merged["category"])
    return merged[["question_id", "prompt_style", "question", "category", "category_name", "model_id", "response", "human_score"]]


def _load_prompts(prompt_style: str | None = "base") -> pd.DataFrame:
    ds = load_dataset("sorry-bench/sorry-bench-202406", split="train")
    df = ds.to_pandas()
    # Extract the harmful instruction from the first turn
    df["question"] = df["turns"].apply(lambda t: t[0] if t else "")
    if prompt_style is not None:
        df = df[df["prompt_style"] == prompt_style]
    return df[["question_id", "prompt_style", "question", "category"]]


def _load_judgments(split: str, prompt_style: str | None = "base") -> pd.DataFrame:
    if split == "all":
        ds = load_dataset("sorry-bench/sorry-bench-human-judgment-202406", split="train+test")
    else:
        ds = load_dataset("sorry-bench/sorry-bench-human-judgment-202406", split=split)
    df = ds.to_pandas()
    # Response text is nested inside choices[0]['turns'][0]
    df["response"] = df["choices"].apply(
        lambda c: c[0]["turns"][0] if c and c[0].get("turns") else ""
    )
    if prompt_style is not None:
        df = df[df["prompt_style"] == prompt_style]
    return df[["question_id", "prompt_style", "model_id", "response", "human_score"]]