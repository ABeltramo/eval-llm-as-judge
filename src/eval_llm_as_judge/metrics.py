"""Classification metrics for judge evaluation results."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute evaluation metrics from a results DataFrame.

    Args:
        df: DataFrame with at minimum columns 'human_score' (int 0/1) and
            'predicted_human_score' (int 0/1). May also include 'predicted_category'
            and 'category' for per-category breakdowns.

    Returns:
        Dictionary with keys: accuracy, classification_report, confusion_matrix,
        and optionally per_category.
    """
    y_true = df["human_score"]
    y_pred = df["predicted_human_score"]

    result: dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["refusal", "complied"],
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    col = "category_name" if "category_name" in df.columns else "category"
    if col in df.columns:
        result["per_category"] = _per_category_accuracy(df, col)

    if "predicted_category" in df.columns:
        result["category_distribution"] = (
            df["predicted_category"].value_counts(normalize=True).to_dict()
        )

    if "parse_error" in df.columns:
        result["parse_error_rate"] = _parse_error_rates(df)

    return result


def _parse_error_rates(df: pd.DataFrame) -> dict:
    """Parse error rate as a fraction of each true-label class.

    Returns the proportion of samples that failed to parse, broken down by
    true label (complied / refusal), so it can be read alongside recall figures.
    """
    errors = df[df["parse_error"]]
    n_complied = (df["human_score"] == 1).sum()
    n_refusal = (df["human_score"] == 0).sum()
    return {
        "overall": len(errors) / max(len(df), 1),
        "of_complied": (errors["human_score"] == 1).sum() / max(n_complied, 1),
        "of_refusal": (errors["human_score"] == 0).sum() / max(n_refusal, 1),
    }


def compute_running_stats(results: list, records) -> dict | None:
    """Compute live accuracy/recall/parse-error stats from a partially-filled results list.

    Args:
        results: list of JudgeResult | None, same length as records. None means not yet done.
        records: iterable of objects with a .human_score attribute.

    Returns:
        Dict with keys n, total, accuracy, complied_recall, refusal_recall, parse_error_rate,
        or None if no results have arrived yet.
    """
    records = list(records)
    done = [i for i in range(len(records)) if results[i] is not None]
    n = len(done)
    if n == 0:
        return None
    y_true = [records[i].human_score for i in done]
    y_pred = [results[i].predicted_human_score for i in done]
    n_complied_true = sum(1 for y in y_true if y == 1)
    n_refusal_true = sum(1 for y in y_true if y == 0)
    return {
        "n": n,
        "total": len(records),
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / n,
        "complied_recall": (
            sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1) / n_complied_true
            if n_complied_true else float("nan")
        ),
        "refusal_recall": (
            sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0) / n_refusal_true
            if n_refusal_true else float("nan")
        ),
        "parse_error_rate": sum(1 for i in done if results[i].parse_error) / n,
    }


def _per_category_accuracy(df: pd.DataFrame, col: str = "category") -> pd.DataFrame:
    rows = []
    for category, group in df.groupby(col):
        acc = accuracy_score(group["human_score"], group["predicted_human_score"])
        rows.append({"category": category, "n": len(group), "accuracy": acc})
    return pd.DataFrame(rows).sort_values("accuracy")
