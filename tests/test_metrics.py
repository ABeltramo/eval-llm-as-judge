"""Tests for metrics.py — every expected value is derived by hand below each case."""

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from eval_llm_as_judge.metrics import compute_metrics, compute_running_stats

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_df(human_scores, predicted_scores, parse_errors=None, category_names=None):
    d = {
        "human_score": human_scores,
        "predicted_human_score": predicted_scores,
    }
    if parse_errors is not None:
        d["parse_error"] = parse_errors
    if category_names is not None:
        d["category_name"] = category_names
    return pd.DataFrame(d)


def make_result(predicted_human_score, parse_error=False):
    """Minimal stand-in for JudgeResult."""
    return SimpleNamespace(predicted_human_score=predicted_human_score, parse_error=parse_error)


def make_record(human_score):
    return SimpleNamespace(human_score=human_score)


# ---------------------------------------------------------------------------
# compute_metrics — accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_perfect_classifier(self):
        # [0,0,1,1] vs [0,0,1,1] → 4/4 correct
        df = make_df([0, 0, 1, 1], [0, 0, 1, 1])
        assert compute_metrics(df)["accuracy"] == pytest.approx(1.0)

    def test_all_wrong(self):
        # [0,0,1,1] vs [1,1,0,0] → 0/4 correct
        df = make_df([0, 0, 1, 1], [1, 1, 0, 0])
        assert compute_metrics(df)["accuracy"] == pytest.approx(0.0)

    def test_always_predict_refusal(self):
        # [0,0,1,1] vs [0,0,0,0] → 2/4 correct
        df = make_df([0, 0, 1, 1], [0, 0, 0, 0])
        assert compute_metrics(df)["accuracy"] == pytest.approx(0.5)

    def test_always_predict_complied(self):
        # [0,0,1,1] vs [1,1,1,1] → 2/4 correct
        df = make_df([0, 0, 1, 1], [1, 1, 1, 1])
        assert compute_metrics(df)["accuracy"] == pytest.approx(0.5)

    def test_known_values(self):
        # y_true=[0,0,0,0,0,0,1,1,1,1], y_pred=[0,0,0,0,0,1,1,1,0,0]
        # correct: indices 0–4 (5 refusals), 6,7 (2 complied) → 7/10
        df = make_df(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        )
        assert compute_metrics(df)["accuracy"] == pytest.approx(7 / 10)


# ---------------------------------------------------------------------------
# compute_metrics — complied recall & precision
# Label convention: 1 = complied, 0 = refusal
# complied_recall  = TP / (TP + FN)  — of actual complied, how many caught
# complied_precision = TP / (TP + FP) — of predicted complied, how many right
# ---------------------------------------------------------------------------


class TestCompliedRecallAndPrecision:
    def test_perfect(self):
        # TP=2, FN=0  → recall=1.0; TP=2, FP=0 → precision=1.0
        df = make_df([0, 0, 1, 1], [0, 0, 1, 1])
        cr = compute_metrics(df)["classification_report"]["complied"]
        assert cr["recall"] == pytest.approx(1.0)
        assert cr["precision"] == pytest.approx(1.0)

    def test_always_refusal_null_classifier(self):
        # y_true=[0,0,1,1], y_pred=[0,0,0,0]
        # TP=0, FN=2 → recall=0/2=0; TP=0, FP=0 → precision=0 (sklearn returns 0)
        df = make_df([0, 0, 1, 1], [0, 0, 0, 0])
        cr = compute_metrics(df)["classification_report"]["complied"]
        assert cr["recall"] == pytest.approx(0.0)

    def test_always_complied_null_classifier(self):
        # y_true=[0,0,1,1], y_pred=[1,1,1,1]
        # TP=2, FN=0 → recall=2/2=1.0
        # TP=2, FP=2 → precision=2/4=0.5
        df = make_df([0, 0, 1, 1], [1, 1, 1, 1])
        cr = compute_metrics(df)["classification_report"]["complied"]
        assert cr["recall"] == pytest.approx(1.0)
        assert cr["precision"] == pytest.approx(0.5)

    def test_known_values(self):
        # y_true=[0,0,0,0,0,0,1,1,1,1], y_pred=[0,0,0,0,0,1,1,1,0,0]
        # TP=2 (idx 6,7), FN=2 (idx 8,9), FP=1 (idx 5)
        # recall=2/(2+2)=0.5, precision=2/(2+1)=2/3
        df = make_df(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        )
        cr = compute_metrics(df)["classification_report"]["complied"]
        assert cr["recall"] == pytest.approx(2 / 4)
        assert cr["precision"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# compute_metrics — refusal recall & precision
# refusal_recall    = TN / (TN + FP)  — of actual refusals, how many caught
# refusal_precision = TN / (TN + FN)  — of predicted refusals, how many right
# ---------------------------------------------------------------------------


class TestRefusalRecallAndPrecision:
    def test_perfect(self):
        df = make_df([0, 0, 1, 1], [0, 0, 1, 1])
        cr = compute_metrics(df)["classification_report"]["refusal"]
        assert cr["recall"] == pytest.approx(1.0)
        assert cr["precision"] == pytest.approx(1.0)

    def test_always_refusal_null_classifier(self):
        # y_true=[0,0,1,1], y_pred=[0,0,0,0]
        # TN=2, FP=0 → recall=2/2=1.0
        # TN=2, FN=2 → precision=2/4=0.5
        df = make_df([0, 0, 1, 1], [0, 0, 0, 0])
        cr = compute_metrics(df)["classification_report"]["refusal"]
        assert cr["recall"] == pytest.approx(1.0)
        assert cr["precision"] == pytest.approx(0.5)

    def test_always_complied_null_classifier(self):
        # TN=0, FP=2 → recall=0/2=0.0
        df = make_df([0, 0, 1, 1], [1, 1, 1, 1])
        cr = compute_metrics(df)["classification_report"]["refusal"]
        assert cr["recall"] == pytest.approx(0.0)

    def test_known_values(self):
        # y_true=[0,0,0,0,0,0,1,1,1,1], y_pred=[0,0,0,0,0,1,1,1,0,0]
        # TN=5 (idx 0–4), FP=1 (idx 5), FN=2 (idx 8,9)
        # recall=5/(5+1)=5/6, precision=5/(5+2)=5/7
        df = make_df(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        )
        cr = compute_metrics(df)["classification_report"]["refusal"]
        assert cr["recall"] == pytest.approx(5 / 6)
        assert cr["precision"] == pytest.approx(5 / 7)


# ---------------------------------------------------------------------------
# compute_metrics — confusion matrix layout
# sklearn with target_names=["refusal", "complied"] (label 0 first, label 1 second)
# cm[0][0]=TN, cm[0][1]=FP, cm[1][0]=FN, cm[1][1]=TP
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_perfect_classifier(self):
        df = make_df([0, 0, 1, 1], [0, 0, 1, 1])
        # TN=2, FP=0, FN=0, TP=2
        assert compute_metrics(df)["confusion_matrix"] == [[2, 0], [0, 2]]

    def test_always_predict_refusal(self):
        df = make_df([0, 0, 1, 1], [0, 0, 0, 0])
        # TN=2, FP=0, FN=2, TP=0
        assert compute_metrics(df)["confusion_matrix"] == [[2, 0], [2, 0]]

    def test_always_predict_complied(self):
        df = make_df([0, 0, 1, 1], [1, 1, 1, 1])
        # TN=0, FP=2, FN=0, TP=2
        assert compute_metrics(df)["confusion_matrix"] == [[0, 2], [0, 2]]

    def test_known_values(self):
        df = make_df(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        )
        # TN=5, FP=1, FN=2, TP=2
        assert compute_metrics(df)["confusion_matrix"] == [[5, 1], [2, 2]]


# ---------------------------------------------------------------------------
# compute_metrics — parse error rates
# overall       = parse_errors / N
# of_complied   = parse_errors where human_score==1 / total human_score==1
# of_refusal    = parse_errors where human_score==0 / total human_score==0
# ---------------------------------------------------------------------------


class TestParseErrorRates:
    def setup_method(self):
        # 10 samples: 4 complied (human=1), 6 refusal (human=0)
        # parse errors: 2 among complied, 1 among refusal → 3 total
        # overall = 3/10, of_complied = 2/4, of_refusal = 1/6
        human =  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        pred =   [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        errors = [False, False, True, True, False, False, False, False, False, True]
        self.metrics = compute_metrics(make_df(human, pred, parse_errors=errors))

    def test_overall(self):
        assert self.metrics["parse_error_rate"]["overall"] == pytest.approx(3 / 10)

    def test_of_complied(self):
        assert self.metrics["parse_error_rate"]["of_complied"] == pytest.approx(2 / 4)

    def test_of_refusal(self):
        assert self.metrics["parse_error_rate"]["of_refusal"] == pytest.approx(1 / 6)

    def test_no_parse_errors(self):
        df = make_df([0, 1], [0, 1], parse_errors=[False, False])
        pe = compute_metrics(df)["parse_error_rate"]
        assert pe["overall"] == pytest.approx(0.0)
        assert pe["of_complied"] == pytest.approx(0.0)
        assert pe["of_refusal"] == pytest.approx(0.0)

    def test_all_parse_errors(self):
        # 2 complied, 2 refusal, all errored
        df = make_df([0, 0, 1, 1], [0, 0, 0, 0], parse_errors=[True, True, True, True])
        pe = compute_metrics(df)["parse_error_rate"]
        assert pe["overall"] == pytest.approx(1.0)
        assert pe["of_complied"] == pytest.approx(1.0)
        assert pe["of_refusal"] == pytest.approx(1.0)

    def test_absent_when_column_missing(self):
        df = make_df([0, 1], [0, 1])
        assert "parse_error_rate" not in compute_metrics(df)


# ---------------------------------------------------------------------------
# compute_running_stats
# ---------------------------------------------------------------------------


class TestComputeRunningStats:
    def test_all_none_returns_none(self):
        results = [None, None, None]
        records = [make_record(0), make_record(1), make_record(0)]
        assert compute_running_stats(results, records) is None

    def test_empty_lists_returns_none(self):
        assert compute_running_stats([], []) is None

    def test_n_and_total(self):
        # 2 of 4 done
        records = [make_record(h) for h in [0, 1, 0, 1]]
        results = [make_result(0), make_result(1), None, None]
        s = compute_running_stats(results, records)
        assert s["n"] == 2
        assert s["total"] == 4

    def test_perfect_partial_results(self):
        # done: true=0→pred=0, true=1→pred=1 — both correct
        # accuracy=1.0, complied_recall=1.0, refusal_recall=1.0, parse_error_rate=0.0
        records = [make_record(0), make_record(1), make_record(0), make_record(1)]
        results = [make_result(0), make_result(1), None, None]
        s = compute_running_stats(results, records)
        assert s["accuracy"] == pytest.approx(1.0)
        assert s["complied_recall"] == pytest.approx(1.0)
        assert s["refusal_recall"] == pytest.approx(1.0)
        assert s["parse_error_rate"] == pytest.approx(0.0)

    def test_known_values_full(self):
        # Same case as TestCompliedRecallAndPrecision.test_known_values
        # y_true=[0,0,0,0,0,0,1,1,1,1], y_pred=[0,0,0,0,0,1,1,1,0,0]
        # accuracy=7/10, complied_recall=2/4, refusal_recall=5/6
        records = [make_record(h) for h in [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]
        results = [make_result(p) for p in [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]]
        s = compute_running_stats(results, records)
        assert s["accuracy"] == pytest.approx(7 / 10)
        assert s["complied_recall"] == pytest.approx(2 / 4)
        assert s["refusal_recall"] == pytest.approx(5 / 6)

    def test_only_considers_done_entries(self):
        # records: [true=1, true=0, true=1, true=0]
        # results: [pred=1, None, pred=0, None]
        # done indices: 0 (true=1,pred=1 ✓), 2 (true=1,pred=0 ✗)
        # accuracy=1/2=0.5, complied_recall=1/2=0.5, refusal_recall=nan
        records = [make_record(h) for h in [1, 0, 1, 0]]
        results = [make_result(1), None, make_result(0), None]
        s = compute_running_stats(results, records)
        assert s["n"] == 2
        assert s["accuracy"] == pytest.approx(0.5)
        assert s["complied_recall"] == pytest.approx(0.5)
        assert math.isnan(s["refusal_recall"])

    def test_no_complied_in_done_gives_nan_complied_recall(self):
        # Only refusal records completed
        records = [make_record(0), make_record(1), make_record(0)]
        results = [make_result(0), None, make_result(0)]
        s = compute_running_stats(results, records)
        assert math.isnan(s["complied_recall"])
        assert s["refusal_recall"] == pytest.approx(1.0)

    def test_no_refusal_in_done_gives_nan_refusal_recall(self):
        # Only complied records completed
        records = [make_record(0), make_record(1), make_record(1)]
        results = [None, make_result(1), make_result(0)]
        s = compute_running_stats(results, records)
        assert math.isnan(s["refusal_recall"])
        assert s["complied_recall"] == pytest.approx(0.5)

    def test_parse_error_rate(self):
        # 5 done: 2 are parse errors
        records = [make_record(h) for h in [0, 0, 1, 0, 1]]
        results = [
            make_result(0, parse_error=False),
            make_result(0, parse_error=True),
            make_result(0, parse_error=True),
            make_result(0, parse_error=False),
            make_result(1, parse_error=False),
        ]
        s = compute_running_stats(results, records)
        assert s["parse_error_rate"] == pytest.approx(2 / 5)

    def test_parse_error_rate_none_errors(self):
        records = [make_record(0), make_record(1)]
        results = [make_result(0), make_result(1)]
        s = compute_running_stats(results, records)
        assert s["parse_error_rate"] == pytest.approx(0.0)

    def test_consistency_with_compute_metrics(self):
        # Running stats on a complete result set must agree with compute_metrics
        # on accuracy, complied_recall, and refusal_recall
        human = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        pred =  [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
        records = [make_record(h) for h in human]
        results = [make_result(p) for p in pred]

        s = compute_running_stats(results, records)
        m = compute_metrics(make_df(human, pred))

        assert s["accuracy"] == pytest.approx(m["accuracy"])
        assert s["complied_recall"] == pytest.approx(
            m["classification_report"]["complied"]["recall"]
        )
        assert s["refusal_recall"] == pytest.approx(
            m["classification_report"]["refusal"]["recall"]
        )
