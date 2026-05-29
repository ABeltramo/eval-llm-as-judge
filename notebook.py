import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt

    return alt, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # LLM-as-Judge Evaluation
    """)
    return


@app.cell
def _(mo):
    qp = mo.query_params()
    return (qp,)


@app.cell
def _(mo):
    mo.md("""
    ## Judge endpoint
    """)
    return


@app.cell
def _(mo, qp):
    endpoint = mo.ui.text(
        value=qp.get("endpoint", "http://localhost:8000/v1"),
        label="OpenAI-compatible base URL",
        full_width=True,
        on_change=lambda v: qp.set("endpoint", v),
    )
    api_key = mo.ui.text(
        value=qp.get("api_key", ""),
        label="API key",
        kind="password",
        full_width=True,
        on_change=lambda v: qp.set("api_key", v)
    )
    model_name = mo.ui.text(
        value=qp.get("model_name", ""),
        label="Model name",
        placeholder="e.g. mistral-7b-instruct",
        full_width=True,
        on_change=lambda v: qp.set("model_name", v),
    )
    mo.vstack([endpoint, api_key, model_name])
    return api_key, endpoint, model_name


@app.cell
def _(mo):
    mo.md("""
    ## Dataset
    """)
    return


@app.cell
def _(mo, qp):
    split = mo.ui.dropdown(
        options=["test", "train", "all"],
        value=qp.get("split", "test"),
        label="Human judgment split",
        on_change=lambda v: qp.set("split", v),
    )
    prompt_style = mo.ui.dropdown(
        options=["all styles", "base", "translate-fr", "authority_endorsement",
                 "translate-ta", "technical_terms", "evidence-based_persuasion",
                 "uncommon_dialects", "expert_endorsement", "role_play", "caesar",
                 "translate-mr", "ascii", "translate-zh-cn", "misrepresentation",
                 "morse", "logical_appeal", "atbash", "slang", "translate-ml"],
        value=qp.get("prompt_style", "all styles"),
        label="Prompt style",
        on_change=lambda v: qp.set("prompt_style", v),
    )
    max_samples = mo.ui.slider(
        start=0,
        stop=500,
        step=10,
        value=int(qp.get("max_samples", 0)),
        label="Max samples (0 = all)",
        on_change=lambda v: qp.set("max_samples", v),
    )
    workers = mo.ui.slider(
        start=1,
        stop=32,
        step=1,
        value=int(qp.get("workers", 16)),
        label="Parallel workers",
        on_change=lambda v: qp.set("workers", v),
    )
    mo.hstack([split, prompt_style, max_samples, workers])
    return max_samples, prompt_style, split, workers


@app.cell
def _(max_samples, prompt_style, split):
    from eval_llm_as_judge import load_merged_dataset

    _style = None if prompt_style.value == "all styles" else prompt_style.value
    dataset = load_merged_dataset(split=split.value, prompt_style=_style)
    if max_samples.value > 0:
        dataset = dataset.sample(max_samples.value)
    print(f"Loaded {len(dataset)} samples")
    dataset
    return (dataset,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run evaluation")
    run_button
    return (run_button,)


@app.cell
def _(api_key, dataset, endpoint, mo, model_name, pd, run_button, workers):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    mo.stop(not run_button.value, mo.md("Press **Run evaluation** to start."))

    from eval_llm_as_judge import MulticlassJudge, compute_metrics, compute_running_stats

    judge = MulticlassJudge(
        base_url=endpoint.value,
        api_key=api_key.value,
        model=model_name.value,
    )

    records = list(dataset.itertuples(index=False))

    from eval_llm_as_judge.judge import JudgeResult, CATEGORIES
    _error_result = JudgeResult(
        complied=0, rejected=0, alternative=0, other=0,
        parse_error=True,
        raw="",
    )

    errors: list[dict] = []

    def _live_stats_md(results, records):
        s = compute_running_stats(results, records)
        if s is None:
            return mo.md("*Waiting for first results…*")
        return mo.md(f"""
    **Live stats** — {s['n']} / {s['total']} completed

    | Metric | Value |
    |--------|-------|
    | Accuracy | {s['accuracy']:.1%} |
    | Complied recall | {s['complied_recall']:.1%} |
    | Refusal recall | {s['refusal_recall']:.1%} |
    | Parse error rate | {s['parse_error_rate']:.1%} |
    """)

    _update_every = max(1, len(records) // 20)

    with ThreadPoolExecutor(max_workers=workers.value) as executor:
        future_to_idx = {
            executor.submit(judge.classify, row.question, row.response): i
            for i, row in enumerate(records)
        }
        results = [None] * len(records)
        _done_count = 0
        for future in mo.status.progress_bar(
            as_completed(future_to_idx),
            total=len(records),
            title=f"Classifying ({workers.value} workers)",
        ):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as _exc:
                errors.append({"index": idx, "question_id": records[idx].question_id, "error": str(_exc)})
                results[idx] = _error_result
            _done_count += 1
            if _done_count % _update_every == 0 or _done_count == len(records):
                mo.output.replace_at_index(_live_stats_md(results, records), 1)

    if errors:
        mo.output.append(
            mo.callout(
                mo.vstack([
                    mo.md(f"**{len(errors)} request(s) failed** and were counted as `other`. Details:"),
                    mo.ui.table(pd.DataFrame(errors)),
                ]),
                kind="warn",
            )
        )

    rows = [
        {
            "question_id": records[i].question_id,
            "category": records[i].category,
            "category_name": getattr(records[i], "category_name", records[i].category),
            "human_score": records[i].human_score,
            "predicted_category": results[i].predicted_category,
            "predicted_human_score": results[i].predicted_human_score,
            "complied_pct": results[i].complied,
            "rejected_pct": results[i].rejected,
            "alternative_pct": results[i].alternative,
            "other_pct": results[i].other,
            "parse_error": results[i].parse_error,
            "raw_judge_output": results[i].raw,
        }
        for i in range(len(records))
    ]

    results_df = pd.DataFrame(rows)
    results_df
    return compute_metrics, results_df


@app.cell
def _(compute_metrics, mo, results_df):
    metrics = compute_metrics(results_df)

    _pe = metrics.get("parse_error_rate", {})
    _pe_rows = ""
    if _pe:
        _pe_rows = (
            f"| Parse errors (overall) | {_pe['overall']:.1%} | Requests where the judge returned unparseable output |\n"
            f"    | Parse errors (of complied) | {_pe['of_complied']:.1%} | Parse failures as % of all truly-complied samples |\n"
            f"    | Parse errors (of refusal) | {_pe['of_refusal']:.1%} | Parse failures as % of all truly-refused samples |"
        )

    mo.md(f"""
    ## Results

    | Metric | Value | Notes |
    |--------|-------|-------|
    | Overall accuracy | **{metrics['accuracy']:.1%}** | Share of samples where the judge's binary prediction matches the human label |
    | Complied precision | {metrics['classification_report']['complied']['precision']:.1%} | Of samples the judge flagged as complied, how many actually were |
    | Complied recall | {metrics['classification_report']['complied']['recall']:.1%} | Of samples humans labelled as complied, how many the judge caught |
    | Refusal precision | {metrics['classification_report']['refusal']['precision']:.1%} | Of samples the judge flagged as refused, how many actually were |
    | Refusal recall | {metrics['classification_report']['refusal']['recall']:.1%} | Of samples humans labelled as refused, how many the judge caught |
    {_pe_rows}
    """)
    return (metrics,)


@app.cell
def _(alt, metrics, mo, pd):
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [
            {"true": t, "predicted": p, "count": cm[ti][pi]}
            for ti, t in enumerate(["refusal", "complied"])
            for pi, p in enumerate(["refusal", "complied"])
        ]
    )

    heatmap = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", title="Predicted"),
            y=alt.Y("true:N", title="True"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["true", "predicted", "count"],
        )
        .properties(title="Confusion matrix", width=300, height=300)
    )

    text = (
        alt.Chart(cm_df)
        .mark_text(baseline="middle", fontSize=18)
        .encode(
            x="predicted:N",
            y="true:N",
            text="count:Q",
        )
    )

    mo.ui.altair_chart(heatmap + text)
    return


@app.cell
def _(alt, metrics, mo):
    if "per_category" not in metrics:
        mo.stop(True)

    per_cat = metrics["per_category"]
    bar = (
        alt.Chart(per_cat)
        .mark_bar()
        .encode(
            x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Accuracy"),
            y=alt.Y("category:N", sort="-x", title=None),
            color=alt.Color(
                "accuracy:Q",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                legend=None,
            ),
            tooltip=["category", "n", alt.Tooltip("accuracy:Q", format=".1%")],
        )
        .properties(title="Accuracy per safety category", width=500)
    )

    mo.ui.altair_chart(bar)
    return


@app.cell
def _(mo, results_df):
    if "parse_error" not in results_df.columns or not results_df["parse_error"].any():
        mo.stop(True)

    _cols = ["question_id", "category_name", "human_score", "raw_judge_output"]
    _parse_errors_df = (
        results_df[results_df["parse_error"]][_cols]
        .reset_index(drop=True)
    )

    mo.vstack([
        mo.md(f"### Parse error inspection ({len(_parse_errors_df)} samples)"),
        mo.ui.table(_parse_errors_df),
    ])
    return


if __name__ == "__main__":
    app.run()
