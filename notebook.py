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
    mo.md("""
    ## Judge endpoint
    """)
    return


@app.cell
def _(mo):
    endpoint = mo.ui.text(
        value="http://localhost:8000/v1",
        label="OpenAI-compatible base URL",
        full_width=True,
    )
    api_key = mo.ui.text(
        value="none",
        label="API key",
        kind="password",
        full_width=True,
    )
    model_name = mo.ui.text(
        value="",
        label="Model name",
        placeholder="e.g. mistral-7b-instruct",
        full_width=True,
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
def _(mo):
    split = mo.ui.dropdown(
        options=["test", "train", "all"],
        value="test",
        label="Human judgment split",
    )
    prompt_style = mo.ui.dropdown(
        options=["all styles", "base", "translate-fr", "authority_endorsement",
                 "translate-ta", "technical_terms", "evidence-based_persuasion",
                 "uncommon_dialects", "expert_endorsement", "role_play", "caesar",
                 "translate-mr", "ascii", "translate-zh-cn", "misrepresentation",
                 "morse", "logical_appeal", "atbash", "slang", "translate-ml"],
        value="all styles",
        label="Prompt style",
    )
    max_samples = mo.ui.slider(
        start=0,
        stop=500,
        step=10,
        value=0,
        label="Max samples (0 = all)",
    )
    workers = mo.ui.slider(
        start=1,
        stop=32,
        step=1,
        value=16,
        label="Parallel workers",
    )
    mo.hstack([split, prompt_style, max_samples, workers])
    return max_samples, prompt_style, split, workers


@app.cell
def _(max_samples, prompt_style, split):
    from eval_llm_as_judge import load_merged_dataset

    _style = None if prompt_style.value == "all styles" else prompt_style.value
    dataset = load_merged_dataset(split=split.value, prompt_style=_style)
    if max_samples.value > 0:
        dataset = dataset.head(max_samples.value)
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

    from eval_llm_as_judge import MulticlassJudge, compute_metrics

    judge = MulticlassJudge(
        base_url=endpoint.value,
        api_key=api_key.value,
        model=model_name.value,
    )

    records = list(dataset.itertuples(index=False))

    from eval_llm_as_judge.judge import JudgeResult, CATEGORIES
    _error_result = JudgeResult(
        complied=0, rejected=0, alternative=0, other=100,
        explanations={c: "error" for c in CATEGORIES}, raw="",
    )

    errors: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers.value) as executor:
        future_to_idx = {
            executor.submit(judge.classify, row.question, row.response): i
            for i, row in enumerate(records)
        }
        results = [None] * len(records)
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
        }
        for i in range(len(records))
    ]

    results_df = pd.DataFrame(rows)
    results_df
    return compute_metrics, results_df


@app.cell
def _(compute_metrics, mo, results_df):
    metrics = compute_metrics(results_df)

    mo.md(f"""
    ## Results

    | Metric | Value | Notes |
    |--------|-------|-------|
    | Overall accuracy | **{metrics['accuracy']:.1%}** | Share of samples where the judge's binary prediction matches the human label |
    | Acceptance precision | {metrics['classification_report']['acceptance']['precision']:.1%} | Of samples the judge flagged as accepted, how many actually were |
    | Acceptance recall | {metrics['classification_report']['acceptance']['recall']:.1%} | Of samples humans labelled as accepted, how many the judge caught |
    | Refusal precision | {metrics['classification_report']['refusal']['precision']:.1%} | Of samples the judge flagged as refused, how many actually were |
    | Refusal recall | {metrics['classification_report']['refusal']['recall']:.1%} | Of samples humans labelled as refused, how many the judge caught |
    """)
    return (metrics,)


@app.cell
def _(alt, metrics, mo, pd):
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [
            {"true": t, "predicted": p, "count": cm[ti][pi]}
            for ti, t in enumerate(["refusal", "acceptance"])
            for pi, p in enumerate(["refusal", "acceptance"])
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


if __name__ == "__main__":
    app.run()
