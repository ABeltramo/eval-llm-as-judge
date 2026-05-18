# AGENTS.md

Agent instructions for working on this repository.

## Project summary

This repository benchmarks LLMs used as safety classifiers (judges). It merges two HuggingFace datasets — harmful prompts from SORRY-Bench and human-annotated model responses — then runs each `(prompt, response)` pair through a multiclass judge LLM and measures how closely the judge's output agrees with the human labels. The deliverable is a Marimo notebook that can be pointed at any OpenAI-compatible endpoint.

## Repository layout

```
src/eval_llm_as_judge/
  datasets.py   load and merge the two HuggingFace datasets
  judge.py      MulticlassJudge client + Garak prompts
  metrics.py    accuracy, confusion matrix, per-category breakdown
notebook.py     Marimo reactive notebook (entry point)
pyproject.toml  uv project; single source of truth for dependencies
```

## Secrets and environment variables

Both HuggingFace datasets are gated. A valid `HF_TOKEN` must be present in the environment when the notebook loads the datasets. The recommended way to supply it without committing it to git is to add it to `.claude/settings.local.json` (which is gitignored):

```json
{
  "env": {
    "HF_TOKEN": "hf_..."
  }
}
```

Claude Code injects keys declared under `"env"` into every shell it spawns, so `uv run` commands and notebook cells will all see the token automatically.

## Environment

All commands use `uv`. Never use `pip` directly.

```bash
uv sync                        # install / sync dependencies
uv run marimo edit notebook.py # open the notebook
uv run pytest                  # run tests
uv run ruff check src          # lint
uv run ruff format src         # format
```

Python version is pinned in `.python-version` (3.11). Do not change it without updating `pyproject.toml` as well.

## Key constraints

- **No `pip`.** Dependencies go in `pyproject.toml` under `[project] dependencies` (runtime) or `[tool.uv] dev-dependencies` (dev-only). Add them with `uv add <pkg>` or `uv add --dev <pkg>`.
- **Prompts come from Garak directly.** `src/eval_llm_as_judge/judge.py` imports `DEFAULT_MULTICLASS_SYSTEM_PROMPT` and `DEFAULT_MULTICLASS_USER_PROMPT` from `garak.detectors.judge` at runtime — they are not copied into this repo. Do not replace these imports with local string literals. If a prompt needs to change, change it in the Garak fork.
- **Binary label convention.** The human judgment dataset uses `human_score` 1 (fulfillment) and 0 (refusal). The judge's four-class output is collapsed via `JudgeResult.predicted_human_score`: `complied` maps to 1, everything else maps to 0. Do not change this mapping without updating `metrics.py` and the README.
- **No implicit state in the notebook.** Marimo cells must declare all their inputs as function arguments. Do not use global variables or module-level side effects inside `@app.cell` functions.

## Adding or changing functionality

- New helper logic belongs in `src/eval_llm_as_judge/`, not inside `notebook.py`.
- Notebook cells should stay thin: import from the package, bind UI elements, call functions, render outputs.
- If you add a new metric, add it to `compute_metrics` in `metrics.py` and expose it in the results cell of `notebook.py`.
- If you change the dataset loading logic in `datasets.py`, verify that the returned DataFrame still has the columns `question_id`, `question`, `category`, `response`, `human_score` — the notebook and metrics depend on these names.

## Tests

Tests live in `tests/`. Run with `uv run pytest`.

```bash
uv run pytest              # all tests
uv run pytest -v           # verbose (shows each test name)
uv run pytest tests/test_metrics.py   # metrics only
```

Current coverage:

- `tests/test_metrics.py` — 34 tests for `compute_metrics` and `compute_running_stats` in `metrics.py`. Every expected value is derived by hand in the test file. Covers: accuracy, complied/refusal recall and precision, confusion matrix layout, parse error rate breakdown, partial-result handling, NaN for undefined recalls, and a consistency check that `compute_running_stats` on a full result set agrees exactly with `compute_metrics`.

**Constraints:** tests must not call external APIs or download datasets; use `types.SimpleNamespace` or plain fixtures instead of real `JudgeResult` / dataset objects.

## External references

- TrustyAI Garak fork: https://github.com/trustyai-explainability/garak/tree/feature/multiclass-judge
- SORRY-Bench prompts: https://huggingface.co/datasets/sorry-bench/sorry-bench-202503
- SORRY-Bench human judgments: https://huggingface.co/datasets/sorry-bench/sorry-bench-human-judgment-202406
- Marimo docs: https://docs.marimo.io
