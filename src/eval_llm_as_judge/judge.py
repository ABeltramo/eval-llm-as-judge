"""Multiclass judge client.

Uses the prompts and response-parsing utilities from the TrustyAI Garak fork
(garak.detectors.judge) directly. The OpenAI-compatible client replaces Garak's
internal generator plugin system so the benchmark can target any endpoint without
a full Garak scan context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import OpenAI

from garak.detectors.judge import (
    DEFAULT_MULTICLASS_SYSTEM_PROMPT,
    DEFAULT_MULTICLASS_USER_PROMPT,
    MulticlassJudge as _GarakMulticlassJudge,
)

logger = logging.getLogger(__name__)

CATEGORIES = ("complied", "rejected", "alternative", "other")


@dataclass
class JudgeResult:
    complied: float
    rejected: float
    alternative: float
    other: float
    explanations: dict[str, str]
    raw: str

    @property
    def predicted_category(self) -> str:
        scores = {c: getattr(self, c) for c in CATEGORIES}
        return max(scores, key=scores.__getitem__)

    @property
    def predicted_human_score(self) -> int:
        """Map the four-class prediction to the binary human label convention.

        complied -> 1 (fulfillment), everything else -> 0 (refusal).
        """
        return 1 if self.predicted_category == "complied" else 0


class MulticlassJudge:
    """OpenAI-compatible wrapper around Garak's multiclass judge detector.

    Uses the system/user prompts and JSON parsing from garak.detectors.judge
    verbatim. The Garak detector class itself requires Attempt objects and the
    Garak generator plugin system; this wrapper replaces only that transport
    layer so any OpenAI-compatible endpoint can be benchmarked standalone.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def classify(self, question: str, response: str) -> JudgeResult:
        user_content = DEFAULT_MULTICLASS_USER_PROMPT.format(
            question=question, response=response
        )
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": DEFAULT_MULTICLASS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        raw = completion.choices[0].message.content or ""
        return _parse_result(raw)


def _parse_result(raw: str) -> JudgeResult:
    cleaned = _GarakMulticlassJudge._strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
        return JudgeResult(
            complied=float(data["complied"]["percentage"]),
            rejected=float(data["rejected"]["percentage"]),
            alternative=float(data["alternative"]["percentage"]),
            other=float(data["other"]["percentage"]),
            explanations={c: data[c].get("explanation", "") for c in CATEGORIES},
            raw=raw,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse judge output: %s — %s", exc, raw[:200])
        return JudgeResult(
            complied=0, rejected=0, alternative=0, other=100,
            explanations={c: "" for c in CATEGORIES},
            raw=raw,
        )