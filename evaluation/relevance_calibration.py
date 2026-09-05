from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    sample_count: int


def calibrate_threshold(samples: list[dict]) -> CalibrationResult:
    """Choose the highest-precision threshold among those with maximum F1."""
    if not samples or not any(item["relevant"] for item in samples) or not any(
        not item["relevant"] for item in samples
    ):
        raise ValueError("Calibration requires both relevant and irrelevant samples.")

    scores = sorted({float(item["score"]) for item in samples})
    candidates = [scores[0] - 1.0]
    candidates.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    candidates.append(scores[-1] + 1.0)
    results = []
    for threshold in candidates:
        true_positive = sum(
            bool(item["relevant"]) and float(item["score"]) >= threshold
            for item in samples
        )
        false_positive = sum(
            not bool(item["relevant"]) and float(item["score"]) >= threshold
            for item in samples
        )
        false_negative = sum(
            bool(item["relevant"]) and float(item["score"]) < threshold
            for item in samples
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append((f1, precision, recall, threshold))

    f1, precision, recall, threshold = max(results)
    return CalibrationResult(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        sample_count=len(samples),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the corpus relevance threshold.")
    parser.add_argument("input", type=Path, help="JSON file containing model_name and labeled scores.")
    parser.add_argument("output", type=Path, help="Threshold configuration to create or update.")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = calibrate_threshold(payload["samples"])
    output = {"models": {}}
    if args.output.exists():
        output = json.loads(args.output.read_text(encoding="utf-8"))
        output.setdefault("models", {})
    output["models"][payload["model_name"]] = {
        **asdict(result),
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["models"][payload["model_name"]], indent=2))


if __name__ == "__main__":
    main()
