"""Recompute the paper's Qwen3-4B AUROC and AUARC from released rollouts."""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import importlib.util
import json
import re
from functools import lru_cache
from pathlib import Path

from sauce import sauce_score

# Paper's Qwen3-4B rows, in percentage units (AUROC, AUARC).
PAPER_RESULTS = {
    ("math_500", "debate"): (500, 68.81, 84.12),
    ("math_500", "dylan"): (500, 73.11, 85.60),
    ("mmlu_pro", "debate"): (12032, 77.05, 82.72),
    ("mmlu_pro", "dylan"): (12032, 78.14, 83.37),
    ("bbh", "debate"): (1250, 73.68, 89.09),
    ("bbh", "dylan"): (1250, 70.18, 87.93),
}


@lru_cache(maxsize=200_000)
def parse_math(text: str):
    from math_verify import parse

    try:
        return parse(text.replace(r"\%", "%").strip())
    except Exception:
        return None


def normalize_choice(text: str) -> str:
    text = text.strip().upper()
    wrapper = re.fullmatch(r"\\(?:TEXT|MATHRM|BOXED)\{([A-J])\}", text)
    if wrapper:
        return wrapper.group(1)
    if re.fullmatch(r"\([A-J]\)", text):
        return text[1]
    return text


def answer_correct(prediction: str, reference: str) -> bool:
    """Use the paper's letter matching and math-verify answer-checking policy."""
    prediction, reference = prediction.strip(), reference.strip()
    if not prediction or not reference:
        return False
    if re.fullmatch(r"[A-J]|\([A-J]\)", reference.upper()):
        return normalize_choice(prediction) == normalize_choice(reference)
    from math_verify import verify

    parsed_prediction, parsed_reference = parse_math(prediction), parse_math(reference)
    if parsed_prediction is None or parsed_reference is None:
        return False
    try:
        return bool(verify(parsed_reference, parsed_prediction, strict=True, timeout_seconds=5))
    except Exception:
        return False


def ranking_metrics(scores: list[float], labels: list[int]) -> dict[str, float]:
    """Compute tie-aware AUROC and stable-order mean accuracy over coverage."""
    if not scores or len(scores) != len(labels):
        raise ValueError("Scores and labels must have the same nonzero length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Labels must be 0 (incorrect) or 1 (correct)")
    n_correct = sum(labels)
    n_incorrect = len(labels) - n_correct
    if not n_correct or not n_incorrect:
        raise ValueError("AUROC requires both correct and incorrect answers")
    confidence_order = sorted(zip((-score for score in scores), labels))
    rank_sum, i = 0.0, 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and confidence_order[j][0] == confidence_order[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        rank_sum += average_rank * sum(label for _, label in confidence_order[i:j])
        i = j
    auroc = (rank_sum - n_correct * (n_correct + 1) / 2) / (n_correct * n_incorrect)
    # Sorting only by score preserves source order when scores tie.
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    cumulative, accuracy_sum = 0, 0.0
    for count, (_, label) in enumerate(ordered, start=1):
        cumulative += label
        accuracy_sum += cumulative / count
    return {"auroc": auroc, "auarc": accuracy_sum / len(labels)}


def evaluate(path: Path, dataset: str, protocol: str) -> dict:
    scores, labels = [], []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["dataset"] != dataset or record["protocol"] != protocol:
                raise ValueError(f"Unexpected setting in {record['record_uid']}")
            if any(not agent["quality"]["entropy_available"]
                   for rd in record["round_outputs"] for agent in rd["agents"]):
                raise ValueError(f"Missing entropy in {record['record_uid']}")
            scores.append(sauce_score(
                record["signals"]["agreement"], record["signals"]["token_entropy"],
                protocol=protocol,
            ))
            labels.append(int(answer_correct(record["consensus_answer"], record["answer"])))
    expected_n, expected_roc, expected_arc = PAPER_RESULTS[(dataset, protocol)]
    metrics = ranking_metrics(scores, labels)
    matches = {
        "samples": len(labels) == expected_n,
        "auroc": f"{100 * metrics['auroc']:.2f}" == f"{expected_roc:.2f}",
        "auarc": f"{100 * metrics['auarc']:.2f}" == f"{expected_arc:.2f}",
    }
    return {"dataset": dataset, "protocol": protocol, "samples": len(labels),
            "correct": sum(labels), "metrics": metrics,
            "paper_percent": {"auroc": expected_roc, "auarc": expected_arc},
            "matches_paper": matches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("math_500", "mmlu_pro", "bbh", "all"),
                        default="math_500")
    parser.add_argument("--protocol", choices=("debate", "dylan", "all"), default="debate")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets/qwen3_4b")
    parser.add_argument("--output", type=Path, help="Write metric values and paper comparisons as JSON.")
    args = parser.parse_args()
    if importlib.util.find_spec("math_verify") is None:
        parser.error('Install the evaluation extra first: pip install ".[reproduce]"')
    datasets = ("math_500", "mmlu_pro", "bbh") if args.dataset == "all" else (args.dataset,)
    protocols = ("debate", "dylan") if args.protocol == "all" else (args.protocol,)
    settings = [(dataset, protocol) for dataset in datasets for protocol in protocols]
    for dataset, protocol in settings:
        path = args.data_dir / f"{protocol}_{dataset}.jsonl.gz"
        if not path.exists():
            parser.error(f"Missing {path}; run examples/download_rollouts.py --all")
    results = []
    print("Setting                    N       AUROC    AUARC    Paper")
    for dataset, protocol in settings:
        result = evaluate(args.data_dir / f"{protocol}_{dataset}.jsonl.gz", dataset, protocol)
        results.append(result)
        status = "PASS" if all(result["matches_paper"].values()) else "DIFF"
        print(f"{dataset + '/' + protocol:<25} {result['samples']:>6}    "
              f"{100 * result['metrics']['auroc']:>6.2f}   "
              f"{100 * result['metrics']['auarc']:>6.2f}   {status}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {"model": "Qwen/Qwen3-4B", "dataset_revision": "qwen-rollouts-v1",
                  "metrics_units": "fraction; paper_percent is percentage",
                  "math_verify_version": importlib.metadata.version("math-verify"),
                  "settings": results}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(not all(result["matches_paper"].values()) for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
