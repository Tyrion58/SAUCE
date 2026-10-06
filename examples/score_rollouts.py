"""Read the released JSONL trajectories and score their saved round signals."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

from sauce import sauce_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets/qwen3_4b/debate_math_500.jsonl.gz",
    )
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of records to score; 0 scores the whole file.")
    parser.add_argument("--exclude-flag", action="append", default=[],
                        help="Skip records carrying this quality flag; repeat to combine flags.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if not args.input.exists():
        parser.error("Input file is missing; download it with examples/download_rollouts.py")

    writer = csv.writer(sys.stdout)
    writer.writerow(["record_uid", "question_uid", "protocol", "score", "quality_flags"])
    opener = gzip.open if args.input.suffix == ".gz" else open
    excluded = set(args.exclude_flag)
    scored = 0
    with opener(args.input, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if excluded.intersection(record["quality_flags"]):
                continue
            if any(not agent["quality"]["entropy_available"]
                   for rd in record["round_outputs"] for agent in rd["agents"]):
                raise ValueError(f"Missing entropy in {record['record_uid']}")
            score = sauce_score(
                record["signals"]["agreement"], record["signals"]["token_entropy"],
                protocol=record["protocol"],
            )
            writer.writerow([record["record_uid"], record["question_uid"],
                             record["protocol"], f"{score:.12f}",
                             ";".join(record["quality_flags"])])
            scored += 1
            if args.limit and scored >= args.limit:
                break


if __name__ == "__main__":
    main()
