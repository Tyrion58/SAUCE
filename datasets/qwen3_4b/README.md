# Qwen3-4B rollout trajectories

Saved multi-agent reasoning trajectories from the SAUCE paper, accepted at
NeurIPS 2026. This first dataset release covers **Qwen3-4B**, two protocols,
and three benchmarks: **27,564 trajectories and 176,898 agent responses**.
It supports uncertainty estimation, consensus analysis, and follow-up work
without generating new model outputs.

Each file contains one complete trajectory per JSONL line and is compressed
with gzip. All source trajectories are retained, including generation and
answer-format failures. No quality-based filtering has been applied.

## Download

The data is hosted on [Hugging Face](https://huggingface.co/datasets/Tyrion279/SAUCE-Qwen3-4B-Rollouts).
Download an individual setting from the versioned `qwen-rollouts-v1` release:

| Benchmark | Protocol | Trajectories | Agent responses | Download |
|---|---|---:|---:|---|
| MATH-500 | Debate | 500 | 4,500 | [JSONL.gz](https://huggingface.co/datasets/Tyrion279/SAUCE-Qwen3-4B-Rollouts/resolve/qwen-rollouts-v1/data/debate_math_500.jsonl.gz) |
| MATH-500 | DyLAN | 500 | 2,208 | [JSONL.gz](https://huggingface.co/datasets/Tyrion279/SAUCE-Qwen3-4B-Rollouts/resolve/qwen-rollouts-v1/data/dylan_math_500.jsonl.gz) |
| MMLU-Pro | Debate | 12,032 | 108,288 | [JSONL.gz](https://huggingface.co/datasets/Tyrion279/SAUCE-Qwen3-4B-Rollouts/resolve/qwen-rollouts-v1/data/debate_mmlu_pro.jsonl.gz) |
| MMLU-Pro | DyLAN | 12,032 | 45,316 | [JSONL.gz](https://huggingface.co/datasets/Tyrion279/SAUCE-Qwen3-4B-Rollouts/resolve/qwen-rollouts-v1/data/dylan_mmlu_pro.jsonl.gz) |
| BBH | Debate | 1,250 | 11,250 | [JSONL.gz](https://huggingface.co/datasets/Tyrion279/SAUCE-Qwen3-4B-Rollouts/resolve/qwen-rollouts-v1/data/debate_bbh.jsonl.gz) |
| BBH | DyLAN | 1,250 | 5,336 | [JSONL.gz](https://huggingface.co/datasets/Tyrion279/SAUCE-Qwen3-4B-Rollouts/resolve/qwen-rollouts-v1/data/dylan_bbh.jsonl.gz) |

Total compressed size is about **66 MB**. Exact byte counts, source hashes,
and per-setting quality statistics are in [manifest.json](manifest.json).
Every exported trajectory was checked against its source: original text,
answers, and measurements were preserved, and restored choices were verified.
The results are in [validation.json](validation.json).
To verify the local files:

```bash
python examples/download_rollouts.py --all
cd datasets/qwen3_4b
sha256sum -c SHA256SUMS
```

The two protocols use the same question set within each benchmark. BBH
covers five tasks, with 250 questions each:
`tracking_shuffled_objects_seven_objects`,
`logical_deduction_seven_objects`, `geometric_shapes`, `hyperbaton`, and
`salient_translation_error_detection`.

## Read and score

After installing SAUCE, run from the repository root:

```bash
# Download one setting and preview five trajectories.
python examples/download_rollouts.py
python examples/score_rollouts.py

# Download and score another setting.
python examples/download_rollouts.py --protocol dylan --dataset mmlu_pro
python examples/score_rollouts.py datasets/qwen3_4b/dylan_mmlu_pro.jsonl.gz \
  --limit 0 > scores.csv
```

For direct access using Python's standard library:

```python
import gzip
import json
from sauce import sauce_score

with gzip.open("datasets/qwen3_4b/debate_math_500.jsonl.gz", "rt", encoding="utf-8") as f:
    record = json.loads(next(f))

score = sauce_score(
    record["signals"]["agreement"],
    record["signals"]["token_entropy"],
    protocol=record["protocol"],
)
print(record["record_uid"], score)
print(record["round_outputs"][0]["agents"][0]["response"])
```

Higher scores indicate greater uncertainty. These scores are ranking
signals; converting them to error probabilities requires calibration.

Load directly with Hugging Face Datasets (requires `pip install datasets`):

```python
from datasets import load_dataset

trajectories = load_dataset(
    "Tyrion279/SAUCE-Qwen3-4B-Rollouts", "debate_math_500",
    split="test", revision="qwen-rollouts-v1",
)
```

The other configuration names are `dylan_math_500`, `debate_mmlu_pro`,
`dylan_mmlu_pro`, `debate_bbh`, and `dylan_bbh`.

## Record fields

| Field | Meaning |
|---|---|
| `schema_version` | Export schema, currently `1.0` |
| `id`, `record_uid` | Unique trajectory identifier, including protocol, benchmark, and source position |
| `question_uid` | Shared question identifier across the two protocols |
| `original_id`, `original_row_id`, `source_index` | Original identifiers and zero-based position in the source file |
| `model`, `protocol`, `dataset` | Model and setting identifiers |
| `dataset_id`, `dataset_split`, `dataset_row_id` | Original benchmark identifiers; the old row ID alone may repeat |
| `bbh_subset` | BBH task name, when applicable |
| `prompt`, `answer` | Saved question text and reference answer |
| `options` | Original ordered MMLU-Pro choices; A corresponds to position 0 |
| `num_agents`, `num_rounds` | Initial team size and requested maximum rounds |
| `round_outputs` | Executed rounds, agent responses, stored parsed answers, and quality annotations |
| `agent_final_answers`, `consensus_answer` | Last-round agent outputs and stored final consensus |
| `dylan_roles`, `dylan_ranker_trace` | Saved DyLAN roles and ranking/pruning history, when applicable |
| `metadata` | Saved generation settings: model, seed, temperature, top-p, token budget, provider, and task type |
| `observation_noise` | Original per-round signal arrays and auxiliary collector values |
| `signals` | Ready-to-use agreement and mean token entropy sequences for SAUCE |
| `quality_flags` | Union of agent quality flags, plus `empty_consensus` when applicable |

MMLU-Pro question IDs are preserved in `question_uid`. For MATH-500 and
BBH, the question UID hashes the benchmark, split, task, and question text.
Use `question_uid` for matching across protocols and `record_uid` for unique
records; do not deduplicate by the old ID or bare MMLU-Pro question stem.

The saved `prompt` is the question text, not a complete request/message log.
MMLU-Pro options were restored from the matching source revision
`54611cde22c74cca43dd78732198de6abe971398`, with all question IDs, question
texts, and reference answers checked after the collector's outer-whitespace
trimming. System prompts and complete historical requests were not saved.

## Signal conventions

- `signals.agreement[t]` is the saved `observation_noise.rounds[t].M_t`.
  It divides the modal parsed-answer count by the **initial** team size,
  including after DyLAN pruning. When every parsed answer is null, the
  saved research collector uses a fallback of **1 / initial team size**.
  These saved values are preserved; the public `majority_margin` helper
  raises an error for entirely unparseable rounds, so use `signals` directly.
- `signals.token_entropy[t]` is the arithmetic mean of the saved
  `token_entropy_t` array for that round. Entropy is in nats and is averaged
  over tokens within each agent, then over the agents that actually ran.
- Per-agent arrays follow the order of `round_outputs[t].agents`; after
  pruning, array position is not necessarily the agent index.
- The collector also saved auxiliary values named `R_t`, `R_t_i`, and
  `o_t`. Use `signals` with the released SAUCE scorer; those auxiliary
  values are not the current mean-token-entropy input.

Debate uses three agents for three rounds. DyLAN uses up to three rounds
and can stop or prune agents. The MATH-500 DyLAN setting starts with **four**
agents; the other DyLAN settings start with three. Shorter DyLAN trajectories
are normal completed runs. Generation budgets are 8,192 tokens except
**BBH / Debate**, whose saved budget is 12,288. All settings record seed 0,
temperature 0.7, and top-p 0.9.

## Quality annotations

Each round agent has a `quality` object. These annotations leave the
original response, parsed answer, and numerical measurements unchanged.
Final-agent entries carry the corresponding last-round annotations.

| Annotation | Meaning |
|---|---|
| `empty_response` | Response has no non-whitespace text |
| `parse_missing` | Original parsed answer is null or empty; not necessarily an incorrect answer |
| `entropy_available` | Saved entropy is finite and has no missing-logprob sentinel |
| `inferred_token_count` | Token count recovered from a consistent saved sum/mean pair, or null |
| `reencoded_token_count` | Visible response retokenized for the legacy MATH-500 / Debate file, otherwise null |
| `token_count_method` | `saved_sum_mean` or `visible_text_reencoding` |
| `output_budget_candidate` | Count is within one token of the budget, or within two tokens for visible-text reencoding; null if count is unavailable |
| `repetition_candidate` | Repeated-line or low-diversity-tail heuristic triggered |
| `unfinished_boxed_candidate` | Original answer-box check found a possibly unfinished box |
| `dangling_tail_candidate` | Text ends with a colon, comma, operator, slash, or opening delimiter |
| `termination_reason_available` | Whether the original agent entry saved `finish_reason` |

Budget and text annotations are **review candidates**, not definitive
truncation or invalidity labels. Manual inspection confirmed some loops
and unfinished outputs, but the data does not record termination reasons.
Repetition detection flags a long line appearing at least eight times and
covering more than 35% of the text, or at least 30 nonempty lines with no
more than five distinct lines in the final 8,000 characters.

| Setting | Budget candidates | Repetition candidates | Missing parsed answers |
|---|---:|---:|---:|
| MATH-500 / Debate | 15* | 7 | 14 |
| MMLU-Pro / Debate | 181 | 73 | 468 |
| BBH / Debate | 42 | 13 | 31 |
| MATH-500 / DyLAN | 15 | 6 | 11 |
| MMLU-Pro / DyLAN | 194 | 87 | 517 |
| BBH / DyLAN | 48 | 17 | 34 |

*The legacy MATH-500 / Debate file lacks raw sums. Its 15 candidates use
visible-text reencoding, which may differ by terminal tokens from the
original measured count. In total, there are **495 budget candidates**
(about 0.28% of agent responses), **203 repetition candidates**, and
**1,075 missing parsed answers**. Categories can overlap. There are no empty
agent responses or missing entropy measurements; **20 trajectories have
empty final consensus**, all in MMLU-Pro (11 Debate, 9 DyLAN).

For an optional filtered view, explicitly choose flags to exclude:

```bash
python examples/score_rollouts.py datasets/qwen3_4b/debate_mmlu_pro.jsonl.gz \
  --limit 0 --exclude-flag output_budget_candidate \
  --exclude-flag repetition_candidate > filtered_scores.csv
```

Filtering changes the evaluated sample set. The full files retain the
original experimental distribution.

## Scope and attribution

This release covers the paper's Qwen3-4B settings. It does not contain the
other models' trajectories, original top-k token distributions, provider
usage counters, model revision identifiers, or serving infrastructure.
The collector's legacy `uncertainty` snapshot was omitted; original
responses, parsed answers, consensus, and `observation_noise` were preserved.
Deployment endpoints were removed, and quality annotations, normalized
identifiers, options, and scoring signals were added.

Please cite the [SAUCE paper](../../README.md#citation) and the benchmarks
used in your work. Our contributions use the repository's [MIT license](../../LICENSE).
Benchmark attribution and upstream license copies are in [NOTICE.md](NOTICE.md)
and [licenses/](licenses/).
