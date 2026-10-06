"""Agreement and predictive-entropy signals for SAUCE."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def majority_margin(
    parsed_answers: Sequence[str | None],
    n_agents: int | None = None,
) -> float:
    """Return the modal answer count divided by the specified team size.

    Answers must already be parsed into comparable strings. None and empty
    strings indicate unparseable responses and contribute no vote. The
    denominator defaults to the number of supplied responses, including
    unparseable ones. Pass the initial team size explicitly to preserve
    the paper's DyLAN trace convention after pruning. Raises ValueError
    when no answer can be parsed or the denominator is invalid.
    """
    if n_agents is None:
        n_agents = len(parsed_answers)
    if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents <= 0:
        raise ValueError("n_agents must be a positive integer")
    if len(parsed_answers) > n_agents:
        raise ValueError("n_agents cannot be smaller than the response count")
    for answer in parsed_answers:
        if answer is not None and not isinstance(answer, str):
            raise ValueError("parsed answers must be strings or None")
    valid = [answer for answer in parsed_answers if answer is not None and answer != ""]
    if not valid:
        raise ValueError("at least one answer must be parseable")
    return max(Counter(valid).values()) / n_agents


def compute_token_entropy(topk_logprobs: Sequence[Sequence[float]]) -> float:
    """Return mean token entropy in nats from top-k log-probabilities.

    Each inner sequence contains the alternatives for one generated token.
    Probabilities are renormalized over these alternatives using softmax,
    matching the research implementation. Log-probabilities must be <= 0;
    negative infinity represents zero probability. Every token must have
    at least one finite alternative. Missing or empty measurements raise
    ValueError instead of being interpreted as zero entropy.
    """
    if topk_logprobs is None or len(topk_logprobs) == 0:
        raise ValueError("topk_logprobs must contain at least one measured token")
    total_entropy = 0.0
    for t, token_lps in enumerate(topk_logprobs):
        if token_lps is None or len(token_lps) == 0:
            raise ValueError(f"token {t} has no log-probability measurements")
        try:
            values = [float(lp) for lp in token_lps]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"token {t} contains invalid log-probabilities") from exc
        if any(math.isnan(lp) or lp > 0.0 for lp in values):
            raise ValueError(f"token {t} log-probabilities must be <= 0 and not NaN")
        max_lp = max(values)
        if not math.isfinite(max_lp):
            raise ValueError(f"token {t} must have a finite log-probability")
        exps = [math.exp(lp - max_lp) for lp in values]
        sum_exps = sum(exps)
        entropy_j = 0.0
        for value in exps:
            probability = value / sum_exps
            if probability > 0.0:
                entropy_j -= probability * math.log(probability)
        total_entropy += entropy_j
    return total_entropy / len(topk_logprobs)
