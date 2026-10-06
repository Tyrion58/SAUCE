"""SAUCE: Sequential Agent Uncertainty through Consensus Evolution."""

from .core import lambda_kalman_forward, sauce, sauce_score
from .signals import compute_token_entropy, majority_margin

__version__ = "0.1.0"
__all__ = [
    "sauce",
    "sauce_score",
    "lambda_kalman_forward",
    "majority_margin",
    "compute_token_entropy",
]
