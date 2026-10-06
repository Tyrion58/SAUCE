"""Deterministic checks of the paper update, scoring, and signal conventions."""

import math
import unittest

from sauce import compute_token_entropy, lambda_kalman_forward, majority_margin
from sauce import sauce, sauce_score


class SauceTests(unittest.TestCase):
    def test_one_round_matches_hand_calculation(self) -> None:
        # lambda=1, q=0, v0=1, R=1: gain=1/(2+epsilon).
        result = sauce([0.75], [1.0], lam=1.0, q=0.0, sigma2_0=1.0)
        gain = 1.0 / (2.0 + 1e-8)
        self.assertAlmostEqual(result["score"], -gain * 0.75, places=14)
        self.assertAlmostEqual(result["trace"]["final_sigma2"], 1.0 - gain, places=14)

    def test_multiround_uses_updated_state_and_temporal_mean(self) -> None:
        # With v0=1, q=0, lambda=1, R=1, observations 1 then 0,
        # the unstabilized posterior means are 1/2 then 1/3.
        result = sauce([1.0, 0.0], [1.0, 1.0], lam=1.0, q=0.0, sigma2_0=1.0)
        rounds = result["trace"]["rounds"]
        self.assertAlmostEqual(rounds[0]["mu"], 0.5, places=7)
        self.assertAlmostEqual(rounds[1]["mu"], 1.0 / 3.0, places=7)
        self.assertAlmostEqual(result["score"], -5.0 / 12.0, places=7)
        self.assertNotAlmostEqual(result["score"], -result["trace"]["final_mu"])

    def test_protocol_presets_and_explicit_overrides(self) -> None:
        for protocol, lam, q in [("debate", 0.95, 0.10), ("dylan", 0.70, 0.001)]:
            with self.subTest(protocol=protocol):
                result = sauce([2 / 3], [0.2], protocol=protocol)
                trace = result["trace"]
                self.assertEqual((trace["lam"], trace["q"], trace["sigma2_0"], trace["mu_0"]),
                                 (lam, q, 0.25, 0.0))
        overridden = sauce([1.0], [0.2], protocol="dylan", q=0.2)
        self.assertEqual(overridden["trace"]["q"], 0.2)
        self.assertEqual(overridden["trace"]["lam"], 0.70)

    def test_stronger_agreement_means_lower_uncertainty(self) -> None:
        entropy = [0.4, 0.2, 0.1]
        self.assertLess(sauce_score([2 / 3, 1.0, 1.0], entropy),
                        sauce_score([1 / 3, 1 / 3, 2 / 3], entropy))

    def test_missing_or_invalid_measurements_are_rejected(self) -> None:
        cases = [([], []), ([1.0], []), ([None], [0.2]),
                 ([1.0], [None]), ([float("nan")], [0.2]),
                 ([1.0], [float("inf")]), ([1.1], [0.2]), ([1.0], [-0.1])]
        for agreement, entropy in cases:
            with self.subTest(agreement=agreement, entropy=entropy):
                with self.assertRaises(ValueError):
                    sauce(agreement, entropy)
        for options in [{"protocol": "unknown"}, {"lam": -0.1}, {"lam": 1.1},
                        {"q": -1.0}, {"sigma2_0": 0.0}, {"mu_0": float("nan")}]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    sauce([1.0], [0.2], **options)
        with self.assertRaises(ValueError):
            lambda_kalman_forward([1.0], [-0.2])

    def test_measured_zero_entropy_is_finite(self) -> None:
        result = sauce([1.0], [0.0])
        self.assertTrue(math.isfinite(result["score"]))
        self.assertGreater(result["trace"]["final_sigma2"], 0.0)

    def test_majority_denominator_is_explicit_after_pruning(self) -> None:
        self.assertEqual(majority_margin(["A", "A"]), 1.0)
        self.assertEqual(majority_margin(["A", "A"], n_agents=3), 2 / 3)
        self.assertEqual(majority_margin(["A", None, "A"]), 2 / 3)
        self.assertEqual(majority_margin(["A", "B", "C"]), 1 / 3)
        for answers, count in [([None, ""], None), (["A", "B"], 1), (["A"], 0)]:
            with self.subTest(answers=answers, count=count):
                with self.assertRaises(ValueError):
                    majority_margin(answers, n_agents=count)

    def test_entropy_renormalization_and_token_average(self) -> None:
        # Two equally likely returned alternatives have entropy log(2),
        # even if together they represent only half the full probability mass.
        entropy = compute_token_entropy([[math.log(0.25), math.log(0.25)], [0.0]])
        self.assertAlmostEqual(entropy, math.log(2.0) / 2, places=14)
        self.assertEqual(compute_token_entropy([[0.0, float("-inf")]]), 0.0)

    def test_missing_or_invalid_topk_logprobs_are_rejected(self) -> None:
        for values in [None, [], [[]], [None], [[None]], [[float("nan")]],
                       [[float("-inf")]], [[0.1]]]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    compute_token_entropy(values)


if __name__ == "__main__":
    unittest.main()
