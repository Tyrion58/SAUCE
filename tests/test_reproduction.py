"""Check ranking direction, ties, and the reproduction answer policy."""

import importlib.util
import unittest

from examples.reproduce_paper import answer_correct, ranking_metrics


class ReproductionTests(unittest.TestCase):
    def test_low_uncertainty_predicts_correctness(self) -> None:
        result = ranking_metrics([-1.0, 0.0], [1, 0])
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["auarc"], 0.75)

    def test_auroc_uses_average_ranks_and_auarc_preserves_ties(self) -> None:
        result = ranking_metrics([-0.8, -0.5, -0.5, -0.1], [1, 1, 0, 0])
        self.assertEqual(result["auroc"], 0.875)
        self.assertAlmostEqual(result["auarc"], (1 + 1 + 2 / 3 + 0.5) / 4)
        swapped = ranking_metrics([-0.8, -0.5, -0.5, -0.1], [1, 0, 1, 0])
        self.assertEqual(swapped["auroc"], result["auroc"])
        self.assertLess(swapped["auarc"], result["auarc"])

    def test_degenerate_metric_inputs_raise(self) -> None:
        for scores, labels in [([], []), ([0], []), ([0, 1], [1, 1]),
                               ([0, 1], [0, 0]), ([0, 1], [1, 2])]:
            with self.subTest(scores=scores, labels=labels):
                with self.assertRaises(ValueError):
                    ranking_metrics(scores, labels)

    def test_choice_wrappers_and_empty_predictions(self) -> None:
        for prediction in ["A", "(A)", r"\boxed{A}", r"\mathrm{A}", r"\text{A}"]:
            self.assertTrue(answer_correct(prediction, "(A)"))
        self.assertFalse(answer_correct("B", "A"))
        self.assertFalse(answer_correct("", "A"))

    @unittest.skipUnless(importlib.util.find_spec("math_verify"), "Optional math-verify is not installed")
    def test_math_equivalence_and_percent_format(self) -> None:
        self.assertTrue(answer_correct(r"\frac{1}{2}", "0.5"))
        self.assertTrue(answer_correct(r"80\%", "80%"))
        self.assertFalse(answer_correct("2", "3"))


if __name__ == "__main__":
    unittest.main()
