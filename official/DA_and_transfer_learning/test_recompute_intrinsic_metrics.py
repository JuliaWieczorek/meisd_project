import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("recompute_intrinsic_metrics.py")
SPEC = importlib.util.spec_from_file_location("recompute_intrinsic_metrics", MODULE_PATH)
metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


class MetricTests(unittest.TestCase):
    def test_pairwise_metrics_are_one_for_identical_long_texts(self):
        text = "one two three four five six seven"
        pairs = [(text, text), ("alpha beta gamma delta epsilon", "alpha beta gamma delta epsilon")]
        self.assertAlmostEqual(metrics.corpus_bleu(pairs), 1.0)
        self.assertAlmostEqual(metrics.corpus_chrf(pairs), 1.0)

    def test_diversity_and_novelty(self):
        diversity = metrics.corpus_diversity(["a b c", "a b d"])
        self.assertEqual(diversity["token_count"], 6)
        self.assertEqual(diversity["type_count"], 4)
        self.assertAlmostEqual(diversity["ttr"], 4 / 6)
        self.assertAlmostEqual(diversity["utr"], 1.0)
        self.assertAlmostEqual(metrics.mean_jaccard_novelty(["x y"], ["a b"]), 1.0)

    def test_source_pairs_are_never_fabricated(self):
        without_source = pd.DataFrame({"conversation": ["transformed text"]})
        pairs, column = metrics.source_pairs(without_source, "conversation")
        self.assertEqual(pairs, [])
        self.assertIsNone(column)

        with_source = pd.DataFrame(
            {"conversation": ["transformed text"], "source_text": ["source text"]}
        )
        pairs, column = metrics.source_pairs(with_source, "conversation")
        self.assertEqual(pairs, [("source text", "transformed text")])
        self.assertEqual(column, "source_text")

    def test_cli_writes_null_for_unavailable_pairwise_metrics(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            target = directory / "target.csv"
            source = directory / "source.csv"
            augmented = directory / "augmented.csv"
            output = directory / "results"

            pd.DataFrame({"conversation": ["target example"]}).to_csv(target, index=False)
            pd.DataFrame({"conversation": ["source example"]}).to_csv(source, index=False)
            pd.DataFrame(
                {
                    "conversation": ["target example", "new transformed example"],
                    "is_augmented": [False, True],
                }
            ).to_csv(augmented, index=False)

            exit_code = metrics.main(
                [
                    "--target-corpus", str(target),
                    "--source-corpus", str(source),
                    "--dataset", f"fixture={augmented}",
                    "--output-dir", str(output),
                    "--skip-perplexity",
                ]
            )
            self.assertEqual(exit_code, 0)
            summary = pd.read_csv(output / "intrinsic_metrics_summary.csv")
            self.assertTrue(math.isnan(summary.loc[0, "bleu"]))
            self.assertTrue(math.isnan(summary.loc[0, "chrf"]))


if __name__ == "__main__":
    unittest.main()
