import importlib.util
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("recompute_transformation_quality.py")
SPEC = importlib.util.spec_from_file_location("recompute_transformation_quality", MODULE_PATH)
quality = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)


class TransformationQualityTests(unittest.TestCase):
    def test_declared_target_mapping(self):
        self.assertEqual([quality.map_target_score(value) for value in range(1, 6)], [
            "low", "low", "high", "high", "high"
        ])

    def test_pronouns_are_detected_as_tokens(self):
        pattern = quality.TargetPattern("low", 2.0, tuple(), 1)
        without_pronoun = quality.quality_components("difficult situation", pattern)
        with_pronoun = quality.quality_components("I worry", pattern)
        with_contracted_pronoun = quality.quality_components("I'm worried", pattern)
        self.assertEqual(without_pronoun["q_pronoun"], 0.0)
        self.assertEqual(with_pronoun["q_pronoun"], 1.0)
        self.assertEqual(with_contracted_pronoun["q_pronoun"], 1.0)

    def test_keyword_matching_uses_complete_unigrams_and_bigrams(self):
        pattern = quality.TargetPattern(
            "high", 4.0, ("feel", "feel better", "help", "pain", "support"), 1
        )
        result = quality.quality_components("I feel better with helpfulness", pattern)
        self.assertEqual(result["keyword_matches"], 2)

    def test_composite_formula(self):
        pattern = quality.TargetPattern(
            "low",
            10.0,
            ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"),
            1,
        )
        result = quality.quality_components("I alpha beta one two three four five", pattern)
        self.assertAlmostEqual(result["q_length"], 0.8)
        self.assertAlmostEqual(result["q_keyword"], 0.2)
        self.assertAlmostEqual(result["q_pronoun"], 1.0)
        self.assertAlmostEqual(result["quality"], 0.6)

    def test_multiset_subtraction_preserves_extra_duplicate(self):
        frame = pd.DataFrame({"conversation": ["target", "new", "target"]})
        transformed, rule = quality.select_transformed_rows(
            frame, "conversation", Counter({"target": 1})
        )
        self.assertEqual(transformed["conversation"].tolist(), ["new", "target"])
        self.assertIn("multiset", rule)

    def test_cli_writes_auditable_outputs(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            target_path = directory / "target.csv"
            augmented_path = directory / "augmented.csv"
            output_dir = directory / "results"

            pd.DataFrame(
                {
                    "conversation": [
                        "I feel calm today",
                        "My concern is manageable",
                        "I need urgent emotional help",
                        "My pain feels overwhelming",
                        "We need support immediately",
                    ],
                    "intensity": [1, 2, 3, 4, 5],
                    "label": [0, 0, 0, 1, 1],
                }
            ).to_csv(target_path, index=False)
            pd.DataFrame(
                {
                    "conversation": [
                        "I feel calm today",
                        "My concern is manageable",
                        "I need urgent emotional help",
                        "My pain feels overwhelming",
                        "We need support immediately",
                        "I remain reflective",
                        "I urgently need help",
                    ],
                    "label": [0, 0, 0, 1, 1, 0, 1],
                }
            ).to_csv(augmented_path, index=False)

            exit_code = quality.main(
                [
                    "--target-corpus", str(target_path),
                    "--dataset", f"fixture={augmented_path}",
                    "--output-dir", str(output_dir),
                    "--target-label-column", "label",
                    "--expected-target-count", "5",
                    "--expected-total-count", "7",
                    "--expected-transformed-count", "2",
                ]
            )
            self.assertEqual(exit_code, 0)
            summary = pd.read_csv(output_dir / "transformation_quality_summary.csv")
            self.assertEqual(summary.loc[0, "transformed_instances"], 2)
            self.assertTrue(math.isfinite(summary.loc[0, "mean_quality_score"]))
            details = pd.read_csv(output_dir / "transformation_quality_per_sample.csv")
            self.assertEqual(len(details), 2)
            self.assertTrue(details["text_sha256"].str.fullmatch(r"[0-9a-f]{64}").all())
            manifest = json.loads(
                (output_dir / "transformation_quality_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["configuration"]["target_label_mapping"],
                {"low": [1, 2], "high": [3, 4, 5]},
            )
            self.assertEqual(
                manifest["target_corpus"]["stored_label_audit"]["disagreements_with_declared_mapping"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
