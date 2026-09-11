#!/usr/bin/env python3
"""Recompute transformation-only quality statistics with explicit provenance.

This analysis is intentionally separate from the historical publication code.
It removes authentic target-domain rows by multiset subtraction, evaluates only
the remaining transformed source-domain rows, and records every scoring choice
needed to reproduce the resulting table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
TEXT_COLUMN_CANDIDATES = ("conversation", "utterances", "utterance", "text", "content")
SOURCE_COLUMN_CANDIDATES = ("source_text", "original_text", "source", "original")
FIRST_PERSON_PRONOUNS = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
)
TARGET_SCORE_TO_CLASS = {1: "low", 2: "low", 3: "high", 4: "high", 5: "high"}
TRANSFORMED_LABEL_TO_CLASS = {0: "low", 1: "high"}
KEYWORD_VECTORIZER_ARGUMENTS = {
    "max_features": 50,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "lowercase": True,
}
KEYWORD_ANALYZER = TfidfVectorizer(
    stop_words=KEYWORD_VECTORIZER_ARGUMENTS["stop_words"],
    ngram_range=KEYWORD_VECTORIZER_ARGUMENTS["ngram_range"],
    lowercase=KEYWORD_VECTORIZER_ARGUMENTS["lowercase"],
).build_analyzer()


@dataclass(frozen=True)
class DatasetSpec:
    method: str
    path: Path


@dataclass(frozen=True)
class TargetPattern:
    class_name: str
    mean_length: float
    keywords: tuple[str, ...]
    instance_count: int


def normalise_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(normalise_text(text).lower())


def word_count(text: str) -> int:
    return len(tokenise(text))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table format: {path}")


def resolve_column(frame: pd.DataFrame, preferred: str | None, candidates: Sequence[str]) -> str:
    lower_to_original = {str(column).lower(): str(column) for column in frame.columns}
    if preferred:
        if preferred in frame.columns:
            return preferred
        match = lower_to_original.get(preferred.lower())
        if match:
            return match
        raise ValueError(f"Column {preferred!r} is absent; available columns: {list(frame.columns)}")
    for candidate in candidates:
        match = lower_to_original.get(candidate.lower())
        if match:
            return match
    raise ValueError(f"No matching column found; available columns: {list(frame.columns)}")


def parse_dataset_spec(raw: str) -> DatasetSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Dataset must use METHOD=PATH syntax")
    method, raw_path = raw.split("=", 1)
    method = method.strip()
    if not method:
        raise argparse.ArgumentTypeError("Dataset method name cannot be empty")
    return DatasetSpec(method=method, path=Path(raw_path.strip()).expanduser().resolve())


def truthy_mask(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "yes", "y", "augmented", "synthetic", "transformed"}
    return series.fillna(False).map(lambda value: str(value).strip().lower() in truthy)


def select_transformed_rows(
    frame: pd.DataFrame,
    text_column: str,
    target_text_counts: Counter[str],
) -> tuple[pd.DataFrame, str]:
    lower_to_original = {str(column).lower(): str(column) for column in frame.columns}
    if "is_augmented" in lower_to_original:
        column = lower_to_original["is_augmented"]
        return frame.loc[truthy_mask(frame[column])].copy(), f"explicit {column} flag"

    for candidate in SOURCE_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            column = lower_to_original[candidate]
            mask = frame[column].map(normalise_text).ne("")
            return frame.loc[mask].copy(), f"non-empty {column} provenance"

    remaining_target_counts = target_text_counts.copy()
    transformed_mask: list[bool] = []
    for value in frame[text_column]:
        text = normalise_text(value).lower()
        if text and remaining_target_counts[text] > 0:
            remaining_target_counts[text] -= 1
            transformed_mask.append(False)
        else:
            transformed_mask.append(True)
    return (
        frame.loc[pd.Series(transformed_mask, index=frame.index)].copy(),
        "multiset subtraction of authentic target-corpus instances",
    )


def map_target_score(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Target intensity {value!r} is not numeric") from error
    if not numeric.is_integer() or int(numeric) not in TARGET_SCORE_TO_CLASS:
        raise ValueError(f"Target intensity {value!r} is outside the supported 1--5 scale")
    return TARGET_SCORE_TO_CLASS[int(numeric)]


def map_transformed_label(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Transformed label {value!r} is not numeric") from error
    if not numeric.is_integer() or int(numeric) not in TRANSFORMED_LABEL_TO_CLASS:
        raise ValueError(f"Transformed label {value!r} is not binary")
    return TRANSFORMED_LABEL_TO_CLASS[int(numeric)]


def extract_target_keywords(texts: Sequence[str], keyword_count: int = 10) -> tuple[str, ...]:
    if not texts:
        raise ValueError("Cannot extract target keywords from an empty class")
    vectorizer = TfidfVectorizer(**KEYWORD_VECTORIZER_ARGUMENTS)
    matrix = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    means = np.asarray(matrix.mean(axis=0)).ravel()
    ranked = sorted(zip(features, means), key=lambda item: (-item[1], item[0]))
    return tuple(feature for feature, _ in ranked[:keyword_count])


def build_target_patterns(
    target: pd.DataFrame,
    text_column: str,
    intensity_column: str,
    keyword_count: int = 10,
) -> dict[str, TargetPattern]:
    prepared = target[[text_column, intensity_column]].copy()
    prepared[text_column] = prepared[text_column].map(normalise_text)
    prepared = prepared.loc[prepared[text_column].ne("")]
    prepared["_class"] = prepared[intensity_column].map(map_target_score)

    patterns: dict[str, TargetPattern] = {}
    for class_name in ("low", "high"):
        texts = prepared.loc[prepared["_class"] == class_name, text_column].tolist()
        lengths = [word_count(text) for text in texts]
        patterns[class_name] = TargetPattern(
            class_name=class_name,
            mean_length=float(np.mean(lengths)),
            keywords=extract_target_keywords(texts, keyword_count=keyword_count),
            instance_count=len(texts),
        )
    return patterns


def present_ngrams(text: str) -> set[str]:
    return set(KEYWORD_ANALYZER(text))


def contains_first_person_pronoun(text: str) -> bool:
    for token in tokenise(text):
        base = token.split("'", 1)[0]
        if base in FIRST_PERSON_PRONOUNS:
            return True
    return False


def quality_components(
    text: str,
    target_pattern: TargetPattern,
    keyword_denominator: int = 10,
) -> dict[str, float | int | bool]:
    if target_pattern.mean_length <= 0:
        raise ValueError("Target mean length must be positive")
    length = word_count(text)
    q_length = max(0.0, 1.0 - abs(length - target_pattern.mean_length) / target_pattern.mean_length)

    units = present_ngrams(text)
    keyword_matches = sum(keyword in units for keyword in target_pattern.keywords)
    q_keyword = min(1.0, keyword_matches / keyword_denominator)

    pronoun_present = contains_first_person_pronoun(text)
    q_pronoun = 1.0 if pronoun_present else 0.0
    quality = 0.4 * q_length + 0.4 * q_keyword + 0.2 * q_pronoun
    return {
        "length": length,
        "keyword_matches": keyword_matches,
        "pronoun_present": pronoun_present,
        "q_length": q_length,
        "q_keyword": q_keyword,
        "q_pronoun": q_pronoun,
        "quality": quality,
    }


def sample_standard_deviation(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def evaluate_transformed_rows(
    transformed: pd.DataFrame,
    text_column: str,
    label_column: str,
    target_patterns: dict[str, TargetPattern],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    for source_row_index, row in transformed.iterrows():
        text = normalise_text(row[text_column])
        if not text:
            raise ValueError("A transformed row contains empty text")
        class_name = map_transformed_label(row[label_column])
        class_counts[class_name] += 1
        components = quality_components(text, target_patterns[class_name])
        expected_quality = (
            0.4 * float(components["q_length"])
            + 0.4 * float(components["q_keyword"])
            + 0.2 * float(components["q_pronoun"])
        )
        if not math.isclose(float(components["quality"]), expected_quality, abs_tol=1e-12):
            raise AssertionError("Composite quality score failed its internal reconciliation")
        if not 0.0 <= float(components["quality"]) <= 1.0:
            raise AssertionError("Quality score is outside [0, 1]")
        records.append(
            {
                "source_row_index": str(source_row_index),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "class": class_name,
                **components,
            }
        )

    texts = [normalise_text(value) for value in transformed[text_column]]
    lengths = [int(record["length"]) for record in records]
    qualities = [float(record["quality"]) for record in records]
    q_lengths = [float(record["q_length"]) for record in records]
    q_keywords = [float(record["q_keyword"]) for record in records]
    q_pronouns = [float(record["q_pronoun"]) for record in records]
    keyword_matches = [int(record["keyword_matches"]) for record in records]

    total = len(records)
    if total == 0:
        raise ValueError("No transformed rows were identified")
    summary = {
        "transformed_instances": total,
        "low_instances": class_counts["low"],
        "high_instances": class_counts["high"],
        "transformed_class_balance_ratio": min(class_counts.values()) / max(class_counts.values()),
        "exact_uniqueness_ratio": len(set(texts)) / total,
        "mean_length_words": float(np.mean(lengths)),
        "sd_length_words": sample_standard_deviation(lengths),
        "mean_quality_score": float(np.mean(qualities)),
        "sd_quality_score": sample_standard_deviation(qualities),
        "minimum_quality_score": min(qualities),
        "maximum_quality_score": max(qualities),
        "mean_length_component": float(np.mean(q_lengths)),
        "mean_keyword_component": float(np.mean(q_keywords)),
        "mean_pronoun_component": float(np.mean(q_pronouns)),
        "mean_keyword_matches": float(np.mean(keyword_matches)),
        "first_person_pronoun_ratio": float(np.mean(q_pronouns)),
    }
    return summary, records


def stored_target_label_audit(
    target: pd.DataFrame,
    intensity_column: str,
    label_column: str | None,
) -> dict[str, Any]:
    if label_column is None or label_column not in target.columns:
        return {"status": "unavailable", "reason": "no stored target label column"}
    expected = target[intensity_column].map(map_target_score).map({"low": 0, "high": 1})
    stored = pd.to_numeric(target[label_column], errors="coerce")
    comparable = stored.notna()
    disagreements = int((stored.loc[comparable].astype(int) != expected.loc[comparable]).sum())
    return {
        "status": "checked",
        "compared_instances": int(comparable.sum()),
        "disagreements_with_declared_mapping": disagreements,
    }


def write_outputs(
    rows: list[dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
    output_dir: Path,
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "transformation_quality_summary.csv", index=False)
    pd.DataFrame(per_sample_rows).to_csv(
        output_dir / "transformation_quality_per_sample.csv", index=False
    )
    (output_dir / "transformation_quality_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    columns = [
        "method",
        "transformed_instances",
        "low_instances",
        "high_instances",
        "exact_uniqueness_ratio",
        "mean_length_words",
        "sd_length_words",
        "mean_quality_score",
        "sd_quality_score",
        "minimum_quality_score",
        "maximum_quality_score",
    ]
    lines = [
        "# Corrected transformation-only quality statistics",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Quality score: `0.4 * Q_length + 0.4 * Q_keyword + 0.2 * Q_pronoun`.",
            "The summary evaluates transformed rows only; see the manifest for input hashes and target patterns.",
        ]
    )
    (output_dir / "transformation_quality_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-corpus", required=True, type=Path)
    parser.add_argument("--dataset", action="append", required=True, type=parse_dataset_spec)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-text-column")
    parser.add_argument("--target-intensity-column", default="intensity")
    parser.add_argument(
        "--target-label-column",
        help="Optional stored binary target label to audit against the declared 1--2/3--5 mapping",
    )
    parser.add_argument("--text-column")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--expected-target-count", type=int)
    parser.add_argument("--expected-total-count", type=int)
    parser.add_argument("--expected-transformed-count", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_path = args.target_corpus.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target corpus not found: {target_path}")

    target = read_table(target_path)
    if args.expected_target_count is not None and len(target) != args.expected_target_count:
        raise ValueError(
            f"Expected {args.expected_target_count} target rows, found {len(target)}"
        )
    target_text_column = resolve_column(target, args.target_text_column, TEXT_COLUMN_CANDIDATES)
    target_intensity_column = resolve_column(
        target, args.target_intensity_column, ("intensity", "original_intensity", "score")
    )
    target_patterns = build_target_patterns(target, target_text_column, target_intensity_column)
    target_text_counts = Counter(
        normalise_text(value).lower()
        for value in target[target_text_column]
        if normalise_text(value)
    )
    target_label_column = None
    if args.target_label_column:
        target_label_column = resolve_column(target, args.target_label_column, (args.target_label_column,))

    rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    datasets_manifest: dict[str, Any] = {}
    expected_class_counts: dict[str, int] | None = None
    for spec in args.dataset:
        if not spec.path.exists():
            raise FileNotFoundError(f"Augmented dataset not found: {spec.path}")
        frame = read_table(spec.path)
        text_column = resolve_column(frame, args.text_column, TEXT_COLUMN_CANDIDATES)
        label_column = resolve_column(frame, args.label_column, ("label", "binary_label"))
        if args.expected_total_count is not None and len(frame) != args.expected_total_count:
            raise ValueError(
                f"{spec.method}: expected {args.expected_total_count} total rows, found {len(frame)}"
            )
        transformed, selection_rule = select_transformed_rows(frame, text_column, target_text_counts)
        if args.expected_transformed_count is not None and len(transformed) != args.expected_transformed_count:
            raise ValueError(
                f"{spec.method}: expected {args.expected_transformed_count} transformed rows, "
                f"found {len(transformed)}"
            )
        result, detail_rows = evaluate_transformed_rows(
            transformed, text_column, label_column, target_patterns
        )
        result = {"method": spec.method, **result}
        for detail in detail_rows:
            detail["method"] = spec.method
        class_counts = {"low": result["low_instances"], "high": result["high_instances"]}
        if expected_class_counts is None:
            expected_class_counts = class_counts
        elif class_counts != expected_class_counts:
            raise ValueError(
                f"{spec.method}: transformed class counts {class_counts} differ from "
                f"the first dataset {expected_class_counts}"
            )
        rows.append(result)
        per_sample_rows.extend(detail_rows)
        datasets_manifest[spec.method] = {
            "path": str(spec.path),
            "sha256": file_sha256(spec.path),
            "total_instances": len(frame),
            "transformed_instances": len(transformed),
            "selection_rule": selection_rule,
            "text_column": text_column,
            "label_column": label_column,
        }
        print(f"Completed {spec.method}: {len(transformed)} transformed instances")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "configuration": {
            "target_label_mapping": {"low": [1, 2], "high": [3, 4, 5]},
            "transformed_label_mapping": {"0": "low", "1": "high"},
            "tokenisation": TOKEN_RE.pattern,
            "first_person_pronouns": sorted(FIRST_PERSON_PRONOUNS),
            "keyword_extraction": {
                "method": "mean TF-IDF within each target class",
                "max_features": 50,
                "stop_words": "english",
                "ngram_range": [1, 2],
                "selected_keywords": 10,
                "ranking": "descending mean TF-IDF; alphabetical tie-break",
            },
            "quality_score": {
                "length": "max(0, 1 - abs(L_t - L_target) / L_target)",
                "keyword": "min(1, K_m / 10)",
                "pronoun": "1 for exact first-person-pronoun token presence; 0 otherwise",
                "composite": "0.4 * Q_length + 0.4 * Q_keyword + 0.2 * Q_pronoun",
            },
            "standard_deviation": "sample standard deviation (ddof=1)",
            "uniqueness": "case-sensitive exact-text uniqueness after whitespace normalisation",
        },
        "target_corpus": {
            "path": str(target_path),
            "sha256": file_sha256(target_path),
            "text_column": target_text_column,
            "intensity_column": target_intensity_column,
            "stored_label_audit": stored_target_label_audit(
                target, target_intensity_column, target_label_column
            ),
            "patterns": {
                class_name: {
                    "instance_count": pattern.instance_count,
                    "mean_length": pattern.mean_length,
                    "keywords": list(pattern.keywords),
                }
                for class_name, pattern in target_patterns.items()
            },
        },
        "datasets": datasets_manifest,
    }
    write_outputs(rows, per_sample_rows, output_dir, manifest)
    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
