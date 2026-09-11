#!/usr/bin/env python3
"""Recompute intrinsic augmentation metrics with explicit provenance.

The legacy quality-analysis script silently used each transformed sample as its
own reference when ``source_text`` was unavailable and returned ``0.0`` when
perplexity could not be computed.  This script never substitutes references and
records unavailable metrics as null values with a reason in the output manifest.

Example
-------
python recompute_intrinsic_metrics.py \
  --target-corpus esconv_both_parts.csv \
  --source-corpus filtered_negative_MEISD_intensity_max_first_25_conv.csv \
  --dataset rule=esconv_enhanced_classical_augmentation_70percent_balanced.xlsx \
  --dataset rule_llm=esconv_enhanced_mixed_augmentation_70percent_balanced.xlsx \
  --dataset llm=esconv_enhanced_llm_augmentation_70percent_balanced.xlsx \
  --dataset nlp=esconv_enhanced_nlp_augmentation_70percent_balanced.xlsx \
  --dataset nlp_llm=esconv_enhanced_llm_nlp_augmentation_70percent_balanced.xlsx \
  --output-dir analysis_results/corrected_intrinsic_metrics
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
TEXT_COLUMN_CANDIDATES = ("conversation", "utterances", "utterance", "text", "content")
SOURCE_COLUMN_CANDIDATES = ("source_text", "original_text", "source", "original")


@dataclass(frozen=True)
class DatasetSpec:
    method: str
    path: Path


def normalise_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


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
    raise ValueError(f"No text column found; available columns: {list(frame.columns)}")


def parse_dataset_spec(raw: str) -> DatasetSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Dataset must use METHOD=PATH syntax")
    method, raw_path = raw.split("=", 1)
    method = method.strip()
    path = Path(raw_path.strip()).expanduser().resolve()
    if not method:
        raise argparse.ArgumentTypeError("Dataset method name cannot be empty")
    return DatasetSpec(method=method, path=path)


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


def corpus_diversity(texts: Sequence[str]) -> dict[str, float | int | None]:
    tokenised = [tokenise(text) for text in texts]
    tokens = [token for sample in tokenised for token in sample]
    trigram_list = [gram for sample in tokenised for gram in ngrams(sample, 3).elements()]
    return {
        "token_count": len(tokens),
        "type_count": len(set(tokens)),
        "trigram_count": len(trigram_list),
        "unique_trigram_count": len(set(trigram_list)),
        "ttr": len(set(tokens)) / len(tokens) if tokens else None,
        "utr": len(set(trigram_list)) / len(trigram_list) if trigram_list else None,
    }


def closest_other_length(lengths: Sequence[int], index: int) -> int:
    own = lengths[index]
    counts = Counter(lengths)
    if counts[own] > 1:
        return own
    unique = sorted(counts)
    position = bisect.bisect_left(unique, own)
    choices: list[int] = []
    if position > 0:
        choices.append(unique[position - 1])
    if position + 1 < len(unique):
        choices.append(unique[position + 1])
    return min(choices, key=lambda value: (abs(value - own), value)) if choices else own


def brevity_penalty(candidate_length: int, reference_length: int) -> float:
    if candidate_length == 0:
        return 0.0
    if candidate_length > reference_length:
        return 1.0
    return math.exp(1.0 - reference_length / candidate_length)


def self_bleu(texts: Sequence[str], max_order: int = 4, epsilon: float = 0.1) -> float | None:
    """Compute exact mean sentence Self-BLEU without materialising O(n^2) references."""
    tokenised = [tokenise(text) for text in texts]
    if len(tokenised) < 2:
        return None

    counters_by_order = [[ngrams(tokens, order) for tokens in tokenised] for order in range(1, max_order + 1)]
    reference_maxima: list[dict[tuple[str, ...], tuple[int, int, int]]] = []
    for sentence_counters in counters_by_order:
        maxima: dict[tuple[str, ...], tuple[int, int, int]] = {}
        for counter in sentence_counters:
            for gram, count in counter.items():
                first, first_owners, second = maxima.get(gram, (0, 0, 0))
                if count > first:
                    maxima[gram] = (count, 1, first)
                elif count == first:
                    maxima[gram] = (first, first_owners + 1, second)
                elif count > second:
                    maxima[gram] = (first, first_owners, count)
        reference_maxima.append(maxima)

    lengths = [len(tokens) for tokens in tokenised]
    scores: list[float] = []
    for index, tokens in enumerate(tokenised):
        if not tokens:
            scores.append(0.0)
            continue
        log_precisions = []
        for order, sentence_counters in enumerate(counters_by_order, start=1):
            candidate_counts = sentence_counters[index]
            denominator = max(1, sum(candidate_counts.values()))
            numerator = 0
            for gram, candidate_count in candidate_counts.items():
                first, owners, second = reference_maxima[order - 1][gram]
                reference_count = second if candidate_count == first and owners == 1 else first
                numerator += min(candidate_count, reference_count)
            precision = numerator / denominator if numerator else epsilon / denominator
            log_precisions.append(math.log(precision))
        reference_length = closest_other_length(lengths, index)
        score = brevity_penalty(len(tokens), reference_length) * math.exp(sum(log_precisions) / max_order)
        scores.append(score)
    return float(np.mean(scores))


def corpus_bleu(pairs: Sequence[tuple[str, str]], max_order: int = 4, epsilon: float = 0.1) -> float | None:
    if not pairs:
        return None
    clipped = [0] * max_order
    totals = [0] * max_order
    reference_length = 0
    candidate_length = 0
    for reference, candidate in pairs:
        ref_tokens = tokenise(reference)
        cand_tokens = tokenise(candidate)
        reference_length += len(ref_tokens)
        candidate_length += len(cand_tokens)
        for order in range(1, max_order + 1):
            ref_counts = ngrams(ref_tokens, order)
            cand_counts = ngrams(cand_tokens, order)
            clipped[order - 1] += sum(min(count, ref_counts[gram]) for gram, count in cand_counts.items())
            totals[order - 1] += sum(cand_counts.values())
    precisions = [
        (match / total) if match else (epsilon / max(1, total))
        for match, total in zip(clipped, totals)
    ]
    return brevity_penalty(candidate_length, reference_length) * math.exp(
        sum(math.log(value) for value in precisions) / max_order
    )


def character_ngrams(text: str, order: int) -> Counter[str]:
    compact = re.sub(r"\s+", "", text.lower())
    return Counter(compact[i : i + order] for i in range(len(compact) - order + 1))


def corpus_chrf(pairs: Sequence[tuple[str, str]], max_order: int = 6, beta: float = 2.0) -> float | None:
    if not pairs:
        return None
    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, max_order + 1):
        common = reference_total = candidate_total = 0
        for reference, candidate in pairs:
            ref_counts = character_ngrams(reference, order)
            cand_counts = character_ngrams(candidate, order)
            common += sum((ref_counts & cand_counts).values())
            reference_total += sum(ref_counts.values())
            candidate_total += sum(cand_counts.values())
        precisions.append(common / candidate_total if candidate_total else 0.0)
        recalls.append(common / reference_total if reference_total else 0.0)
    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    return (1 + beta_squared) * precision * recall / denominator if denominator else 0.0


def mean_jaccard_novelty(texts: Sequence[str], source_texts: Sequence[str]) -> float | None:
    source_sets = [set(tokenise(text)) for text in source_texts]
    source_sets = [tokens for tokens in source_sets if tokens]
    if not source_sets:
        return None
    scores: list[float] = []
    for text in texts:
        generated = set(tokenise(text))
        if not generated:
            scores.append(0.0)
            continue
        maximum = max(
            len(generated & source) / len(generated | source)
            for source in source_sets
        )
        scores.append(1.0 - maximum)
    return float(np.mean(scores))


@lru_cache(maxsize=None)
def load_causal_lm(model_name: str, device_name: str):
    """Load a causal language model once and reuse it across augmentation methods."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return torch, tokenizer, model, device


def causal_lm_perplexity(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device_name: str,
) -> tuple[float | None, dict[str, Any]]:
    try:
        torch, tokenizer, model, device = load_causal_lm(model_name, device_name)
    except ImportError as exc:
        return None, {"status": "unavailable", "reason": f"missing dependency: {exc.name}"}
    except Exception as exc:  # model cache/network errors must be recorded, not converted to 0
        return None, {"status": "unavailable", "reason": f"model initialisation failed: {exc}"}
    total_negative_log_likelihood = 0.0
    total_predicted_tokens = 0
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            labels = encoded["input_ids"].clone()
            labels[encoded["attention_mask"] == 0] = -100
            output = model(**encoded, labels=labels)
            predicted_tokens = int((encoded["attention_mask"].sum(dim=1) - 1).clamp(min=0).sum().item())
            if predicted_tokens:
                total_negative_log_likelihood += float(output.loss.item()) * predicted_tokens
                total_predicted_tokens += predicted_tokens
    if total_predicted_tokens == 0:
        return None, {"status": "unavailable", "reason": "no predicted tokens"}
    perplexity = math.exp(total_negative_log_likelihood / total_predicted_tokens)
    return perplexity, {
        "status": "computed",
        "model": model_name,
        "device": str(device),
        "predicted_tokens": total_predicted_tokens,
        "max_length": max_length,
        "batch_size": batch_size,
    }


def source_pairs(frame: pd.DataFrame, transformed_text_column: str) -> tuple[list[tuple[str, str]], str | None]:
    lower_to_original = {str(column).lower(): str(column) for column in frame.columns}
    source_column = next((lower_to_original[name] for name in SOURCE_COLUMN_CANDIDATES if name in lower_to_original), None)
    if source_column is None:
        return [], None
    pairs = []
    for source, transformed in zip(frame[source_column], frame[transformed_text_column]):
        source_text = normalise_text(source)
        transformed_text = normalise_text(transformed)
        if source_text and transformed_text:
            pairs.append((source_text, transformed_text))
    return pairs, source_column


def format_number(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "intrinsic_metrics_summary.csv"
    fieldnames = list(rows[0]) if rows else []
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "intrinsic_metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    columns = [
        "method", "total_instances", "transformed_instances", "full_set_uniqueness_ratio",
        "transformed_uniqueness_ratio",
        "self_bleu", "ttr", "utr", "novelty", "bleu", "chrf", "perplexity",
    ]
    lines = [
        "# Corrected intrinsic augmentation metrics",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_number(row[column]) for column in columns) + " |")
    lines.extend([
        "",
        "`NA` means that the metric was not computed. See `intrinsic_metrics_manifest.json` for the reason and full configuration.",
        "",
    ])
    (output_dir / "intrinsic_metrics_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec, required=True, metavar="METHOD=PATH")
    parser.add_argument("--target-corpus", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path)
    parser.add_argument("--text-column")
    parser.add_argument("--target-text-column")
    parser.add_argument("--source-text-column")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results/corrected_intrinsic_metrics"))
    parser.add_argument("--perplexity-model", default="gpt2")
    parser.add_argument("--perplexity-batch-size", type=int, default=8)
    parser.add_argument("--perplexity-max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    target_path = args.target_corpus.expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target corpus not found: {target_path}")
    target_frame = read_table(target_path)
    target_column = resolve_column(target_frame, args.target_text_column, TEXT_COLUMN_CANDIDATES)
    target_texts_list = [normalise_text(value) for value in target_frame[target_column]]
    target_text_counts = Counter(text.lower() for text in target_texts_list if text)

    source_texts_list: list[str] = []
    source_metadata: dict[str, Any] | None = None
    if args.source_corpus:
        source_path = args.source_corpus.expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source corpus not found: {source_path}")
        source_frame = read_table(source_path)
        source_column = resolve_column(source_frame, args.source_text_column, TEXT_COLUMN_CANDIDATES)
        source_texts_list = [normalise_text(value) for value in source_frame[source_column] if normalise_text(value)]
        source_metadata = {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "text_column": source_column,
            "instances": len(source_texts_list),
        }

    rows: list[dict[str, Any]] = []
    dataset_manifest: dict[str, Any] = {}
    for spec in args.dataset:
        print(f"Evaluating {spec.method}...")
        if not spec.path.exists():
            raise FileNotFoundError(f"Augmented dataset not found: {spec.path}")
        frame = read_table(spec.path)
        text_column = resolve_column(frame, args.text_column, TEXT_COLUMN_CANDIDATES)
        all_texts = [normalise_text(value) for value in frame[text_column] if normalise_text(value)]
        transformed, selection_rule = select_transformed_rows(frame, text_column, target_text_counts)
        transformed_texts = [normalise_text(value) for value in transformed[text_column] if normalise_text(value)]
        pairs, source_column = source_pairs(transformed, text_column)

        diversity = corpus_diversity(transformed_texts)
        pair_status = {
            "status": "computed" if pairs else "unavailable",
            "paired_instances": len(pairs),
            "source_column": source_column,
            "reason": None if pairs else "no retained source_text column; references were not substituted",
        }
        perplexity: float | None = None
        if args.skip_perplexity:
            perplexity_status = {"status": "skipped", "reason": "--skip-perplexity"}
        else:
            perplexity, perplexity_status = causal_lm_perplexity(
                transformed_texts,
                model_name=args.perplexity_model,
                batch_size=args.perplexity_batch_size,
                max_length=args.perplexity_max_length,
                device_name=args.device,
            )

        row = {
            "method": spec.method,
            "total_instances": len(frame),
            "transformed_instances": len(transformed_texts),
            "paired_instances": len(pairs),
            "full_set_uniqueness_ratio": len(set(all_texts)) / len(all_texts) if all_texts else None,
            "transformed_uniqueness_ratio": (
                len(set(transformed_texts)) / len(transformed_texts) if transformed_texts else None
            ),
            "self_bleu": self_bleu(transformed_texts),
            "ttr": diversity["ttr"],
            "utr": diversity["utr"],
            "novelty": mean_jaccard_novelty(transformed_texts, source_texts_list) if source_texts_list else None,
            "bleu": corpus_bleu(pairs),
            "chrf": corpus_chrf(pairs),
            "perplexity": perplexity,
        }
        rows.append(row)
        dataset_manifest[spec.method] = {
            "path": str(spec.path),
            "sha256": file_sha256(spec.path),
            "text_column": text_column,
            "selection_rule": selection_rule,
            "total_instances": len(frame),
            "transformed_instances": len(transformed_texts),
            "pairwise_metrics": pair_status,
            "perplexity": perplexity_status,
        }
        print(f"Completed {spec.method} ({len(transformed_texts)} transformed instances).")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "configuration": {
            "seed": args.seed,
            "tokenisation": TOKEN_RE.pattern,
            "self_bleu": {"orders": 4, "smoothing_epsilon": 0.1, "references": "all other transformed samples"},
            "bleu": {"orders": 4, "smoothing_epsilon": 0.1, "requires_retained_source_text": True},
            "chrf": {"character_orders": 6, "beta": 2.0, "whitespace": False, "requires_retained_source_text": True},
            "novelty": "mean one-minus-maximum token-set Jaccard similarity to any source-corpus sample",
        },
        "target_corpus": {
            "path": str(target_path),
            "sha256": file_sha256(target_path),
            "text_column": target_column,
            "instances": len(target_texts_list),
        },
        "source_corpus": source_metadata,
        "datasets": dataset_manifest,
    }
    write_outputs(args.output_dir.resolve(), rows, manifest)
    print(f"Saved corrected metrics to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
