# Reproducing the generative-data-augmentation study

## Canonical scope

The primary reproduction target is the code in
`official/DA_and_transfer_learning/`. The similarly named scripts under
`pipeline/` are historical development variants and must not be mixed with the
publication-oriented workflow without explicit justification.

## Intended pipeline

1. Prepare compatible MEISD and ESConv tables.
2. Generate basic domain statistics.
3. Extract ESConv stylistic patterns.
4. Transform MEISD examples with LLM, rule, NLP, and hybrid methods.
5. Evaluate generated text quality.
6. Train the BERT-LSTM model on MEISD.
7. Fine-tune and evaluate on ESConv.
8. Recreate the statistical comparisons and publication tables.

Relevant scripts:

```text
official/DA_and_transfer_learning/generate_basic_domain_statistics.py
official/DA_and_transfer_learning/data_augmentation.py
official/DA_and_transfer_learning/TextAugmentationEvaluator.py
official/DA_and_transfer_learning/recompute_intrinsic_metrics.py
official/DA_and_transfer_learning/recompute_transformation_quality.py
official/DA_and_transfer_learning/binary_intensity_classification_LLM.py
```

For new intrinsic-evaluation runs, use
`recompute_intrinsic_metrics.py`. It requires every transformed record to retain
its `source_text` before it will calculate paired BLEU or CHRF, uses a causal
language model for perplexity, and writes unavailable metrics as null values
with an explanation instead of substituting the transformed text as its own
reference or returning a numerical zero. The output directory contains:

```text
intrinsic_metrics_summary.csv
intrinsic_metrics_manifest.json
intrinsic_metrics_report.md
```

The manifest records input hashes, evaluated sample counts, selection rules,
metric settings, reference availability, and the perplexity model status.

For the dissertation's transformation-quality table, use
`recompute_transformation_quality.py`. This analysis is separate from the
historical evaluator and applies the declared target mapping (scores 1--2 are
low; scores 3--5 are high). It evaluates only transformed source-domain rows,
uses multiset subtraction when explicit provenance columns are unavailable,
and implements the quality score as follows:

```text
Q_length  = max(0, 1 - abs(L_t - L_target) / L_target)
Q_keyword = min(1, K_m / 10)
Q_pronoun = 1 for exact first-person-pronoun token presence, otherwise 0
Q          = 0.4 * Q_length + 0.4 * Q_keyword + 0.2 * Q_pronoun
```

The script writes:

```text
transformation_quality_summary.csv
transformation_quality_per_sample.csv
transformation_quality_manifest.json
transformation_quality_report.md
```

The per-sample output retains text hashes and every score component. The
manifest records input hashes, target-class counts, extracted keywords, target
lengths, column selections, dependency versions, and the full scoring
definition.

## Portability warning

The historical scripts contain absolute input, output, and local LLaMA model
paths. Treat them as provenance. For a new run, make paths configurable in a
separate branch and record the change. Never regenerate a historical dataset
and assign it the same identity merely because the filename matches.

## Required run manifest

Record at least:

- Git commit;
- source and target dataset versions and SHA-256 hashes;
- label mappings for MEISD and ESConv;
- augmentation method, ratio, prompt version, and generation parameters;
- exact LLaMA model filename and checksum;
- random seeds;
- Python and dependency versions;
- GPU/CPU backend;
- training configuration;
- prediction and metric file hashes.

## Validation targets

The manuscript reports multiple augmentation families and a sequential
MEISD-to-ESConv training design. A reproduction is complete only when the data
counts, class distributions, augmentation-quality tables, classification
metrics, and statistical comparisons agree within a documented tolerance.
