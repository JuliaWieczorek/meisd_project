# MEISD Emotion-Intensity Research Workspace

Historical research workspace for emotion, emotion-intensity, sentiment,
augmentation, and cross-dataset transfer experiments involving MEISD and
ESConv.

## Primary publication supported by this repository

**Leveraging Generative Artificial Intelligence for Enhanced Data Augmentation
in Emotion Intensity Classification: A Comprehensive Framework for
Cross-Dataset Transfer Learning** (Springer book chapter, accepted, 2026).

The publication studies LLaMA-2-based, rule-based, NLP-based, and hybrid data
augmentation, followed by sequential MEISD-to-ESConv transfer with a BERT-LSTM
classifier. The publication-oriented implementation is under:

```text
official/DA_and_transfer_learning/
```

In particular:

- `data_augmentation.py` implements target-style analysis and the augmentation
  variants;
- `TextAugmentationEvaluator.py` evaluates generated text;
- `binary_intensity_classification_LLM.py` implements the two-phase BERT-LSTM
  classification workflow;
- `generate_basic_domain_statistics.py` compares the source and target domains.

## Relationship to the multi-task repository

This repository also contains the historical development tree
`pipeline/EMOTIA/` for joint emotion, intensity, and sentiment modelling. The
cleaner publication-facing snapshot is maintained separately in
`mtl-emotion-intensity-sentiment`.

For citation and reproduction of **Multi-Task Aware Learning for Joint Emotion,
Intensity, and Sentiment Analysis**, use the dedicated repository. Retain this
copy as provenance for the earlier development history.

## Repository map

```text
official/    publication-oriented augmentation and transfer-learning code
pipeline/    exploratory pipelines, notebooks, analyses, and EMOTIA development
data/        locally tracked prepared MEISD data and conversion utilities
project/     early modular emotion-tagging prototype
chatbot/     experimental demonstration application
```

The directories outside `official/` are not a single reproducible pipeline.
They contain successive experiments and prototypes and should not be run in
bulk.

## Reproducibility status

This is a **historical research workspace**, not a packaged Python project.
Some scripts contain machine-specific paths and the exact historical
environments are not pinned. The repository may also omit generated datasets,
local LLaMA weights, trained checkpoints, and large result artifacts.

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before attempting to reproduce the
book-chapter results.

## Data and ethics

Respect the licences and access conditions of MEISD and ESConv. Do not commit
restricted raw conversations, model weights, API credentials, generated
sensitive text, or participant-level predictions. A public release should
contain code, configuration, schemas, and hashes or retrieval instructions
rather than unauthorised copies of source data.

## Maintenance policy

- Preserve the current history as the record of manuscript development.
- Put publication corrections in small, documented commits.
- Move or rename legacy files only in a dedicated cleanup branch.
- Do not silently replace the implementation that generated reported results.

