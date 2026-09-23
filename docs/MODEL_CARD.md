# Model Card — Multilingual Sentiment Transformer Lab

## Summary

This repository contains two related experiment tracks:

1. a compact TensorFlow Transformer trained on English IMDB reviews;
2. an XLM-R fine-tuning pipeline intended for labeled Persian-English data.

It is a research and portfolio project, not a deployed universal sentiment service.

## Verified result

The preserved IMDB notebook records 85.99% accuracy and 0.3346 loss on the 25,000-example English test split. That result cannot be transferred to Persian, code-switched text, or other domains without a separate evaluation.

## Intended use

- learning Transformer classification architecture;
- comparing English and multilingual transfer-learning tracks;
- experimenting with Persian-English review sentiment;
- practicing reproducible evaluation and error analysis.

## Out-of-scope use

- decisions about employment, credit, healthcare, policing, or access to services;
- inferring a person's mental state or protected characteristics;
- treating sentiment as an objective measurement of human intent;
- production use without domain-specific validation and monitoring.

## Data requirements for the multilingual track

A credible benchmark should include balanced positive and negative examples for each language, native Persian writing, informal spelling, emoji, negation, sarcasm, named entities, transliterated Persian, and Persian-English code-switching. Splits should prevent duplicates or near-duplicates from crossing train and evaluation sets.

## Metrics

Report accuracy together with precision, recall, F1, confusion matrices, and per-language slices. If probabilities drive downstream decisions, measure calibration and choose thresholds on validation data rather than on the test set.

## Limitations

Sentiment is context-dependent. Domain shift, irony, cultural references, long documents, spelling variation, mixed scripts, and class imbalance may substantially reduce performance. Script detection in this repository is a deterministic routing utility, not full language identification.

