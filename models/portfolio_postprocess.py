"""Postprocessing for the direction predictor: print per-asset hit rate
and confusion-matrix metrics for train/validation/test.

Kept in its own file so training/data logic (models/portfolio_lstm.py)
stays separate from presentation (printouts). This module is a LIBRARY -
it has no CLI of its own; main.py at the repo root calls these functions
after running models/portfolio_lstm.py's pipeline, so the printout always
reflects the exact model that was just trained/loaded.

Neutral band: result.probabilities_* may contain exact 0.5 values - the
model's explicit ABSTENTIONS (see portfolio_lstm.apply_neutral_band).
Every metric printed here is computed over decided samples only, with a
`coverage` column showing how often the model made a call at all -
accuracy and coverage must always be read together.
"""

from __future__ import annotations

import logging

from models.portfolio_lstm import PredictionResult, confusion_matrix_metrics

logger = logging.getLogger(__name__)


def print_hit_rates(result: PredictionResult) -> None:
    """Print per-asset hit rate (over DECIDED samples - see
    portfolio_lstm._decided_hit_rate) for all three splits, plus the mean
    across assets."""
    print(f"{'pair':<10}{'train':>8}{'val':>8}{'test':>8}")
    for i, pair in enumerate(result.pairs):
        print(f"{pair:<10}{result.hit_rate_train[i]:>8.3f}{result.hit_rate_val[i]:>8.3f}{result.hit_rate_test[i]:>8.3f}")
    print(f"{'mean':<10}{result.hit_rate_train.mean():>8.3f}{result.hit_rate_val.mean():>8.3f}{result.hit_rate_test.mean():>8.3f}")
    if result.neutral_band > 0:
        print(f"(hit rates exclude abstentions: neutral band = 0.5 +/- {result.neutral_band:.3f})")


def print_confusion_matrices(result: PredictionResult) -> None:
    """Print the full confusion-matrix breakdown (counts + derived
    metrics: accuracy, precision, recall, specificity, F1) for each split,
    one asset per row - over DECIDED samples only, with `abst` (abstained
    count) and `cover` (coverage) columns showing how often the model
    actually made a call (see this module's docstring)."""
    band = result.neutral_band
    splits = [
        ("train", result.probabilities_train, result.direction_labels_train),
        ("validation", result.probabilities_val, result.direction_labels_val),
        ("test", result.probabilities_test, result.direction_labels_test),
    ]
    for split_name, probs, labels in splits:
        print(f"\n--- {split_name} ---")
        metrics = confusion_matrix_metrics(probs, labels, neutral_band=band)
        print(
            f"{'pair':<10}{'TP':>6}{'FP':>6}{'TN':>6}{'FN':>6}{'abst':>6}"
            f"{'cover':>8}{'acc':>8}{'prec':>8}{'recall':>8}{'spec':>8}{'f1':>8}"
        )
        for i, pair in enumerate(result.pairs):
            print(
                f"{pair:<10}{metrics['tp'][i]:>6}{metrics['fp'][i]:>6}{metrics['tn'][i]:>6}{metrics['fn'][i]:>6}"
                f"{metrics['abstained'][i]:>6}{metrics['coverage'][i]:>8.3f}"
                f"{metrics['accuracy'][i]:>8.3f}{metrics['precision'][i]:>8.3f}{metrics['recall'][i]:>8.3f}"
                f"{metrics['specificity'][i]:>8.3f}{metrics['f1'][i]:>8.3f}"
            )
