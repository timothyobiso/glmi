#!/usr/bin/env python3
"""Layer-wise logistic regression probes.

Trains probes at each layer to predict the target concept from nonce word
representations, for each experiment model.
Stub — full implementation post-dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def train_probes(representations_dir: Path, output_dir: Path):
    """Train layer-wise logistic regression probes.

    For each model in config.EXPERIMENT_MODELS:
      For each layer:
        1. Train logistic regression to predict target concept from nonce repr
        2. Evaluate accuracy across conditions
        3. Compare probe accuracy between models

    Args:
        representations_dir: Path to extracted representations
        output_dir: Directory for probe results
    """
    # TODO: For each model:
    #   - Split data into train/test (stratified by concept)
    #   - Train sklearn LogisticRegression per layer
    #   - Evaluate accuracy, F1 by condition
    #   - Plot accuracy by layer and condition, per model
    #   - Compare cross-model probe transferability
    raise NotImplementedError("Full implementation post-dataset generation")


if __name__ == "__main__":
    train_probes(
        config.DATA / "representations",
        config.DATA / "probing_results",
    )
