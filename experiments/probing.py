#!/usr/bin/env python3
"""Layer-wise logistic regression probes.

Trains probes at each layer to predict the target concept from nonce word
representations. Tests whether qualia accumulation improves decodability.
Stub — full implementation post-dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def train_probes(representations_dir: Path, output_dir: Path):
    """Train layer-wise logistic regression probes.

    For each layer:
    1. Train logistic regression to predict target concept from nonce word representation
    2. Evaluate accuracy across conditions (more qualia = higher accuracy?)
    3. Identify which layers encode concept identity most strongly

    Args:
        representations_dir: Path to extracted representations
        output_dir: Directory for probe results
    """
    # TODO: Split data into train/test
    # TODO: Train sklearn LogisticRegression per layer
    # TODO: Evaluate accuracy, F1, confusion matrices
    # TODO: Plot accuracy by layer and condition
    raise NotImplementedError("Full implementation post-dataset generation")


if __name__ == "__main__":
    train_probes(
        config.DATA / "representations",
        config.DATA / "probing_results",
    )
