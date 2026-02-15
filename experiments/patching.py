#!/usr/bin/env python3
"""Activation patching experiments (optional).

Patches nonce word activations with real concept activations to test
causal role of qualia-informed representations, for each experiment model.
Stub — full implementation post-dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def run_patching(stimuli_path: Path, output_dir: Path):
    """Run activation patching experiments.

    For each model in config.EXPERIMENT_MODELS:
      For each concept:
        1. Run model on nonce word stimulus (e.g., T+A+C+F condition)
        2. Run model on matched real word stimulus
        3. Patch nonce word activations at target layer with real word activations
        4. Measure effect on downstream predictions

    Args:
        stimuli_path: Path to stimuli_final.jsonl
        output_dir: Directory for patching results
    """
    # TODO: For each model:
    #   - Implement activation patching with forward hooks
    #   - Test patching at different layers
    #   - Measure effect on next-token predictions
    #   - Compare patching effects across models
    raise NotImplementedError("Full implementation post-dataset generation")


if __name__ == "__main__":
    run_patching(
        config.STIMULI / "stimuli_final.jsonl",
        config.DATA / "patching_results",
    )
