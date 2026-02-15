#!/usr/bin/env python3
"""Activation patching experiments (optional).

Patches nonce word activations with real concept activations to test
causal role of qualia-informed representations.
Stub — full implementation post-dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def run_patching(stimuli_path: Path, model_name: str, output_dir: Path):
    """Run activation patching experiments.

    For each concept:
    1. Run model on nonce word stimulus (e.g., T+A+C+F condition)
    2. Run model on matched real word stimulus
    3. Patch nonce word activations at target layer with real word activations
    4. Measure effect on downstream predictions

    Args:
        stimuli_path: Path to stimuli_final.jsonl
        model_name: HuggingFace model identifier
        output_dir: Directory for patching results
    """
    # TODO: Implement activation patching with hooks
    # TODO: Test patching at different layers
    # TODO: Measure effect on next-token predictions
    raise NotImplementedError("Full implementation post-dataset generation")


if __name__ == "__main__":
    run_patching(
        config.STIMULI / "stimuli_final.jsonl",
        config.LLAMA_MODEL,
        config.DATA / "patching_results",
    )
