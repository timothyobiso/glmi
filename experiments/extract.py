#!/usr/bin/env python3
"""Hidden state extraction from experiment models.

Extracts representations at each layer for nonce word tokens across all
experimental conditions, for each model in EXPERIMENT_MODELS.
Stub — full implementation post-dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils


def extract_hidden_states(stimuli_path: Path, output_dir: Path):
    """Extract hidden states for all stimuli from each experiment model.

    For each model in config.EXPERIMENT_MODELS:
      For each stimulus:
        1. Tokenize the stimulus text
        2. Run forward pass through the model
        3. Extract hidden states at nonce word token positions across all layers
        4. Save as numpy arrays indexed by stimulus_id

    Token aggregation strategy: save first, last, and mean of nonce word
    subword token representations separately for downstream comparison.

    Args:
        stimuli_path: Path to stimuli_final.jsonl
        output_dir: Directory to save extracted representations
    """
    # TODO: For each model in config.EXPERIMENT_MODELS:
    #   - Load model with output_hidden_states=True
    #   - Map word-level nonce indices to subword token positions
    #   - Extract hidden states at those positions (all layers)
    #   - Save {model_name}/{stimulus_id}.npz with keys: first, last, mean per layer
    raise NotImplementedError("Full implementation post-dataset generation")


if __name__ == "__main__":
    extract_hidden_states(
        config.STIMULI / "stimuli_final.jsonl",
        config.DATA / "representations",
    )
