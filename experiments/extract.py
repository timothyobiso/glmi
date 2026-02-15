#!/usr/bin/env python3
"""Hidden state extraction from Llama-3.1-8B.

Extracts representations at each layer for nonce word tokens across all
experimental conditions. Stub — full implementation post-dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils


def extract_hidden_states(stimuli_path: Path, output_dir: Path, model_name: str = config.LLAMA_MODEL):
    """Extract hidden states for all stimuli from a causal LM.

    For each stimulus:
    1. Tokenize the stimulus text
    2. Run forward pass through the model
    3. Extract hidden states at nonce word token positions across all layers
    4. Save as numpy arrays indexed by stimulus_id

    Args:
        stimuli_path: Path to stimuli_final.jsonl
        output_dir: Directory to save extracted representations
        model_name: HuggingFace model identifier
    """
    # TODO: Load model with torch.no_grad()
    # TODO: Iterate over stimuli, extract hidden states at nonce token indices
    # TODO: Save per-layer representations as .npy files
    raise NotImplementedError("Full implementation post-dataset generation")


if __name__ == "__main__":
    extract_hidden_states(
        config.STIMULI / "stimuli_final.jsonl",
        config.DATA / "representations",
    )
