#!/usr/bin/env python3
"""Hidden state extraction from experiment models.

Extracts representations at each layer for nonce word tokens across all
experimental conditions, for each model in EXPERIMENT_MODELS.

Memory-efficient: saves one memmap file per layer instead of one giant array.
Uses character offset mapping for robust nonce word detection (BPE-safe).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils


def find_nonce_positions_by_offsets(
    text: str,
    nonce: str,
    offsets: list[tuple[int, int]],
) -> list[int]:
    """Find token positions of nonce word using character offset mapping.

    BPE-safe: matches character spans rather than token ID sequences,
    so it works even when the tokenizer merges/splits differently in context.
    """
    text_lower = text.lower()
    nonce_lower = nonce.lower()

    # Find all character-level occurrences
    char_spans = []
    search_start = 0
    while True:
        idx = text_lower.find(nonce_lower, search_start)
        if idx == -1:
            break
        char_spans.append((idx, idx + len(nonce)))
        search_start = idx + 1

    if not char_spans:
        return []

    # Map character spans to token positions
    positions = []
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:  # special/padding token
            continue
        for char_start, char_end in char_spans:
            if tok_start < char_end and tok_end > char_start:
                positions.append(tok_idx)
                break

    return positions


def find_nonce_positions_fallback(
    input_ids: list[int],
    nonce_ids: list[int],
) -> list[int]:
    """Fallback: find nonce by matching token ID subsequence."""
    positions = []
    n = len(nonce_ids)
    for i in range(len(input_ids) - n + 1):
        if input_ids[i : i + n] == nonce_ids:
            positions.extend(range(i, i + n))
    return positions


def extract_for_model(
    model_name: str,
    stimuli: list[dict],
    output_dir: Path,
    batch_size: int = 8,
    max_length: int = 512,
    agg: str = "mean",
):
    """Extract hidden states for all stimuli from one model.

    Saves per-layer memmap files (n_stimuli × hidden_dim) in float16.
    """
    print(f"\n{'='*60}")
    print(f"Extracting from {model_name}")
    print(f"{'='*60}")

    model_slug = model_name.replace("/", "--")
    model_dir = output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if tokenizer supports offset mapping
    use_offsets = tokenizer.is_fast

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size
    n_stimuli = len(stimuli)

    # Create per-layer memmap files (no giant 3D array in RAM)
    layer_memmaps = {}
    for layer_idx in range(n_layers):
        path = model_dir / f"layer_{layer_idx}_{agg}.dat"
        mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(n_stimuli, hidden_dim))
        layer_memmaps[layer_idx] = mm

    valid_mask = np.zeros(n_stimuli, dtype=bool)
    stimulus_ids = []

    # Process in batches
    for batch_start in tqdm(range(0, n_stimuli, batch_size), desc="Extracting"):
        batch = stimuli[batch_start : batch_start + batch_size]
        texts = [s["stimulus"] for s in batch]

        # Tokenize batch
        tokenize_kwargs = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        if use_offsets:
            tokenize_kwargs["return_offsets_mapping"] = True

        encodings = tokenizer(texts, **tokenize_kwargs)

        # Extract offset mapping before moving to GPU
        offset_mappings = None
        if use_offsets and "offset_mapping" in encodings:
            offset_mappings = encodings.pop("offset_mapping")

        encodings = encodings.to(model.device)

        # Find nonce word positions for each item in batch
        batch_nonce_positions = []
        for i, stim in enumerate(batch):
            nonce = stim["nonce_word"]

            if offset_mappings is not None:
                offsets = offset_mappings[i].tolist()
                positions = find_nonce_positions_by_offsets(texts[i], nonce, offsets)
            else:
                nonce_ids = tokenizer.encode(nonce, add_special_tokens=False)
                input_ids = encodings["input_ids"][i].tolist()
                positions = find_nonce_positions_fallback(input_ids, nonce_ids)

            batch_nonce_positions.append(positions)

        # Forward pass
        with torch.no_grad():
            outputs = model(**encodings)

        hidden_states = outputs.hidden_states

        # Extract and write to memmap
        for i, stim in enumerate(batch):
            idx = batch_start + i
            stimulus_ids.append(stim["stimulus_id"])
            positions = batch_nonce_positions[i]

            if not positions:
                continue

            valid_mask[idx] = True
            pos_tensor = torch.tensor(positions, device=hidden_states[0].device)

            for layer_idx in range(n_layers):
                layer_hidden = hidden_states[layer_idx][i]  # (seq_len, hidden_dim)
                nonce_hidden = layer_hidden[pos_tensor]  # (n_positions, hidden_dim)

                if agg == "mean":
                    repr_vec = nonce_hidden.mean(dim=0)
                elif agg == "first":
                    repr_vec = nonce_hidden[0]
                elif agg == "last":
                    repr_vec = nonce_hidden[-1]
                else:
                    repr_vec = nonce_hidden.mean(dim=0)

                layer_memmaps[layer_idx][idx] = repr_vec.cpu().numpy().astype(np.float16)

        # Free GPU memory
        del outputs, hidden_states
        torch.cuda.empty_cache()

    # Flush and close memmaps
    for layer_idx in range(n_layers):
        layer_memmaps[layer_idx].flush()
        del layer_memmaps[layer_idx]

    # Save metadata
    metadata = {
        "stimulus_ids": stimulus_ids,
        "valid_mask": valid_mask.tolist(),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_stimuli": n_stimuli,
        "model_name": model_name,
        "aggregation": agg,
    }
    utils.save_json(model_dir / "metadata.json", metadata)

    n_valid = int(valid_mask.sum())
    print(f"  Extracted {n_valid}/{n_stimuli} stimuli "
          f"({n_stimuli - n_valid} skipped — nonce not found in tokens)")
    print(f"  Saved {n_layers} layer files to {model_dir}")

    # Free model
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from experiment models")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--agg", choices=["mean", "first", "last"], default="mean")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (default: all EXPERIMENT_MODELS)")
    args = parser.parse_args()

    stimuli_path = config.STIMULI / "stimuli_final.jsonl"
    controls_path = config.CONTROLS / "controls.jsonl"
    output_dir = config.DATA / "representations"

    # Load all stimuli + controls
    stimuli = utils.read_jsonl(stimuli_path)
    if controls_path.exists():
        controls = utils.read_jsonl(controls_path)
        stimuli.extend(controls)

    print(f"Total stimuli to process: {len(stimuli)}")

    models = config.EXPERIMENT_MODELS
    if args.model:
        models = [m for m in models if args.model in m]
        if not models:
            models = [args.model]

    for model_name in models:
        extract_for_model(
            model_name, stimuli, output_dir,
            batch_size=args.batch_size, agg=args.agg,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
