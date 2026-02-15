# GL-Qualia Nonce Word Dataset

Large-scale nonce word dataset grounded in Generative Lexicon (GL) theory, for studying how LLM internal representations converge toward target concepts as qualia information accumulates. Built for CoNLL 2026.

## Overview

The dataset pairs **nonce words** (phonotactically legal pseudowords like *dax*, *blicket*) with sentences that progressively reveal a target concept's identity through its four GL qualia roles:

| Qualia Role | What it encodes | Example (knife) |
|---|---|---|
| **Telic** | Purpose / function | "used for cutting" |
| **Agentive** | Origin / how it's made | "forged from steel" |
| **Constitutive** | Parts / materials | "has a blade and handle" |
| **Formal** | Category / type | "a type of utensil" |

The core question: as more qualia roles are revealed, does the LLM's representation of the nonce word converge toward the real concept? And does it matter which roles are provided, and in what order?

## Experimental Design

### Conditions

- **Single-qualia:** T, A, C, F (one sentence each)
- **Accumulation (all 24 orderings):** Every permutation of the 4 roles × 4 steps (e.g., T, T+A, T+A+C, T+A+C+F; A, A+C, A+C+T, A+C+T+F; etc.)
- **Combinations:** All 6 two-role and 4 three-role combos
- **Controls:**
  - Real word (replace nonce with actual concept)
  - Conflicting qualia (mix qualia from different concepts)
  - Scrambled (shuffled non-nonce words)
  - Bare nonce ("I saw a dax." — no qualia info)
  - Information-matched flat (same fillers, no qualia framing)
  - Information-matched swapped (sentences in wrong role order)

### Evaluation

- **Absolute:** Cosine similarity between nonce word representation and target concept reference, across layers
- **Discriminative:** Rank target among 10 distractors (5 hard same-category + 5 random) by cosine sim — MRR, Hits@1, Hits@3
- **Cross-architecture:** Compared across Llama-3.1-8B and Mistral-7B-v0.3

## Pipeline

```
[Seed nouns] ──┬──→ 01_extract_conceptnet  (~3-5 min, local dump)
               ├──→ 02_extract_wordnet     (<1 min)
               ├──→ 03_extract_bso         (<1 min, FrameNet)
               └──→ 05_generate_nonce      (<1 min)
                        │
                  04_merge_ontology ←── merges 01+02+03, filters to ≥4 roles
                        │
                  06_generate_templates ←── 04 + 05
                        │
                  07_naturalize_sentences ←── local LLM via vLLM (~10-30 min)
                        │
                  08_validate_stimuli
                        │
                  09_build_conditions ──→ stimuli_final.jsonl + controls.jsonl
```

Everything runs locally — no API keys needed.

## Quick Start

```bash
# Run full pipeline (uses vLLM for naturalization)
./run.sh

# Or with HuggingFace transformers fallback (slower, no vLLM needed):
./run.sh --transformers
```

The script handles `uv` setup, dependency installation, NLTK data downloads, ConceptNet dump download (~600MB), and Brysbaert concreteness ratings automatically.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (installed automatically by `run.sh` if missing)
- GPU with enough VRAM for Llama-3.1-8B-Instruct (~16GB for vLLM, less with `--transformers` + float16)
- HuggingFace access to `meta-llama/Llama-3.1-8B-Instruct` (naturalization), `meta-llama/Llama-3.1-8B` and `mistralai/Mistral-7B-v0.3` (experiments)
- ~2GB disk for ConceptNet dump + cached index

## Data Sources

- **[ConceptNet 5](https://conceptnet.io/)** — crowdsourced commonsense knowledge graph (local CSV dump, primary source)
- **[WordNet](https://wordnet.princeton.edu/)** — lexical database (hypernyms, meronyms, gloss parsing)
- **[FrameNet](https://framenet.icsi.berkeley.edu/)** — frame semantics (supplementary, low yield for nouns)
- **[Brysbaert et al. (2014)](https://link.springer.com/article/10.3758/s13428-013-0403-5)** — concreteness ratings for seed noun selection

## Output Schema

Each record in `data/stimuli/stimuli_final.jsonl`:

```json
{
  "stimulus_id": "stim_00042",
  "condition_type": "accumulation",
  "condition_label": "T+A",
  "ordering": "TACF",
  "concept": "knife",
  "nonce_word": "blicket",
  "qualia_roles": ["telic", "agentive"],
  "stimulus": "She used the blicket to slice the bread. The blicket was forged by a blacksmith.",
  "sentences": ["She used the blicket to slice the bread.", "The blicket was forged by a blacksmith."],
  "nonce_word_indices": [4, 10],
  "distractors": ["sword", "fork", "spoon", "hammer", "pen", "scissors", "axe", "saw", "chisel", "needle"],
  "hard_distractors": ["sword", "fork", "spoon", "hammer", "scissors"]
}
```

## Project Structure

```
├── run.sh                 # Setup + full pipeline runner
├── pyproject.toml         # Dependencies (managed by uv)
├── config.py              # Mappings, thresholds, paths
├── utils.py               # Shared utilities (ConceptNet index, JSONL I/O)
├── scripts/
│   ├── 01_extract_conceptnet.py   # Local ConceptNet CSV → qualia
│   ├── 02_extract_wordnet.py      # WordNet → qualia
│   ├── 03_extract_bso.py          # FrameNet → qualia
│   ├── 04_merge_ontology.py       # Merge + filter + LLM gap fill
│   ├── 05_generate_nonce_words.py # Phonotactic pseudowords
│   ├── 06_generate_templates.py   # Templated sentences
│   ├── 07_naturalize_sentences.py # Local LLM naturalization (vLLM)
│   ├── 08_validate_stimuli.py     # Validation + dedup
│   └── 09_build_conditions.py     # Experimental conditions + controls
├── experiments/           # Analysis stubs (post-dataset)
│   ├── extract.py         # Hidden state extraction (Llama + Mistral)
│   ├── references.py      # Reference representations
│   ├── rsa.py             # Cosine similarity + discriminative eval
│   ├── probing.py         # Layer-wise probes
│   └── patching.py        # Activation patching
└── data/
    ├── raw/               # ConceptNet dump, source extractions
    ├── ontology/          # Merged qualia ontology
    ├── nonce_words/       # Generated pseudowords
    ├── stimuli/           # Templates → naturalized → validated → final
    └── controls/          # Control conditions + distractor map
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONCRETENESS_THRESHOLD` | 4.0 | Minimum Brysbaert concreteness rating |
| `TARGET_CONCEPTS` | 500 | Target number of concepts with all 4 qualia roles |
| `TARGET_NONCE_WORDS` | 100 | Number of nonce words to generate |
| `MIN_NONCE_TOKENS` / `MAX_NONCE_TOKENS` | 2 / 4 | Subword token count bounds (validated against both models) |
| `NATURALIZE_MODEL` | meta-llama/Llama-3.1-8B-Instruct | Local model for sentence naturalization |
| `EXPERIMENT_MODELS` | Llama-3.1-8B, Mistral-7B-v0.3 | Models for hidden state extraction |
| `N_HARD_DISTRACTORS` | 5 | Same-category distractors (WordNet hypernym match) |
| `N_RANDOM_DISTRACTORS` | 5 | Random distractors from remaining pool |
| `JACCARD_DEDUP_THRESHOLD` | 0.8 | Near-duplicate removal threshold |
