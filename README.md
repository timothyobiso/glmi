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

The core question: as more qualia roles are revealed (T → T+A → T+A+C → T+A+C+F), does the LLM's representation of the nonce word converge toward the real concept?

## Experimental Conditions

- **Single-qualia:** T, A, C, F (one sentence each)
- **Accumulation:** T, T+A, T+A+C, T+A+C+F (concatenated)
- **Combinations:** All 2-role and 3-role combos (10 total)
- **Controls:** Real word, conflicting qualia, scrambled, bare nonce

## Pipeline

```
[Seed nouns] ──┬──→ 01_extract_conceptnet  (~40 min, rate-limited)
               ├──→ 02_extract_wordnet     (<1 min)
               ├──→ 03_extract_bso         (<1 min, FrameNet)
               └──→ 05_generate_nonce      (<1 min)
                        │
                  04_merge_ontology ←── merges 01+02+03, filters to ≥4 roles
                        │
                  06_generate_templates ←── 04 + 05
                        │
                  07_naturalize_sentences ←── Claude Batch API (~1-2 hr)
                        │
                  08_validate_stimuli
                        │
                  09_build_conditions ──→ stimuli_final.jsonl + controls.jsonl
```

## Quick Start

```bash
# Prerequisites: ANTHROPIC_API_KEY set in env
export ANTHROPIC_API_KEY=sk-...

# Run full pipeline
./run.sh

# Or with async concurrent Claude calls instead of batch:
./run.sh --concurrent
```

The script handles `uv` setup, dependency installation, NLTK data downloads, and Brysbaert concreteness ratings automatically.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (installed automatically by `run.sh` if missing)
- `ANTHROPIC_API_KEY` for sentence naturalization (step 07) and optional agentive gap filling (step 04)
- HuggingFace access to `meta-llama/Llama-3.1-8B` for nonce word tokenizer validation (step 05)
- Internet access for ConceptNet API (step 01)

## Data Sources

- **[ConceptNet 5](https://conceptnet.io/)** — crowdsourced commonsense knowledge graph (primary source)
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
  "concept": "knife",
  "nonce_word": "blicket",
  "qualia_roles": ["telic", "agentive"],
  "stimulus": "She used the blicket to slice the bread. The blicket was forged by a blacksmith.",
  "sentences": ["She used the blicket to slice the bread.", "The blicket was forged by a blacksmith."],
  "nonce_word_indices": [4, 10]
}
```

## Project Structure

```
├── run.sh                 # Setup + full pipeline runner
├── pyproject.toml         # Dependencies (managed by uv)
├── config.py              # Mappings, thresholds, paths
├── utils.py               # Shared utilities
├── scripts/
│   ├── 01_extract_conceptnet.py
│   ├── 02_extract_wordnet.py
│   ├── 03_extract_bso.py
│   ├── 04_merge_ontology.py
│   ├── 05_generate_nonce_words.py
│   ├── 06_generate_templates.py
│   ├── 07_naturalize_sentences.py
│   ├── 08_validate_stimuli.py
│   └── 09_build_conditions.py
├── experiments/           # Analysis stubs (post-dataset)
│   ├── extract.py         # Hidden state extraction
│   ├── references.py      # Reference representations
│   ├── rsa.py             # Cosine similarity analysis
│   ├── probing.py         # Layer-wise probes
│   └── patching.py        # Activation patching
└── data/
    ├── raw/               # Source extractions + caches
    ├── ontology/          # Merged qualia ontology
    ├── nonce_words/       # Generated pseudowords
    ├── stimuli/           # Templates → naturalized → validated → final
    └── controls/          # Control conditions
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONCRETENESS_THRESHOLD` | 4.0 | Minimum Brysbaert concreteness rating |
| `TARGET_CONCEPTS` | 500 | Target number of concepts with all 4 qualia roles |
| `TARGET_NONCE_WORDS` | 100 | Number of nonce words to generate |
| `MIN_NONCE_TOKENS` / `MAX_NONCE_TOKENS` | 2 / 4 | Llama subword token count bounds |
| `CLAUDE_MODEL` | claude-sonnet-4-5-20250929 | Model for naturalization |
| `JACCARD_DEDUP_THRESHOLD` | 0.8 | Near-duplicate removal threshold |
