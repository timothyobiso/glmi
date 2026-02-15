#!/usr/bin/env bash
set -euo pipefail

# GL-Qualia Nonce Word Dataset Generation
# Full setup + sequential pipeline execution
# All processing is local — no API keys needed.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "══════════════════════════════════════════════════════════"
echo "  GL-Qualia Nonce Word Dataset Generation Pipeline"
echo "══════════════════════════════════════════════════════════"

# ── Step 0: Setup ──────────────────────────────────────────────────────────────
echo ""
echo "▸ Step 0: Project setup"

# Create directory tree
mkdir -p data/{raw,ontology,nonce_words,stimuli,controls}
mkdir -p scripts experiments

# Initialize uv project and install deps
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "  Installing Python dependencies..."
uv sync

# Download NLTK data
echo "  Downloading NLTK data..."
uv run python -c "
import nltk
for pkg in ['wordnet', 'omw-1.4', 'words', 'averaged_perceptron_tagger']:
    nltk.download(pkg, quiet=True)
    print(f'  ✓ {pkg}')
try:
    nltk.download('framenet_v17', quiet=True)
    print('  ✓ framenet_v17')
except Exception as e:
    print(f'  ⚠ framenet_v17 skipped ({e})')
try:
    nltk.download('names', quiet=True)
    print('  ✓ names')
except:
    pass
"

# Download Brysbaert concreteness ratings if not present
if [ ! -f data/raw/brysbaert_concreteness.csv ]; then
    echo "  Downloading Brysbaert concreteness ratings..."
    curl -L -o data/raw/brysbaert_concreteness.csv \
        'https://github.com/ArtsEngine/concreteness/raw/master/data/concrete.csv' \
        2>/dev/null || {
        echo "  ⚠ Could not download concreteness ratings automatically."
        echo "    Please download manually to data/raw/brysbaert_concreteness.csv"
    }
fi

# Download ConceptNet assertions dump if not present (~600MB compressed)
if [ ! -f data/raw/conceptnet-assertions-5.7.0.csv.gz ]; then
    echo "  Downloading ConceptNet assertions dump (~600MB)..."
    wget -O data/raw/conceptnet-assertions-5.7.0.csv.gz \
        'https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz' \
        || curl -L -o data/raw/conceptnet-assertions-5.7.0.csv.gz \
        'https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz'
fi

echo "  ✓ Setup complete"

# ── Steps 1-3 + 5: Extraction (all fast with local data) ─────────────────────
echo ""
echo "▸ Steps 1-3: Extracting qualia from ConceptNet, WordNet, FrameNet"

# ConceptNet (local dump — builds index on first run, ~3-5 min, then cached)
uv run python scripts/01_extract_conceptnet.py &
PID_CN=$!

# WordNet and FrameNet in parallel (both fast)
uv run python scripts/02_extract_wordnet.py &
PID_WN=$!

uv run python scripts/03_extract_bso.py &
PID_BSO=$!

# Nonce word generation in parallel (independent)
echo ""
echo "▸ Step 5: Generating nonce words (parallel with extraction)"
uv run python scripts/05_generate_nonce_words.py &
PID_NONCE=$!

# Wait for all
echo "  Waiting for WordNet..."
wait $PID_WN && echo "  ✓ WordNet extraction complete" || echo "  ✗ WordNet extraction failed"

echo "  Waiting for FrameNet..."
wait $PID_BSO && echo "  ✓ FrameNet extraction complete" || echo "  ✗ FrameNet extraction failed"

echo "  Waiting for nonce words..."
wait $PID_NONCE && echo "  ✓ Nonce word generation complete" || echo "  ✗ Nonce word generation failed"

echo "  Waiting for ConceptNet..."
wait $PID_CN && echo "  ✓ ConceptNet extraction complete" || echo "  ✗ ConceptNet extraction failed"

# ── Step 4: Merge ─────────────────────────────────────────────────────────────
echo ""
echo "▸ Step 4: Merging ontology sources"
uv run python scripts/04_merge_ontology.py
echo "  ✓ Ontology merge complete"

# ── Step 6: Templates ─────────────────────────────────────────────────────────
echo ""
echo "▸ Step 6: Generating templated sentences"
uv run python scripts/06_generate_templates.py
echo "  ✓ Template generation complete"

# ── Step 7: Naturalize ────────────────────────────────────────────────────────
echo ""
echo "▸ Step 7: Naturalizing sentences with local LLM"
echo "  (Uses vLLM by default; pass --transformers for HF pipeline fallback)"
if [ "${1:-}" = "--transformers" ]; then
    uv run python scripts/07_naturalize_sentences.py --transformers
else
    uv run python scripts/07_naturalize_sentences.py
fi
echo "  ✓ Naturalization complete"

# ── Step 8: Validate ──────────────────────────────────────────────────────────
echo ""
echo "▸ Step 8: Validating stimuli"
uv run python scripts/08_validate_stimuli.py
echo "  ✓ Validation complete"

# ── Step 9: Build conditions ──────────────────────────────────────────────────
echo ""
echo "▸ Step 9: Building experimental conditions and controls"
uv run python scripts/09_build_conditions.py
echo "  ✓ Conditions built"

# ── Verification ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Verification"
echo "══════════════════════════════════════════════════════════"

echo ""
echo "Output files:"
for f in \
    data/raw/concrete_nouns.csv \
    data/raw/conceptnet_extract.jsonl \
    data/raw/wordnet_extract.jsonl \
    data/raw/bso_extract.jsonl \
    data/ontology/concept_qualia_merged.json \
    data/nonce_words/nonce_words.jsonl \
    data/stimuli/templates_raw.jsonl \
    data/stimuli/stimuli_naturalized.jsonl \
    data/stimuli/stimuli_validated.jsonl \
    data/stimuli/stimuli_final.jsonl \
    data/controls/controls.jsonl \
    data/stimuli/dataset_statistics.json; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f" 2>/dev/null || echo "?")
        echo "  ✓ $f ($lines lines)"
    else
        echo "  ✗ $f MISSING"
    fi
done

# Validate JSONL
echo ""
echo "Validating JSONL format..."
uv run python -c "
import json
for fname in ['data/stimuli/stimuli_final.jsonl', 'data/controls/controls.jsonl']:
    try:
        with open(fname) as f:
            records = [json.loads(l) for l in f if l.strip()]
        print(f'  ✓ {fname}: {len(records)} valid records')
    except Exception as e:
        print(f'  ✗ {fname}: {e}')
"

echo ""
echo "Dataset statistics:"
uv run python -c "
import json
stats = json.load(open('data/stimuli/dataset_statistics.json'))
print(f'  Total stimuli:  {stats[\"total_stimuli\"]}')
print(f'  Total controls: {stats[\"total_controls\"]}')
print(f'  Concepts:       {stats[\"unique_concepts\"]}')
" 2>/dev/null || echo "  (statistics file not available)"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "══════════════════════════════════════════════════════════"
