# Literature affinity audit

This directory contains a resumable audit pipeline for comparing the `KD` values in the
PCANN datasets with measurements reported in the literature linked to each PDB entry.

The pipeline deliberately separates four things:

1. deterministic dataset and PDB parsing;
2. retrieval of article metadata/text from Europe PMC;
3. evidence extraction by an OpenRouter model, optionally with web search;
4. deterministic unit conversion, `dG` calculation, and comparison.

Model output is treated as a **candidate**, not as verified ground truth. Every numeric
candidate is stored with a quote, location, article identifier, and URL so conflicts can
be reviewed.

## Inputs

The script reads the existing repository files:

- `data/train/pcann-plus-trainval.csv`
- `data/test/testAB-clean.csv`
- `data/test/test-fabs.csv`
- `data/raw/ppb-affinity/pdb/<uid>.pdb`
- `.env` (`OPENROUTER_API_KEY` only; its value is never written to output)

The train/validation folds are assignments over the same affinity rows, so every
train/validation affinity is audited once.

## Usage

From the repository root:

```powershell
# Build the 1,441-row manifest without network access
python affinity_audit/audit.py manifest

# Retrieve Europe PMC article text/abstracts and cache them
python affinity_audit/audit.py sources --workers 8

# Small end-to-end model pilot
python affinity_audit/audit.py run --limit 10 --workers 2 --web-search --max-new-cost 0.10
python affinity_audit/audit.py summarize

# Read the live account totals (the key itself is never printed)
python affinity_audit/audit.py credits

# Zero-cost pre-screen, then model only likely numeric article passages
python affinity_audit/audit.py screen
python affinity_audit/audit.py run --workers 4 --local-candidates-only --no-web-search --max-new-cost 1.00
python affinity_audit/audit.py summarize
```

Or run every phase:

```powershell
python affinity_audit/audit.py all --workers 6 --web-search
```

Defaults can be changed without editing the script:

```powershell
$env:OPENROUTER_MODEL = "google/gemini-2.5-flash"
python affinity_audit/audit.py run --workers 4 --web-search
```

Useful options:

- `--only-uid 1a22` audits a single PDB entry;
- `--limit N` limits model calls for a pilot;
- `--retry-errors` retries cached failures;
- `--no-web-search` uses only the locally retrieved article text;
- `--local-candidates-only` skips paid calls when retrieved text has no local numeric-affinity candidate;
- `--max-new-cost 1.00` stops submitting batches after one dollar of newly reported cost;
- `--work-dir PATH` selects another cache/output directory.

The cost guard defaults to `$1.00`. Because calls are submitted in batches of `--workers`,
the final batch can overshoot the limit slightly. Use fewer workers for a tighter guard.
For zero-cost experiments, use `--model openrouter/free --no-web-search`; free-model daily
rate limits still apply.

```powershell
# Spend no credits; process up to 50 new likely candidates today
python affinity_audit/audit.py run --model openrouter/free --no-web-search `
  --local-candidates-only --unprocessed-only --limit 50 --workers 2 --max-new-cost 0
```

After the free daily reset, retry errors first and then consume the rest of the quota:

```powershell
python affinity_audit/audit.py run --model nvidia/nemotron-3-super-120b-a12b:free `
  --no-web-search --local-candidates-only --errors-only --limit 15 --workers 2 --max-new-cost 0
python affinity_audit/audit.py run --model nvidia/nemotron-3-super-120b-a12b:free `
  --no-web-search --local-candidates-only --unprocessed-only --limit 35 --workers 2 --max-new-cost 0
```

## Outputs

All generated files are under `affinity_audit/work/` and ignored by Git:

```text
work/
  manifest.jsonl             normalized input rows
  sources/<uid>.json         PDB citation + Europe PMC text excerpts
  raw/<item_id>.json         complete OpenRouter response for provenance
  results/<item_id>.json     normalized candidate and comparison
  affinity_audit.csv         one row per dataset record
  summary.json               counts, coverage, model usage, estimated cost
```

The main statuses are:

- `match_candidate`: direct literature `Kd` is within 0.1 log10 units;
- `approx_match_candidate`: within 0.3 log10 units;
- `conflict_candidate`: larger discrepancy for an apparently direct measurement;
- `not_comparable`: only another construct/mutant/measurement type was found;
- `not_found`: no usable measurement was found;
- `needs_review`: ambiguous candidates or unsupported units;
- `error`: retrieval/model/parsing failure.

`conflict_candidate` is not an automatic correction. Assay conditions, construct,
mutation state, temperature, and evidence must be checked before changing source data.
