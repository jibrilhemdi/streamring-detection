# StreamRing

StreamRing is a reproducible research artifact for real-time fraud ring detection on Ethereum Classic. The public artifact contains the paper, source code, configuration, tests, and published results. Raw blockchain data, generated graphs, credentials, virtual environments, and contributor-local files are intentionally excluded.

## Public artifact layout

```text
streamring/
├── configs/default.yaml              # Public defaults and block ranges
├── configs/local.example.yaml        # Copy to configs/local.yaml for private overrides
├── .env.example                      # Copy to .env.local for private Google auth
├── README.md                         # This overview
├── REPRODUCIBILITY.md                # Reviewer/public replication guide
├── pyproject.toml
├── requirements.txt
├── run_all.sh                        # Deterministic experiment entry point
├── scripts/
│   ├── extract_data.sh               # Optional BigQuery extraction
│   ├── build_graphs.sh               # Optional graph construction
│   └── generate_patterns.py          # Optional Tier 1 pattern features
├── src/                              # Core library code
├── experiments/                      # Reproducible experiment entry points
├── tests/                            # Unit and artifact integrity tests
├── results/                          # Published JSON, LaTeX, PNG, and PDF outputs
├── paper/                            # LaTeX paper and references (private)
└── data/                             # Ignored raw/processed/graph artifacts
```

## Setup

For the full local environment, run from the workspace root:

```bash
bash setup_env.sh
source streamring/.venv/bin/activate
cd streamring
```

For a StreamRing-only environment, run from `streamring/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, install the OpenMP runtime if XGBoost fails to load:

```bash
brew install libomp
```

## Quick start: correct run order

The commands below assume you are running from `streamring/`. From the workspace root:

```bash
cd streamring
source .venv/bin/activate
```

`run_all.sh` runs experiments and figure generation. It does not extract raw blockchain data or build graphs, so generated inputs must exist first.

### Full reproduction from scratch

Use this path when you want the full published period set from `configs/default.yaml`:

1. Authenticate for BigQuery:

   ```bash
   gcloud auth application-default login
   ```

2. Extract raw data and build graphs:

   ```bash
   bash scripts/extract_data.sh
   bash scripts/build_graphs.sh
   ```

3. Generate Tier 1 pattern features:

   ```bash
   python scripts/generate_patterns.py --period all --base-dir .
   ```

4. Generate labels from the available graphs and pattern features:

   ```bash
   python -m src.labeling.generate_labels --period all --base-dir .
   ```

5. Run experiments and regenerate figures/tables:

   ```bash
   bash run_all.sh
   ```

### `pipeline.py` quick path

Use this path for a quick end-to-end pipeline run using the periods configured inside `pipeline.py`:

```bash
python -m pipeline --phase all
python scripts/generate_patterns.py --period all --base-dir .
python -m src.labeling.generate_labels --period all --base-dir .
```

If you prefer to run from the workspace root, use the package-style command instead:

```bash
python -m streamring.pipeline --phase all
PYTHONPATH=streamring python scripts/generate_patterns.py --period all --base-dir streamring
PYTHONPATH=streamring python -m src.labeling.generate_labels --period all --base-dir streamring
```

If `run_all.sh` reports missing generated data, rebuild the missing inputs with the full data scripts before running experiments:

```bash
bash scripts/extract_data.sh
bash scripts/build_graphs.sh
python scripts/generate_patterns.py --period all --base-dir .
python -m src.labeling.generate_labels --period all --base-dir .
```

If you only want to regenerate figures and tables from existing JSON results, skip data extraction and run:

```bash
bash run_all.sh --figures
```

## Google BigQuery authentication

Data extraction is optional for reviewers who only need to inspect the paper, code, and published results. If you need to rebuild raw data or graphs, authenticate with one of these methods:

```bash
gcloud auth application-default login
```

or copy `.env.example` to `.env.local` and set your own service-account path:

```bash
cp .env.example .env.local
# Edit .env.local; never commit it.
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-billing-project-id"
```

Extract one period:

```bash
python -m src.etl.bigquery_extractor --config configs/default.yaml --period dao_hack
```

Build graphs from extracted CSV files:

```bash
bash scripts/build_graphs.sh
```

Generate Tier 1 pattern features before running Tier 3 or RDT experiments:

```bash
python scripts/generate_patterns.py --period all --base-dir .
python -m src.labeling.generate_labels --period all --base-dir .
```

## Replication workflow

1. Inspect the public artifact without credentials:
   ```bash
   cd streamring
   pytest -q
   bash run_all.sh --figures
   ```

2. Reproduce from scratch with generated data:
   ```bash
   cd streamring
   source .venv/bin/activate
   gcloud auth application-default login
   bash scripts/extract_data.sh
   bash scripts/build_graphs.sh
   python scripts/generate_patterns.py --period all --base-dir .
   python -m src.labeling.generate_labels --period all --base-dir .
   bash run_all.sh
   ```

   For a quicker pipeline-only run using the periods configured in `pipeline.py`:
   ```bash
   python -m pipeline --phase all
   python scripts/generate_patterns.py --period all --base-dir .
   python -m src.labeling.generate_labels --period all --base-dir .
   ```

3. Run only the faster core v3 suite plus figures:
   ```bash
   bash run_all.sh --core
   ```

4. Regenerate figures and LaTeX tables from existing JSON results:
   ```bash
   bash run_all.sh --figures
   ```

## Deterministic reproduction

Set these environment variables before starting Python:

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export STREAMRING_DETERMINISTIC=1
export STREAMRING_NUM_THREADS=1
```

## Expected public outputs

- Paper: `paper/main.tex`
- Tables: `results/tables/*.json`, `results/tables/*.tex`
- Figures: `results/figures/fig{1..16}_*.{png,pdf}`
- Tests: `tests/`

## Private or contributor-local files

Keep these out of public commits:

- `.venv/`, `__pycache__/`, `.pytest_cache/`
- `.env.local`, `credentials*.json`, `*.key`
- `configs/local.yaml`, `*.local.yaml`
- `data/raw/`, `data/processed/`, `data/graphs/`, `data/embeddings/`
- `models/*.pt`

## Citation

```bibtex
@article{streamring2026,
  title={StreamRing: A Cascading Architecture for Real-Time Detection of Blockchain Fraud Rings},
  author={Ahmad Zulfan, Ahmad Jibril Hemdi, Kyle Nathan Yahya},
  year={2026}
}
```

## License

MIT
