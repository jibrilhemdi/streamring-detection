# StreamRing Reproducibility Guide

This guide is written for public reviewers, external contributors, and non-team members. It separates what should be public from what must remain private and gives a step-by-step replication path.

## Workspace and public artifact layout

The workspace root contains the StreamRing artifact plus its optional local data dependency:

```text
finds2026/
├── streamring/                       # Public research artifact
├── ethereum-etl-develop/             # Local editable dependency for extraction
└── setup_env.sh                      # Optional full-environment bootstrap
```

Public reviewers should be able to inspect the paper, code, configuration, tests, and published outputs without access to private credentials or raw blockchain data.

Keep these files public:

- `streamring/paper/`
- `streamring/results/tables/`
- `streamring/results/figures/`
- `streamring/src/`
- `streamring/experiments/`
- `streamring/tests/`
- `streamring/configs/default.yaml`
- `streamring/README.md`
- `streamring/REPRODUCIBILITY.md`

Do not publish these by default:

- `streamring/data/raw/`
- `streamring/data/processed/`
- `streamring/data/graphs/`
- `streamring/data/embeddings/`
- `streamring/models/*.pt`
- `streamring/.venv/`
- `streamring/.env.local`
- `streamring/credentials*.json`
- `streamring/*.key`
- `streamring/configs/local.yaml`
- `streamring/*.local.yaml`

## Environment setup

Recommended full setup from the workspace root:

```bash
bash setup_env.sh
source streamring/.venv/bin/activate
cd streamring
```

StreamRing-only setup from `streamring/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost may also require the OpenMP runtime:

```bash
brew install libomp
```

For tests:

```bash
pytest -q
```

If pytest crashes inside a native dependency such as XGBoost, rebuild the environment or install the platform runtime required by that dependency.

## Google BigQuery authentication

Data extraction is optional for reviewers who only need to inspect code or regenerate figures from existing results.

Use one of these methods, and never commit credentials:

```bash
gcloud auth application-default login
```

or:

```bash
cp .env.example .env.local
# Edit .env.local with your own service-account path.
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-billing-project-id"
```

Then extract one period:

```bash
python -m src.etl.bigquery_extractor --config configs/default.yaml --period dao_hack
```

To extract all configured periods:

```bash
bash scripts/extract_data.sh
```

Build temporal graphs from extracted CSV files:

```bash
bash scripts/build_graphs.sh
```

Generate labels from graphs before running experiments:

```bash
python -m src.labeling.generate_labels --period all --base-dir .
```

## Deterministic reproduction

Set these variables before starting Python:

```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export STREAMRING_DETERMINISTIC=1
export STREAMRING_NUM_THREADS=1
```

Run the full published artifact pipeline:

```bash
bash run_all.sh
```

Run only the core v3 experiments plus figures:

```bash
bash run_all.sh --core
```

Regenerate figures and LaTeX tables from existing JSON results:

```bash
bash run_all.sh --figures
```

`PYTHONHASHSEED` must be set before Python starts; setting it inside a running process cannot change hash randomization for that process.

## Expected public outputs

- Paper: `paper/main.tex`
- Tables: `results/tables/*.json`, `results/tables/*.tex`
- Figures: `results/figures/fig{1..16}_*.{png,pdf}`
- Tests: `tests/`
