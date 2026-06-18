# Repository Workspace

This workspace contains the StreamRing research artifact and its local Ethereum ETL dependency.

## What is here

- `streamring/` — main research project for Ethereum Classic fraud ring detection.
  - Open `streamring/README.md` for project overview, setup, replication steps, and public artifact layout.
  - Open `streamring/REPRODUCIBILITY.md` for reviewer-facing reproduction instructions.
- `ethereum-etl-develop/` — local editable dependency used by StreamRing's optional data extraction workflow.
- `setup_env.sh` — optional bootstrap script that creates `streamring/.venv`, installs `ethereum-etl-develop`, and installs StreamRing dependencies.

## Quick navigation

```bash
cd streamring
```

From `streamring/`, use:

```bash
pytest -q
bash run_all.sh --figures
```

For the full local environment from this root folder:

```bash
bash setup_env.sh
source streamring/.venv/bin/activate
cd streamring
```
