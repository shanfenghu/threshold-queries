# Threshold Queries

Threshold Queries computes certified decisions over revisable scientific data.
It preserves shared revision choices across a query, propagates exact endpoint
bounds through hierarchical constraints, emits replayable witnesses, and tests
whether a decision is stable for every admissible revision.

The complete evaluation uses historical US Climate Reference Network releases
and US Geological Survey earthquake histories. It includes temporal holdouts,
leave-one-source checks, exhaustive finite-mode programs, continuous threshold
reversals, local-relaxation counterexamples, and scaling workloads.

The central guarantee is query-wide consistency: a revision mode shared by
several records is chosen once, not independently for each occurrence. The
resulting interval contains exactly the attainable query values. Decisions are
returned only when that whole interval lies on one side of the threshold.

## Organisation

- `src/threshold_queries/` contains the reusable algorithms and evaluation workflow.
- `experiments/` contains the command-line entry point.
- `tests/` contains small deterministic checks.
- `data/input/` contains the compact ComCat history used by the workflow.
- `data/fetch_uscrn.py` retrieves the fixed climate release archives.
- `data/README.md` records provenance and input validation.
- `results/` contains machine-readable evaluation outputs.
- `plots/` contains diagnostic visualisations.
- `docs/algorithm.md` explains the dynamic program and certificates.

## Setup

Use Python 3.11 or newer in a clean environment:

```bash
python -m pip install -e '.[dev]'
```

Run commands from the repository root. To invoke an installed command from
elsewhere, set `THRESHOLD_QUERIES_ROOT` to the checkout path.

Fetch and verify the larger climate archives:

```bash
python data/fetch_uscrn.py
make check-data
```

## Verification

```bash
make test
make audit
```

`make audit` runs the deterministic checks and validates every Python module.
To regenerate all outputs after placing the inputs described in
`data/README.md`, run:

```bash
make evaluate
```

The same command is available as `threshold-queries`. The workflow writes CSV
files to `results/`, diagnostic PDFs to `plots/`, and a concise
`results/summary.json`. It is deterministic and uses fixed random seeds for
generated stress tests. Non-timing outputs are deterministic; runtime
measurements naturally depend on the host machine.
