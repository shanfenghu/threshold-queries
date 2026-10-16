# Threshold Queries

Threshold Queries computes certified decisions over revisable scientific data.
It preserves shared revision choices across a query, propagates exact endpoint
bounds through hierarchical constraints, emits replayable witnesses, and tests
whether a decision is stable for every admissible revision.

The complete evaluation uses historical US Climate Reference Network releases
and US Geological Survey earthquake histories. It includes temporal holdouts,
leave-one-source checks, exhaustive finite-mode programs, continuous threshold
reversals, local-relaxation counterexamples, and scaling workloads.

## Organisation

- `src/threshold_queries/` contains the reusable algorithms and evaluation workflow.
- `experiments/` contains the command-line entry point.
- `tests/` contains small deterministic checks.
- `data/` documents the required public inputs; downloaded inputs remain local.
- `results/` contains machine-readable evaluation outputs.
- `plots/` contains diagnostic visualisations.

## Setup

Use Python 3.11 or newer in a clean environment:

```bash
python -m pip install -e '.[dev]'
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

The workflow is deterministic and uses fixed random seeds for synthetic checks.
