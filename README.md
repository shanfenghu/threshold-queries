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
- `reference/` contains the frozen numerical regression bundle.

## Setup

Use Python 3.11 or newer in a clean environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
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
To regenerate and verify all outputs after fetching the declared archives, run:

```bash
make reproduce
```

`make evaluate` (also available as `threshold-queries`) regenerates outputs
without comparing them with the frozen reference. Both commands stop with an
error if any independent verification reports failures. The workflow writes CSV
files to `results/`, diagnostic PDFs to `plots/`, and a concise
`results/summary.json`. It is deterministic and uses fixed random seeds for
generated stress tests. Non-timing outputs are deterministic; runtime
measurements naturally depend on the host machine.

## Expected outcomes

The full run produces 34 CSV outputs, one summary, and five diagnostic plots.
The climate cohort contains eight snapshots, seven transitions, 233 revision
episodes, 3,140 changed cells, and 15,700 window queries. Its temporal holdout
contains 14,955 queries; coupling tightens 5,620 of 14,104 nonzero envelopes.
The earthquake cohort contains 192 events and 195 lineages, with 66 reversals
among 4,992 threshold queries. All verification failure counts must be zero,
including the explicit unchanged-release control.

`make verify-results` checks existing outputs without rerunning the evaluation.
It compares every CSV cell and non-timing summary field with `reference/`,
checks output coverage, and checks that each expected plot has a PDF header.
Integer counts, text, and Boolean values must match exactly. Floating-point
values use relative tolerance `1e-7` and absolute tolerance `1e-9`; missing-value
positions must also match. Positive finite runtime measurements may vary, but
the benchmark problem sizes must match. Inspect plots visually after changing
visualisation code.

## Resources and runtime

Use a CPU-only Linux or macOS environment with 4 GiB RAM and 2 GiB free disk
space. No accelerator, proprietary software, account, or access token is
needed. Downloading the eight climate archives requires about 229 MiB of
network transfer. Their checksums and the bundled earthquake input are verified
before evaluation. Only the eight declared climate archives are processed;
additional archives in the input directory are ignored.

The audited configuration uses Python 3.12 on ARM64 macOS with an Apple M5 CPU
and 24 GiB RAM. Allow fifteen minutes for the complete evaluation, excluding
installation and downloads; this is a working budget rather than a benchmark
guarantee. Archive parsing can run quietly for several minutes before the final
summary appears. Once the data and dependencies are present, evaluation runs
offline. CI checks Python 3.11 and 3.13; its manually triggered workflow also
downloads the archives and runs the full reproduction.
