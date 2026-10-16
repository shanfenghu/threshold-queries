# Certified threshold queries

## Revision model

A data release may later be replaced by one of several admissible modes. One
mode can affect many records, so a valid query must reuse that shared choice
everywhere it appears. Local modes are connected by compatibility constraints;
for hierarchical releases, those constraints form a rooted tree.

Each mode supplies a lower and upper contribution. A query asks for the smallest
and largest total over all globally compatible mode assignments.

## Dynamic program

For each node and parent mode, the workflow stores the smallest and largest
subtree contribution. A candidate local mode is legal only when it accepts the
parent mode. Child endpoints then add independently conditional on that shared
choice. Taking the minimum and maximum over legal local modes yields sharp
subtree endpoints.

The implementation also records the selected mode at every node, producing a
complete witness assignment for each endpoint. A separate replay routine checks
compatibility and recomputes the claimed value without invoking the optimiser.

## Threshold decisions

For threshold `t` and endpoint interval `[L, U]`:

- `L >= t` certifies that every admissible revision is in;
- `U < t` certifies that every admissible revision is out;
- `L < t <= U` means the decision can change;
- an incomplete revision contract produces no certificate.

## Verification

The evaluation compares the dynamic program with complete enumeration on small
trees and a generic linear formulation on larger random trees. It also checks
composed refinements, shared affine parameters, finite-mode projections, stale
witness rejection, deliberate certificate mutations, parity gaps on cycles,
temporal holdouts, leave-one-source perturbations, and continuous earthquake
threshold reversals.

Each check writes a machine-readable CSV file. `results/oracle_audit.csv`
collects the check counts and failure counts in one place.
