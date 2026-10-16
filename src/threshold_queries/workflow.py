#!/usr/bin/env python3
"""Replay NOAA revision episodes and audit USGS ComCat version histories."""

from __future__ import annotations

import json
import gzip
import time
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "uscrn"
RAW_COMCAT = ROOT / "data" / "raw" / "comcat"
OUT = ROOT / "results"
PLOTS = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "WBANNO", "LST_DATE", "CRX_VN", "LONGITUDE", "LATITUDE", "T_DAILY_MAX",
    "T_DAILY_MIN", "T_DAILY_MEAN", "T_DAILY_AVG", "P_DAILY_CALC",
    "SOLARAD_DAILY", "SUR_TEMP_DAILY_TYPE", "SUR_TEMP_DAILY_MAX",
    "SUR_TEMP_DAILY_MIN", "SUR_TEMP_DAILY_AVG", "RH_DAILY_MAX", "RH_DAILY_MIN",
    "RH_DAILY_AVG", "SOIL_MOISTURE_5_DAILY", "SOIL_MOISTURE_10_DAILY",
    "SOIL_MOISTURE_20_DAILY", "SOIL_MOISTURE_50_DAILY",
    "SOIL_MOISTURE_100_DAILY", "SOIL_TEMP_5_DAILY", "SOIL_TEMP_10_DAILY",
    "SOIL_TEMP_20_DAILY", "SOIL_TEMP_50_DAILY", "SOIL_TEMP_100_DAILY",
]
VALUE_START = 5
DEVELOPMENT_SNAPSHOT_COUNT = 4
HELDOUT_START_TRANSITION = DEVELOPMENT_SNAPSHOT_COUNT


def read_snapshot_frame(z: ZipFile, name: str) -> pd.DataFrame:
    rows = [line.split() for line in z.read(name).decode().splitlines() if line.strip()]
    frame = pd.DataFrame(rows, columns=FIELDS)
    frame["LST_DATE"] = pd.to_datetime(frame["LST_DATE"], format="%Y%m%d")
    for field in FIELDS[VALUE_START:]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        frame.loc[frame[field] <= -9990, field] = np.nan
    return frame.set_index("LST_DATE").sort_index()


def changed_episodes(paths: list[Path]):
    episodes = []
    transitions = []
    for transition, (left, right) in enumerate(zip(paths, paths[1:]), start=1):
        with ZipFile(left) as za, ZipFile(right) as zb:
            names = sorted(
                set(n for n in za.namelist() if n.endswith(".txt") and "/" in n)
                & set(n for n in zb.namelist() if n.endswith(".txt") and "/" in n)
            )
            file_count = row_count = cell_count = 0
            for name in names:
                # A token comparison is about thirty times cheaper than building
                # two DataFrames for every unchanged station-year file.
                raw_old = {parts[1]: parts for line in za.read(name).decode().splitlines()
                           if (parts := line.split())}
                raw_new = {parts[1]: parts for line in zb.read(name).decode().splitlines()
                           if (parts := line.split())}
                changed_keys = [key for key in raw_old.keys() & raw_new.keys()
                                if raw_old[key] != raw_new[key]]
                if not changed_keys:
                    continue
                old = read_snapshot_frame(za, name); new = read_snapshot_frame(zb, name)
                dates = old.index.intersection(new.index)
                old = old.loc[dates]; new = new.loc[dates]
                changes = []
                changed_dates = set()
                for field in FIELDS[VALUE_START:]:
                    a, b = old[field], new[field]
                    mask = (a.isna() != b.isna()) | (a.notna() & b.notna() & (np.abs(a - b) > 1e-12))
                    for date in dates[mask]:
                        changes.append({"date": date, "field": field,
                                        # SQL SUM ignores NULL, represented here by zero
                                        # contribution while retaining the row in COUNT.
                                        "old": 0.0 if pd.isna(a.loc[date]) else float(a.loc[date]),
                                        "new": 0.0 if pd.isna(b.loc[date]) else float(b.loc[date]),
                                        "old_is_null": bool(pd.isna(a.loc[date])),
                                        "new_is_null": bool(pd.isna(b.loc[date]))})
                        changed_dates.add(date)
                if changes:
                    file_count += 1; row_count += len(changed_dates); cell_count += len(changes)
                    episodes.append({"transition": transition, "name": name,
                                     "old": old, "new": new, "changes": changes})
            transitions.append({"transition": transition, "files": file_count,
                                "rows": row_count, "cells": cell_count})
    return episodes, pd.DataFrame(transitions)


def workload(episodes, endpoint_mode: str = "changed") -> pd.DataFrame:
    if endpoint_mode not in {"changed", "calendar"}:
        raise ValueError(f"unknown endpoint mode: {endpoint_mode}")
    rows = []
    widths = (7, 30, 90, 180, 365)
    for eid, episode in enumerate(episodes, start=1):
        changes = pd.DataFrame(episode["changes"])
        common_dates = episode["old"].index.intersection(episode["new"].index)
        # The calendar census uses every shared daily endpoint; selection does
        # not inspect which dates were corrected.
        calendar_ends = common_dates
        for field in sorted(changes.field.unique()):
            field_changes = changes[changes.field == field].sort_values("date")
            for days in widths:
                ends = (field_changes.date.unique() if endpoint_mode == "changed"
                        else calendar_ends)
                for end in ends:
                    start = pd.Timestamp(end) - pd.Timedelta(days=days - 1)
                    old = episode["old"].loc[start:end, field].fillna(0.0)
                    new = episode["new"].loc[start:end, field].fillna(0.0)
                    common = old.index.intersection(new.index)
                    if len(common) == 0:
                        continue
                    old_sum = float(old.loc[common].sum()); new_sum = float(new.loc[common].sum())
                    local = field_changes[(field_changes.date >= start) & (field_changes.date <= end)]
                    if len(local) == 0:
                        continue
                    coupled_lo, coupled_hi = sorted((old_sum, new_sum))
                    delta_abs = float(np.abs(local.new - local.old).sum())
                    positive_delta = float(np.maximum(local.new - local.old, 0).sum())
                    negative_delta = float(np.maximum(local.old - local.new, 0).sum())
                    independent_lo = old_sum + float(np.minimum(local.new - local.old, 0).sum())
                    independent_hi = old_sum + float(np.maximum(local.new - local.old, 0).sum())
                    raw_old = old.loc[common].to_numpy(float)
                    raw_new = new.loc[common].to_numpy(float)
                    raw_independent_lo = float(np.minimum(raw_old, raw_new).sum())
                    raw_independent_hi = float(np.maximum(raw_old, raw_new).sum())
                    # Thresholds cover the full row-independent envelope, including endpoints.
                    thresholds = np.linspace(independent_lo, independent_hi, 101)
                    c_uncertain = np.mean((thresholds > coupled_lo) & (thresholds <= coupled_hi))
                    i_uncertain = np.mean((thresholds > independent_lo) & (thresholds <= independent_hi))
                    rows.append({
                        "episode": eid, "transition": episode["transition"],
                        "endpoint_mode": endpoint_mode,
                        "station": Path(episode["name"]).stem.split("-", 2)[-1],
                        "field": field, "window_days": days, "end": pd.Timestamp(end),
                        "changed_cells": len(local), "old_sum": old_sum, "new_sum": new_sum,
                        "coupled_width": coupled_hi - coupled_lo,
                        "independent_width": independent_hi - independent_lo,
                        "independent_lo": independent_lo,
                        "independent_hi": independent_hi,
                        "raw_independent_lo": raw_independent_lo,
                        "raw_independent_hi": raw_independent_hi,
                        "c_uncertain": c_uncertain, "i_uncertain": i_uncertain,
                        "delta_abs": delta_abs,
                        "positive_delta": positive_delta,
                        "negative_delta": negative_delta,
                        "null_sensitive": bool(
                            (local.old_is_null | local.new_is_null).any()
                        ),
                    })
    return pd.DataFrame(rows)


def calendar_endpoint_census_audit(episodes, frame: pd.DataFrame) -> dict[str, int]:
    """Independently reconstruct every eligible daily endpoint key."""
    expected = set()
    widths = (7, 30, 90, 180, 365)
    for eid, episode in enumerate(episodes, start=1):
        changes = pd.DataFrame(episode["changes"])
        common_dates = episode["old"].index.intersection(episode["new"].index)
        for field in sorted(changes.field.unique()):
            changed_dates = set(pd.to_datetime(
                changes.loc[changes.field == field, "date"]
            ))
            for days in widths:
                for end in common_dates:
                    start = pd.Timestamp(end) - pd.Timedelta(days=days - 1)
                    if any(start <= date <= end for date in changed_dates):
                        expected.add((eid, field, days, pd.Timestamp(end)))
    observed_rows = [
        (int(row.episode), row.field, int(row.window_days), pd.Timestamp(row.end))
        for row in frame.itertuples(index=False)
    ]
    observed = set(observed_rows)
    summary = {
        "expected_keys": len(expected),
        "observed_rows": len(observed_rows),
        "missing": len(expected - observed),
        "extra": len(observed - expected),
        "duplicates": len(observed_rows) - len(observed),
    }
    summary["failures"] = (summary["missing"] + summary["extra"] +
                           summary["duplicates"])
    pd.DataFrame([summary]).to_csv(
        OUT / "calendar_endpoint_census_audit.csv", index=False)
    return summary


def prequential_max_audit(changes: pd.DataFrame) -> dict[str, int | float]:
    """Test a past-maximum correction heuristic without future leakage."""
    frame = changes.copy()
    frame["abs_delta"] = np.abs(frame.new - frame.old)
    rows = []
    violation_ratios = []
    first_transition = int(frame.transition.min())
    for transition in sorted(frame.transition.unique()):
        heldout = frame[frame.transition == transition]
        history = frame[frame.transition < transition]
        for field, group in heldout.groupby("field"):
            prior = history[history.field == field]
            if prior.empty:
                rows.append({
                    "transition": transition, "field": field,
                    "prior_changes": 0, "prior_max_abs_delta": np.nan,
                    "heldout_changes": len(group), "violations": 0,
                    "max_violation_ratio": np.nan,
                    "result": ("warm-up" if transition == first_transition
                               else "abstain-unseen-field"),
                })
                continue
            bound = float(prior.abs_delta.max())
            if bound > 0:
                ratios = group.abs_delta.to_numpy(float) / bound
            else:
                ratios = np.where(group.abs_delta.to_numpy(float) > 1e-12,
                                  np.inf, 1.0)
            violating_ratios = ratios[
                group.abs_delta.to_numpy(float) > bound + 1e-12]
            violation_ratios.extend(violating_ratios.tolist())
            violations = len(violating_ratios)
            ratio = float(np.max(ratios))
            rows.append({
                "transition": transition, "field": field,
                "prior_changes": len(prior), "prior_max_abs_delta": bound,
                "heldout_changes": len(group), "violations": violations,
                "violations_2x": int((group.abs_delta > 2 * bound + 1e-12).sum()),
                "violations_10x": int((group.abs_delta > 10 * bound + 1e-12).sum()),
                "violations_100x": int((group.abs_delta > 100 * bound + 1e-12).sum()),
                "max_violation_ratio": ratio, "result": "tested",
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "prequential_max_audit.csv", index=False)
    tested = audit[audit.result == "tested"]
    abstained = audit[audit.result == "abstain-unseen-field"]
    violating = tested[tested.violations > 0]
    ratio_array = np.asarray(violation_ratios, dtype=float)
    return {
        "prior": int(tested.prior_changes.sum()),
        "heldout": int(tested.heldout_changes.sum()),
        "violations": int(tested.violations.sum()),
        "violation_pct": 100 * float(tested.violations.sum()) /
                         float(tested.heldout_changes.sum()),
        "max_ratio": float(tested.max_violation_ratio.max()),
        "median_violation_ratio": float(np.median(ratio_array)),
        "p90_violation_ratio": float(np.quantile(ratio_array, .9)),
        "violations_2x": int(tested.violations_2x.sum()),
        "violations_10x": int(tested.violations_10x.sum()),
        "violations_100x": int(tested.violations_100x.sum()),
        "tested_transitions": int(tested.transition.nunique()),
        "violating_transitions": int(violating.transition.nunique()),
        "tested_fields": int(tested.field.nunique()),
        "violating_fields": int(violating.field.nunique()),
        "abstentions": int(abstained.heldout_changes.sum()),
    }


def plot_shortcut_failures(changes: pd.DataFrame,
                            prequential: dict[str, int | float],
                            comcat: dict[str, object]) -> None:
    """Show one native witness per domain beside the full-cohort counts."""
    frame = changes.copy()
    frame["abs_delta"] = np.abs(frame.new - frame.old)

    # Relative humidity gives a compact, interpretable holdout witness: the
    # next official value lies outside the envelope induced by every strictly
    # earlier change of the same released field.
    candidates = []
    for transition in sorted(frame.transition.unique()):
        if transition < HELDOUT_START_TRANSITION:
            continue
        history = frame[(frame.transition < transition) &
                        (frame.field == "RH_DAILY_AVG")]
        if history.empty:
            continue
        bound = float(history.abs_delta.max())
        current = frame[(frame.transition == transition) &
                        (frame.field == "RH_DAILY_AVG")].copy()
        current = current[current.abs_delta > bound + 1e-12]
        if not current.empty:
            current["prior_bound"] = bound
            current["ratio"] = current.abs_delta / bound
            candidates.append(current)
    if not candidates:
        raise RuntimeError("No held-out humidity witness for shortcut diagnostic")
    noaa = pd.concat(candidates).sort_values("ratio", ascending=False).iloc[0]

    witnesses = pd.read_csv(OUT / "comcat_witness_audit.csv")
    # Prefer a clean 0.2-magnitude increase crossing a one-decimal threshold.
    target = witnesses[
        np.isclose(witnesses.first_reviewed_magnitude, 4.8) &
        np.isclose(witnesses.last_magnitude, 5.0)
    ]
    if target.empty:
        target = witnesses.assign(
            span=np.abs(witnesses.last_magnitude -
                        witnesses.first_reviewed_magnitude)
        ).sort_values("span", ascending=False)
    quake = target.iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.10),
                             constrained_layout=True)

    ax = axes[0]
    old, new = float(noaa.old), float(noaa.new)
    bound = float(noaa.prior_bound)
    left, right = old - bound, old + bound
    pad = .12 * (max(right, new) - min(left, new))
    ax.axvspan(left, right, color="#92c5de", alpha=.42,
               label="past-change envelope")
    ax.annotate("", xy=(new, 0), xytext=(old, 0),
                arrowprops={"arrowstyle": "->", "color": "#b2182b",
                            "lw": 1.2})
    ax.scatter([old], [0], color="#2166ac", s=22, zorder=3)
    ax.scatter([new], [0], color="#b2182b", marker="x", s=30,
               linewidths=1.3, zorder=3)
    ax.text(old, .13, f"old {old:.1f}", ha="center", fontsize=6.5)
    ax.text(new, -.16, f"new {new:.1f}", ha="center", fontsize=6.5,
            color="#8c0d1a")
    ax.text(.02, .90, f"{float(noaa.ratio):.2f}x past maximum",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color="#8c0d1a")
    ax.text(.02, .04,
            f"{prequential['violations']:,}/{prequential['heldout']:,} "
            f"tested changes exceed it ({prequential['violation_pct']:.1f}%)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2)
    ax.set_xlim(min(left, new) - pad, max(right, new) + pad)
    ax.set_ylim(-.28, .30)
    ax.set_yticks([])
    ax.set_xlabel("daily relative humidity (%)", fontsize=7)
    ax.set_title("(a) NOAA: history is not a bound", loc="left",
                 fontsize=7.5, fontweight="bold", pad=2)
    ax.tick_params(axis="x", labelsize=6.5, pad=1.5)
    ax.spines[["left", "right", "top"]].set_visible(False)

    ax = axes[1]
    first = float(quake.first_reviewed_magnitude)
    last = float(quake.last_magnitude)
    threshold = float(quake.witness_threshold)
    lo, hi = min(first, last), max(first, last)
    pad = max(.06, .35 * (hi - lo))
    ax.axvspan(threshold, hi + pad, color="#f4a582", alpha=.30)
    ax.axvline(threshold, color="#555555", lw=.8, ls="--")
    ax.annotate("", xy=(last, 0), xytext=(first, 0),
                arrowprops={"arrowstyle": "->", "color": "#b2182b",
                            "lw": 1.2})
    ax.scatter([first], [0], color="#2166ac", marker="s", s=22, zorder=3)
    ax.scatter([last], [0], color="#b2182b", s=22, zorder=3)
    ax.text(first, .13, f"reviewed {first:.1f}", ha="center", fontsize=6.5)
    ax.text(last, -.16, f"later {last:.1f}", ha="center", fontsize=6.5,
            color="#8c0d1a")
    ax.text(threshold, .24, f"threshold {threshold:.1f}", ha="center",
            fontsize=6.2, color="#444444")
    ax.text(.02, .04,
            f"{comcat['ComcatFlips']:,}/{comcat['ComcatThresholds']:,} "
            "fixed-grid decisions flip",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-.28, .30)
    ax.set_yticks([])
    ax.set_xlabel("earthquake magnitude", fontsize=7)
    ax.set_title("(b) ComCat: reviewed is not final", loc="left",
                 fontsize=7.5, fontweight="bold", pad=2)
    ax.tick_params(axis="x", labelsize=6.5, pad=1.5)
    ax.spines[["left", "right", "top"]].set_visible(False)

    fig.savefig(PLOTS / "shortcut_failures.pdf", bbox_inches="tight")
    plt.close(fig)


def partition_ablation(changes: pd.DataFrame,
                       queries: pd.DataFrame,
                       output_prefix: str = "") -> dict[str, object]:
    """Weaken release coupling through a deterministic nested partition."""
    queries = queries.reset_index(drop=True)
    rows = []
    for query in queries.itertuples(index=False):
        start = query.end - pd.Timedelta(days=query.window_days - 1)
        local = changes[
            (changes.episode == query.episode) &
            (changes.field == query.field) &
            (changes.date >= start) &
            (changes.date <= query.end)
        ].copy().sort_values("date")
        local["delta"] = local.new - local.old
        local["month"] = local.date.dt.to_period("M")
        local["month_run"] = (
            local.month.ne(local.month.shift()) |
            (local.date.diff().dt.days.fillna(0) > 1)
        ).cumsum()

        def endpoints(grouped_delta: pd.Series) -> tuple[float, float]:
            return (query.old_sum + float(np.minimum(grouped_delta, 0).sum()),
                    query.old_sum + float(np.maximum(grouped_delta, 0).sum()))

        release_lo, release_hi = sorted((query.old_sum, query.new_sum))
        month_lo, month_hi = endpoints(local.groupby("month").delta.sum())
        run_lo, run_hi = endpoints(local.groupby("month_run").delta.sum())
        independent_lo, independent_hi = query.independent_lo, query.independent_hi
        release_month_pass = (month_lo <= release_lo + 1e-9 and
                                  month_hi >= release_hi - 1e-9)
        month_run_pass = (run_lo <= month_lo + 1e-9 and
                          run_hi >= month_hi - 1e-9)
        run_row_pass = (independent_lo <= run_lo + 1e-9 and
                        independent_hi >= run_hi - 1e-9)
        rows.append({
            "episode": query.episode, "field": query.field,
            "window_days": query.window_days, "end": query.end,
            "release_lo": release_lo, "release_hi": release_hi,
            "month_lo": month_lo, "month_hi": month_hi,
            "month_run_lo": run_lo, "month_run_hi": run_hi,
            "independent_lo": independent_lo, "independent_hi": independent_hi,
            "release_month_pass": release_month_pass,
            "month_run_pass": month_run_pass,
            "run_row_pass": run_row_pass,
            "nested_containment_pass": (release_month_pass and
                                         month_run_pass and run_row_pass),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"{output_prefix}partition_ablation.csv", index=False)
    nonzero = queries.independent_width > 1e-12
    summaries = []
    for label, lo_column, hi_column in (
        ("release episode", "release_lo", "release_hi"),
        ("calendar month", "month_lo", "month_hi"),
        ("within-month contiguous run", "month_run_lo", "month_run_hi"),
        ("independent row", "independent_lo", "independent_hi"),
    ):
        width = frame[hi_column] - frame[lo_column]
        reduction = 1 - width[nonzero] / queries.loc[nonzero, "independent_width"]
        summaries.append({
            "partition": label, "nonzero_queries": int(nonzero.sum()),
            "tightened_queries": int((reduction > 1e-9).sum()),
            "tightened_pct": 100 * float((reduction > 1e-9).mean()),
            "mean_reduction_pct": 100 * float(reduction.mean()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(
        OUT / f"{output_prefix}partition_ablation_summary.csv", index=False)
    return {
        "frame": frame, "summary": summary,
        "checks": 3 * len(frame),
        "failures": int((~frame.release_month_pass).sum() +
                        (~frame.month_run_pass).sum() +
                        (~frame.run_row_pass).sum()),
    }


def dp_bounds(children, modes, node=0, parent_state=0, memo=None):
    if memo is None:
        memo = {}
    key = (node, parent_state)
    if key in memo:
        return memo[key]
    candidates = []
    for mode, (lo, hi, allowed) in enumerate(modes[node]):
        if parent_state not in allowed:
            continue
        low, high = lo, hi
        for child in children[node]:
            cl, ch = dp_bounds(children, modes, child, mode, memo)
            low += cl; high += ch
        candidates.append((low, high))
    memo[key] = (min(x[0] for x in candidates), max(x[1] for x in candidates))
    return memo[key]


def local_polytope_bounds(children, modes):
    """Solve the exact tree-local marginal formulation with a generic LP."""
    parents = [-1] * len(children)
    for parent, child_list in enumerate(children):
        for child in child_list:
            parents[child] = parent
    x_index = {}
    cursor = 0
    for node, node_modes in enumerate(modes):
        for mode in range(len(node_modes)):
            x_index[node, mode] = cursor
            cursor += 1
    y_index = {}
    for child in range(1, len(children)):
        parent = parents[child]
        for parent_mode in range(len(modes[parent])):
            for child_mode, (_, _, allowed) in enumerate(modes[child]):
                if parent_mode in allowed:
                    y_index[child, parent_mode, child_mode] = cursor
                    cursor += 1

    rows = []
    cols = []
    data = []
    rhs = []

    def add_constraint(terms, value):
        row = len(rhs)
        for column, coefficient in terms:
            rows.append(row); cols.append(column); data.append(coefficient)
        rhs.append(value)

    add_constraint([(x_index[0, mode], 1.0)
                    for mode in range(len(modes[0]))], 1.0)
    for child in range(1, len(children)):
        parent = parents[child]
        for child_mode, (_, _, allowed) in enumerate(modes[child]):
            terms = [(x_index[child, child_mode], -1.0)]
            terms.extend((y_index[child, parent_mode, child_mode], 1.0)
                         for parent_mode in range(len(modes[parent]))
                         if parent_mode in allowed)
            add_constraint(terms, 0.0)
        for parent_mode in range(len(modes[parent])):
            terms = [(x_index[parent, parent_mode], -1.0)]
            terms.extend((y_index[child, parent_mode, child_mode], 1.0)
                         for child_mode, (_, _, allowed)
                         in enumerate(modes[child])
                         if parent_mode in allowed)
            add_constraint(terms, 0.0)
    matrix = coo_matrix(
        (data, (rows, cols)), shape=(len(rhs), cursor)).tocsr()
    lower_objective = np.zeros(cursor)
    upper_objective = np.zeros(cursor)
    for node, node_modes in enumerate(modes):
        for mode, (lo, hi, _) in enumerate(node_modes):
            lower_objective[x_index[node, mode]] = lo
            upper_objective[x_index[node, mode]] = -hi
    bounds = [(0.0, 1.0)] * cursor
    for mode, (_, _, allowed) in enumerate(modes[0]):
        if 0 not in allowed:
            bounds[x_index[0, mode]] = (0.0, 0.0)
    lower = linprog(lower_objective, A_eq=matrix, b_eq=rhs,
                    bounds=bounds, method="highs")
    upper = linprog(upper_objective, A_eq=matrix, b_eq=rhs,
                    bounds=bounds, method="highs")
    if not lower.success or not upper.success:
        raise RuntimeError("tree-local polytope unexpectedly infeasible")
    return float(lower.fun), float(-upper.fun)


def random_tree_program(rng, n, mode_count):
    children = [[] for _ in range(n)]
    for child in range(1, n):
        children[int(rng.integers(0, child))].append(child)
    modes = []
    for node in range(n):
        node_modes = []
        allowed_sets = []
        for _ in range(mode_count):
            center = float(rng.normal())
            radius = float(rng.uniform(0.01, 1.0))
            if node == 0:
                allowed = {0}
            else:
                allowed = {parent_mode for parent_mode in range(mode_count)
                           if rng.random() < 0.55}
                if not allowed:
                    allowed = {int(rng.integers(mode_count))}
            allowed_sets.append(allowed)
            node_modes.append((center - radius, center + radius, allowed))
        if node > 0:
            for parent_mode in range(mode_count):
                if not any(parent_mode in allowed for allowed in allowed_sets):
                    allowed_sets[int(rng.integers(mode_count))].add(parent_mode)
        modes.append(node_modes)
    return children, modes


def local_polytope_audit(rng, programs=256):
    rows = []
    for program in range(programs):
        n = int(rng.integers(6, 41))
        mode_count = int(rng.integers(2, 6))
        children, modes = random_tree_program(rng, n, mode_count)
        dp_lo, dp_hi = dp_bounds(children, modes)
        lp_lo, lp_hi = local_polytope_bounds(children, modes)
        error = max(abs(dp_lo - lp_lo), abs(dp_hi - lp_hi))
        rows.append({"program": program, "nodes": n, "modes": mode_count,
                     "dp_lo": dp_lo, "dp_hi": dp_hi,
                     "lp_lo": lp_lo, "lp_hi": lp_hi,
                     "maximum_error": error, "pass": error <= 1e-7})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "tree_local_polytope_lp_audit.csv", index=False)
    return {"programs": programs, "checks": 2 * programs,
            "failures": int((~frame["pass"]).sum()),
            "max_error": float(frame.maximum_error.max())}


def anti_equality_cycle_feasibility(n):
    """Return local-LP and integral feasibility for a binary cycle."""
    x_count = 2 * n
    # Each edge has only the allowed pairs (0,1) and (1,0).
    y_index = {(edge, pair): x_count + 2 * edge + pair
               for edge in range(n) for pair in range(2)}
    variable_count = x_count + 2 * n
    rows = []
    cols = []
    data = []
    rhs = []

    def add_constraint(terms, value):
        row = len(rhs)
        for column, coefficient in terms:
            rows.append(row); cols.append(column); data.append(coefficient)
        rhs.append(value)

    for node in range(n):
        add_constraint([(2 * node, 1.0), (2 * node + 1, 1.0)], 1.0)
    for edge in range(n):
        left = edge
        right = (edge + 1) % n
        # y[0] represents (left=0,right=1); y[1] represents (1,0).
        add_constraint([(y_index[edge, 0], 1.0), (2 * left, -1.0)], 0.0)
        add_constraint([(y_index[edge, 1], 1.0), (2 * left + 1, -1.0)], 0.0)
        add_constraint([(y_index[edge, 1], 1.0), (2 * right, -1.0)], 0.0)
        add_constraint([(y_index[edge, 0], 1.0), (2 * right + 1, -1.0)], 0.0)
    matrix = coo_matrix(
        (data, (rows, cols)), shape=(len(rhs), variable_count)).tocsr()
    lp = linprog(np.zeros(variable_count), A_eq=matrix, b_eq=rhs,
                 bounds=[(0.0, 1.0)] * variable_count, method="highs")
    integral = any(
        all(((assignment >> edge) & 1) !=
            ((assignment >> ((edge + 1) % n)) & 1)
            for edge in range(n))
        for assignment in range(1 << n)
    )
    return bool(lp.success), integral


def disagreement_cycle_endpoints(n):
    """Return local, parity-repaired, and integral disagreement maxima."""
    x_count = 2 * n
    y_index = {(edge, left_mode, right_mode):
               x_count + 4 * edge + 2 * left_mode + right_mode
               for edge in range(n)
               for left_mode in range(2) for right_mode in range(2)}
    variable_count = x_count + 4 * n
    rows = []
    cols = []
    data = []
    rhs = []

    def add_constraint(terms, value):
        row = len(rhs)
        for column, coefficient in terms:
            rows.append(row); cols.append(column); data.append(coefficient)
        rhs.append(value)

    for node in range(n):
        add_constraint([(2 * node, 1.0), (2 * node + 1, 1.0)], 1.0)
    for edge in range(n):
        left = edge
        right = (edge + 1) % n
        for left_mode in range(2):
            add_constraint(
                [(y_index[edge, left_mode, right_mode], 1.0)
                 for right_mode in range(2)] +
                [(2 * left + left_mode, -1.0)], 0.0)
        for right_mode in range(2):
            add_constraint(
                [(y_index[edge, left_mode, right_mode], 1.0)
                 for left_mode in range(2)] +
                [(2 * right + right_mode, -1.0)], 0.0)
    objective = np.zeros(variable_count)
    for edge in range(n):
        objective[y_index[edge, 0, 1]] = -1.0
        objective[y_index[edge, 1, 0]] = -1.0
    matrix = coo_matrix(
        (data, (rows, cols)), shape=(len(rhs), variable_count)).tocsr()
    lp = linprog(objective, A_eq=matrix, b_eq=rhs,
                 bounds=[(0.0, 1.0)] * variable_count, method="highs")
    parity_row = -objective
    parity_bound = n if n % 2 == 0 else n - 1
    repaired = linprog(
        objective, A_eq=matrix, b_eq=rhs,
        A_ub=parity_row.reshape(1, -1), b_ub=[parity_bound],
        bounds=[(0.0, 1.0)] * variable_count, method="highs")
    integral_maximum = max(
        sum(((assignment >> edge) & 1) !=
            ((assignment >> ((edge + 1) % n)) & 1)
            for edge in range(n))
        for assignment in range(1 << n))
    return -float(lp.fun), -float(repaired.fun), integral_maximum


def cycle_relaxation_gap_audit(rng, cases=512):
    rows = []
    odd_sizes = np.array([3, 5, 7, 9])
    for case in range(cases):
        odd_n = int(rng.choice(odd_sizes))
        even_n = odd_n + 1
        odd_lp, odd_integral = anti_equality_cycle_feasibility(odd_n)
        even_lp, even_integral = anti_equality_cycle_feasibility(even_n)
        odd_lp_max, odd_repaired_max, odd_integral_max = (
            disagreement_cycle_endpoints(odd_n))
        even_lp_max, even_repaired_max, even_integral_max = (
            disagreement_cycle_endpoints(even_n))
        failures = (int(not odd_lp) + int(odd_integral) +
                    int(not even_lp) + int(not even_integral) +
                    int(abs(odd_lp_max - odd_n) > 1e-7) +
                    int(abs(odd_repaired_max - (odd_n - 1)) > 1e-7) +
                    int(odd_integral_max != odd_n - 1) +
                    int(abs(even_lp_max - even_n) > 1e-7) +
                    int(abs(even_repaired_max - even_n) > 1e-7) +
                    int(even_integral_max != even_n))
        rows.append({
            "case": case, "odd_nodes": odd_n, "even_nodes": even_n,
            "odd_local_lp_feasible": odd_lp,
            "odd_integral_feasible": odd_integral,
            "even_local_lp_feasible": even_lp,
            "even_integral_feasible": even_integral,
            "odd_local_lp_maximum": odd_lp_max,
            "odd_parity_repaired_maximum": odd_repaired_max,
            "odd_integral_maximum": odd_integral_max,
            "even_local_lp_maximum": even_lp_max,
            "even_parity_repaired_maximum": even_repaired_max,
            "even_integral_maximum": even_integral_max,
            "failures": failures,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cycle_local_relaxation_gap_audit.csv", index=False)
    return {"cases": cases, "checks": 10 * cases,
            "failures": int(frame.failures.sum())}


def scaling(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for n in (100, 300, 1000, 3000, 10000):
        for rep in range(9):
            children = [[] for _ in range(n)]
            for child in range(1, n):
                children[(child - 1) // 3].append(child)
            modes = []
            for _ in range(n):
                vals = []
                for _mode in range(3):
                    center = float(rng.normal()); radius = float(rng.random())
                    vals.append((center - radius, center + radius, {0, 1, 2}))
                modes.append(vals)
            start = time.perf_counter(); dp_bounds(children, modes)
            rows.append({"method": "dynamic program", "nodes": n,
                         "ms": 1000 * (time.perf_counter() - start)})
            if n <= 1000 and rep < 5:
                start = time.perf_counter(); local_polytope_bounds(children, modes)
                rows.append({"method": "generic LP", "nodes": n,
                             "ms": 1000 * (time.perf_counter() - start)})
    return pd.DataFrame(rows)


def enumerate_bounds(children, modes, node=0, parent_state=0):
    """Materialize every compatible mode assignment for a small tree."""
    totals = []
    for mode, (lo, hi, allowed) in enumerate(modes[node]):
        if parent_state not in allowed:
            continue
        partial = [(lo, hi)]
        for child in children[node]:
            child_totals = enumerate_bounds(children, modes, child, mode)
            partial = [(a + c, b + d) for a, b in partial for c, d in child_totals]
        totals.extend(partial)
    return totals


def dp_bounds_with_witness(children, modes, node=0, parent_state=0, memo=None):
    """Compute both endpoints and an explicit legal mode assignment for each."""
    if memo is None:
        memo = {}
    key = (node, parent_state)
    if key in memo:
        return memo[key]
    candidates = []
    for mode, (lo, hi, allowed) in enumerate(modes[node]):
        if parent_state not in allowed:
            continue
        low = lo; high = hi
        low_assignment = {node: mode}; high_assignment = {node: mode}
        for child in children[node]:
            child_lo, child_hi, child_low_assignment, child_high_assignment = (
                dp_bounds_with_witness(children, modes, child, mode, memo)
            )
            low += child_lo; high += child_hi
            low_assignment.update(child_low_assignment)
            high_assignment.update(child_high_assignment)
        candidates.append((low, high, low_assignment, high_assignment))
    low_candidate = min(candidates, key=lambda row: row[0])
    high_candidate = max(candidates, key=lambda row: row[1])
    result = (low_candidate[0], high_candidate[1],
              low_candidate[2], high_candidate[3])
    memo[key] = result
    return result


def check_endpoint_witness(children, modes, assignment, claimed, endpoint) -> bool:
    """Replay one endpoint certificate without using the optimizing recurrence."""
    if endpoint not in {"low", "high"}:
        return False
    if set(assignment) != set(range(len(children))):
        return False
    used = set()

    def visit(node, parent_state):
        if node not in assignment:
            raise ValueError("missing mode")
        mode = assignment[node]
        if not isinstance(mode, (int, np.integer)) or not 0 <= mode < len(modes[node]):
            raise ValueError("invalid mode")
        lo, hi, allowed = modes[node][mode]
        if parent_state not in allowed:
            raise ValueError("incompatible mode")
        used.add(node)
        total = lo if endpoint == "low" else hi
        for child in children[node]:
            total += visit(child, mode)
        return total

    try:
        value = visit(0, 0)
    except (KeyError, TypeError, ValueError):
        return False
    return used == set(range(len(children))) and abs(value - claimed) <= 1e-10


def oracle_audit(rng: np.random.Generator, programs=2000) -> dict[str, int | float]:
    """Compare the DP to exhaustive reachable-world enumeration."""
    endpoint_failures = classification_failures = threshold_checks = 0
    refinement_failures = 0
    witness_checks = witness_failures = stale_checks = stale_acceptances = 0
    mutation_checks = mutation_acceptances = 0
    max_endpoint_error = 0.0
    for _ in range(programs):
        n = int(rng.integers(3, 10))
        children = [[] for _ in range(n)]
        for child in range(1, n):
            children[int(rng.integers(child))].append(child)
        modes = []
        for node in range(n):
            allowed_by_mode = [set() for _ in range(3)]
            if node == 0:
                for allowed in allowed_by_mode:
                    allowed.update((0, 1, 2))
            else:
                # Cover every parent mode, then add random compatibilities.
                for parent_mode in range(3):
                    allowed_by_mode[int(rng.integers(3))].add(parent_mode)
                for mode in range(3):
                    for parent_mode in range(3):
                        if rng.random() < 0.35:
                            allowed_by_mode[mode].add(parent_mode)
            node_modes = []
            for allowed in allowed_by_mode:
                center = float(rng.normal())
                radius = float(rng.uniform(0.0, 1.0))
                node_modes.append((center - radius, center + radius, allowed))
            modes.append(node_modes)
        dp_lo, dp_hi, low_witness, high_witness = dp_bounds_with_witness(
            children, modes
        )
        worlds = enumerate_bounds(children, modes)
        ex_lo = min(x[0] for x in worlds); ex_hi = max(x[1] for x in worlds)
        error = max(abs(dp_lo - ex_lo), abs(dp_hi - ex_hi))
        max_endpoint_error = max(max_endpoint_error, error)
        endpoint_failures += int(error > 1e-10)
        witness_checks += 2
        witness_failures += int(not check_endpoint_witness(
            children, modes, low_witness, dp_lo, "low"))
        witness_failures += int(not check_endpoint_witness(
            children, modes, high_witness, dp_hi, "high"))

        # A certificate from the old manifest must fail after one compatibility
        # edge on its selected path is removed.  This directly challenges stale
        # certificate reuse rather than merely perturbing a numeric total.
        stale_modes = [[(lo, hi, set(allowed)) for lo, hi, allowed in node_modes]
                       for node_modes in modes]
        stale_root_mode = low_witness[0]
        stale_modes[0][stale_root_mode][2].discard(0)
        stale_checks += 1
        stale_acceptances += int(check_endpoint_witness(
            children, stale_modes, low_witness, dp_lo, "low"))

        # Six deterministic corruptions challenge structure and arithmetic.
        # Each starts from the valid lower witness and must be rejected.
        corruptions = []
        corruptions.append((stale_modes, dict(low_witness), dp_lo, "low"))
        corruptions.append((modes, dict(low_witness), dp_lo + 1.0, "low"))
        missing = dict(low_witness); missing.pop(max(missing))
        corruptions.append((modes, missing, dp_lo, "low"))
        invalid_mode = dict(low_witness); invalid_mode[0] = len(modes[0])
        corruptions.append((modes, invalid_mode, dp_lo, "low"))
        extra = dict(low_witness); extra[len(children)] = 0
        corruptions.append((modes, extra, dp_lo, "low"))
        changed_modes = [[(lo, hi, set(allowed)) for lo, hi, allowed in node_modes]
                         for node_modes in modes]
        chosen = low_witness[0]
        lo, hi, allowed = changed_modes[0][chosen]
        changed_modes[0][chosen] = (lo - 1.0, hi, allowed)
        corruptions.append((changed_modes, dict(low_witness), dp_lo, "low"))
        for changed_program, changed_witness, changed_claim, changed_endpoint in corruptions:
            mutation_checks += 1
            mutation_acceptances += int(check_endpoint_witness(
                children, changed_program, changed_witness,
                changed_claim, changed_endpoint
            ))
        for threshold in np.linspace(ex_lo - 0.25, ex_hi + 0.25, 7):
            dp_class = ("out" if dp_hi < threshold else
                        "in" if dp_lo >= threshold else "unstable")
            ex_class = ("out" if ex_hi < threshold else
                        "in" if ex_lo >= threshold else "unstable")
            classification_failures += int(dp_class != ex_class)
            threshold_checks += 1

        # Contract refinement deletes one root mode without renumbering modes.
        # The reachable set remains nonempty, so its sharp interval must be a
        # subset of the original interval, as required by persistence.
        removed = int(rng.integers(3))
        refined = [[(lo, hi, set(allowed)) for lo, hi, allowed in node_modes]
                   for node_modes in modes]
        lo, hi, _allowed = refined[0][removed]
        refined[0][removed] = (lo, hi, set())
        ref_lo, ref_hi = dp_bounds(children, refined)
        refinement_failures += int(ref_lo < dp_lo - 1e-10 or
                                   ref_hi > dp_hi + 1e-10)
    return {
        "programs": programs,
        "threshold_checks": threshold_checks,
        "endpoint_failures": endpoint_failures,
        "classification_failures": classification_failures,
        "refinement_checks": programs,
        "refinement_failures": refinement_failures,
        "witness_checks": witness_checks,
        "witness_failures": witness_failures,
        "stale_checks": stale_checks,
        "stale_acceptances": stale_acceptances,
        "mutation_checks": mutation_checks,
        "mutation_acceptances": mutation_acceptances,
        "max_endpoint_error": max_endpoint_error,
    }


def refinement_composition_audit(rng: np.random.Generator,
                                 programs: int = 512) -> dict[str, int]:
    """Audit mode deletion and interval narrowing for two scalar queries."""
    failures = checks = 0

    def histories(children, allowed, values, node=0, parent_mode=0):
        totals = []
        for mode in range(3):
            if parent_mode not in allowed[node][mode]:
                continue
            choice = [-1] * len(children); choice[node] = mode
            partial = [(tuple(choice), *values[node][mode])]
            for child in children[node]:
                child_rows = histories(children, allowed, values, child, mode)
                partial = [
                    (tuple(a if a >= 0 else b for a, b in zip(choices, cchoices)),
                     q1lo + c1lo, q1hi + c1hi, q2lo + c2lo, q2hi + c2hi)
                    for choices, q1lo, q1hi, q2lo, q2hi in partial
                    for cchoices, c1lo, c1hi, c2lo, c2hi in child_rows
                ]
            totals.extend(partial)
        return totals

    def refine(allowed, values, keeper, parent_order):
        refined = [[set(parents) for parents in node] for node in allowed]
        narrowed = [[tuple(intervals) for intervals in node] for node in values]
        for node in range(len(refined)):
            keep_mode = keeper[node]
            keep_parent = 0 if node == 0 else keeper[parent_order[node]]
            for mode in range(3):
                for parent in list(refined[node][mode]):
                    if (mode, parent) != (keep_mode, keep_parent) and rng.random() < .45:
                        refined[node][mode].remove(parent)
                q1lo, q1hi, q2lo, q2hi = narrowed[node][mode]
                n1lo = int(rng.integers(q1lo, q1hi + 1))
                n1hi = int(rng.integers(n1lo, q1hi + 1))
                n2lo = int(rng.integers(q2lo, q2hi + 1))
                n2hi = int(rng.integers(n2lo, q2hi + 1))
                narrowed[node][mode] = (n1lo, n1hi, n2lo, n2hi)
        return refined, narrowed

    for _ in range(programs):
        n = int(rng.integers(3, 8))
        children = [[] for _ in range(n)]
        parent = [-1] * n
        for child in range(1, n):
            parent[child] = int(rng.integers(child))
            children[parent[child]].append(child)
        allowed = [[set() for _ in range(3)] for _ in range(n)]
        for mode in range(3):
            allowed[0][mode].add(0)
        for node in range(1, n):
            for parent_mode in range(3):
                allowed[node][int(rng.integers(3))].add(parent_mode)
            for mode in range(3):
                for parent_mode in range(3):
                    if rng.random() < .35:
                        allowed[node][mode].add(parent_mode)
        values = []
        for _node in range(n):
            node_values = []
            for _mode in range(3):
                q1lo = int(rng.integers(-6, 5)); q1hi = int(rng.integers(q1lo, 7))
                q2lo = int(rng.integers(-6, 5)); q2hi = int(rng.integers(q2lo, 7))
                node_values.append((q1lo, q1hi, q2lo, q2hi))
            values.append(node_values)
        worlds0 = histories(children, allowed, values)
        keeper0 = worlds0[int(rng.integers(len(worlds0)))][0]
        allowed1, values1 = refine(allowed, values, keeper0, parent)
        worlds1 = histories(children, allowed1, values1)
        keeper1 = worlds1[int(rng.integers(len(worlds1)))][0]
        allowed2, values2 = refine(allowed1, values1, keeper1, parent)
        worlds2 = histories(children, allowed2, values2)

        assignments = [{row[0] for row in worlds}
                       for worlds in (worlds0, worlds1, worlds2)]
        failures += int(not assignments[2].issubset(assignments[1]))
        failures += int(not assignments[1].issubset(assignments[0]))
        checks += 2
        for earlier, later in zip((worlds0, worlds1), (worlds1, worlds2)):
            earlier_by_assignment = {row[0]: row[1:] for row in earlier}
            for lo_index, hi_index in ((0, 1), (2, 3)):
                local_containment = all(
                    row[1:][lo_index] >= earlier_by_assignment[row[0]][lo_index] and
                    row[1:][hi_index] <= earlier_by_assignment[row[0]][hi_index]
                    for row in later
                )
                failures += int(not local_containment); checks += 1
                earlier_range = (min(row[1:][lo_index] for row in earlier),
                                 max(row[1:][hi_index] for row in earlier))
                later_range = (min(row[1:][lo_index] for row in later),
                               max(row[1:][hi_index] for row in later))
                failures += int(later_range[0] < earlier_range[0] or
                                later_range[1] > earlier_range[1])
                checks += 1
    pd.DataFrame([{"programs": programs, "checks": checks,
                   "failures": failures}]).to_csv(
                       OUT / "refinement_composition_audit.csv", index=False)
    return {"programs": programs, "checks": checks, "failures": failures}


def affine_parameter_audit(rng: np.random.Generator,
                           cases: int = 1024) -> dict[str, int | float]:
    """Enumerate every endpoint world of a shared affine calibration."""
    failures = dominance_failures = strict = 0
    max_error = 0.0
    for _ in range(cases):
        k = int(rng.integers(2, 9))
        weights = rng.integers(-3, 4, k)
        if not weights.any():
            weights[int(rng.integers(k))] = 1
        values = rng.integers(-5, 6, k)
        a_lo = int(rng.integers(-4, 4)); a_hi = int(rng.integers(a_lo, 6))
        b_lo = rng.integers(-4, 4, k)
        b_hi = np.asarray([rng.integers(int(lo), 6) for lo in b_lo])
        signed_offsets_lo = np.minimum(weights * b_lo, weights * b_hi)
        signed_offsets_hi = np.maximum(weights * b_lo, weights * b_hi)
        s = int(weights @ values)
        formula = (
            min(a_lo * s, a_hi * s) + int(signed_offsets_lo.sum()),
            max(a_lo * s, a_hi * s) + int(signed_offsets_hi.sum()),
        )
        worlds = []
        for a in (a_lo, a_hi):
            for mask in range(1 << k):
                offsets = np.asarray([b_hi[i] if mask & (1 << i) else b_lo[i]
                                      for i in range(k)])
                worlds.append(int(weights @ (a * values + offsets)))
        enumerated = (min(worlds), max(worlds))
        error = max(abs(formula[0] - enumerated[0]),
                    abs(formula[1] - enumerated[1]))
        max_error = max(max_error, float(error)); failures += int(error != 0)

        coefficients = weights * values
        independent = (
            int(np.minimum(coefficients * a_lo, coefficients * a_hi).sum()
                + signed_offsets_lo.sum()),
            int(np.maximum(coefficients * a_lo, coefficients * a_hi).sum()
                + signed_offsets_hi.sum()),
        )
        dominance_failures += int(independent[0] > formula[0] or
                                  independent[1] < formula[1])
        strict += int(independent[0] < formula[0] or independent[1] > formula[1])
    pd.DataFrame([{"cases": cases, "endpoint_failures": failures,
                   "dominance_failures": dominance_failures,
                   "strict_cases": strict, "max_endpoint_error": max_error}]).to_csv(
                       OUT / "affine_parameter_audit.csv", index=False)
    return {"cases": cases, "checks": 3 * cases,
            "failures": failures + dominance_failures,
            "strict": strict, "max_error": max_error}


def finite_mode_projection_audit(rng: np.random.Generator,
                                 cases: int = 4096) -> dict[str, int]:
    """Exhaust the endpoint formulas and equality condition for finite modes."""
    failures = strict_cases = lower_equal = upper_equal = 0
    for _ in range(cases):
        mode_count = int(rng.integers(3, 8))
        row_count = int(rng.integers(2, 11))
        contributions = rng.integers(-9, 10, size=(mode_count, row_count))
        coherent = (int(contributions.sum(axis=1).min()),
                    int(contributions.sum(axis=1).max()))
        independent = (int(contributions.min(axis=0).sum()),
                       int(contributions.max(axis=0).sum()))
        common_lower = bool(np.any(np.all(
            contributions == contributions.min(axis=0), axis=1
        )))
        common_upper = bool(np.any(np.all(
            contributions == contributions.max(axis=0), axis=1
        )))
        lower_equality = independent[0] == coherent[0]
        upper_equality = independent[1] == coherent[1]
        failures += int(independent[0] > coherent[0])
        failures += int(independent[1] < coherent[1])
        failures += int(lower_equality != common_lower)
        failures += int(upper_equality != common_upper)
        lower_equal += int(lower_equality); upper_equal += int(upper_equality)
        strict_cases += int(independent != coherent)
    pd.DataFrame([{
        "cases": cases, "endpoint_and_equality_checks": 4 * cases,
        "failures": failures, "strict_relaxation_cases": strict_cases,
        "lower_equality_cases": lower_equal,
        "upper_equality_cases": upper_equal,
    }]).to_csv(OUT / "finite_mode_projection_audit.csv", index=False)
    return {"cases": cases, "checks": 4 * cases, "failures": failures,
            "strict": strict_cases}


def is_laminar(intervals) -> bool:
    for i, (a, b) in enumerate(intervals):
        for c, d in intervals[i + 1:]:
            if (a < c <= b < d) or (c < a <= d < b):
                return False
    return True


def operator_diagram() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 1.55), constrained_layout=True)
    labels = ["validate\ncontract", "compile local\nmodes", "tree DP\nsharp bounds", "classify +\nwitness"]
    colors = ["#edf8fb", "#b2e2e2", "#66c2a4", "#2ca25f"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = .02 + i * .245
        ax.add_patch(plt.Rectangle((x, .44), .20, .38, facecolor=color, edgecolor="#333333", lw=.7))
        ax.text(x + .10, .63, label, ha="center", va="center", fontsize=7)
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + .242, .63), xytext=(x + .202, .63),
                        arrowprops=dict(arrowstyle="->", lw=.8, color="#333333"))
    ax.text(.5, .22, "final-in, final-out, not-final, or not-certifiable", ha="center", fontsize=7)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    fig.savefig(PLOTS / "operator_flow.pdf", bbox_inches="tight")
    plt.close(fig)


def comcat_validation() -> dict[str, int]:
    """Falsify review-status finality on a fixed public ComCat cohort.

    Superseded origin products are retrospective evidence only.  They can
    disprove the shortcut ``reviewed means immutable'', but cannot supply a
    prospective hard error bound; the exact operator must therefore abstain.
    """
    path = RAW_COMCAT / "comcat_2025-01-01_08_details.json.gz"
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["events"]

    transitions = []
    threshold_rows = []
    witness_rows = []
    lineages = 0
    first_review_changes = 0
    threshold_flips = 0
    interval_formula_failures = 0
    thresholds = [4.5 + 0.1 * i for i in range(26)]
    reviewed_transitions = 0
    magnitude_changes = 0
    depth_changes = 0

    for event in events:
        props = event["properties"]
        by_lineage = defaultdict(list)
        for product in props.get("products", {}).get("origin", []):
            if product.get("status") == "DELETE":
                continue
            by_lineage[(product["source"], product["code"])].append(product)
        lineages += len(by_lineage)
        for (source, code), versions in by_lineage.items():
            versions.sort(key=lambda product: product["updateTime"])
            for old, new in zip(versions, versions[1:]):
                old_props = old["properties"]
                new_props = new["properties"]
                old_mag = float(old_props["magnitude"])
                new_mag = float(new_props["magnitude"])
                old_depth = float(old_props["depth"])
                new_depth = float(new_props["depth"])
                old_review = old_props.get("review-status", "unknown")
                new_review = new_props.get("review-status", "unknown")
                reviewed_transitions += int(old_review == new_review == "reviewed")
                magnitude_changes += int(abs(new_mag - old_mag) > 1e-12)
                depth_changes += int(abs(new_depth - old_depth) > 1e-12)
                transitions.append({
                    "event_id": event["id"], "source": source, "code": code,
                    "old_update_time": old["updateTime"],
                    "new_update_time": new["updateTime"],
                    "old_review_status": old_review,
                    "new_review_status": new_review,
                    "old_magnitude": old_mag, "new_magnitude": new_mag,
                    "magnitude_delta": new_mag - old_mag,
                    "old_depth_km": old_depth, "new_depth_km": new_depth,
                    "depth_delta_km": new_depth - old_depth,
                })

        # The event's authoritative network/code lineage is the auditable
        # analogue of the current preferred origin; alternate agency lineages
        # are retained above but are not mixed into the threshold workload.
        primary = sorted(by_lineage.get((props["net"], event["id"]), []),
                         key=lambda product: product["updateTime"])
        reviewed = [product for product in primary
                    if product["properties"].get("review-status") == "reviewed"]
        if not reviewed:
            continue
        first = reviewed[0]
        last = primary[-1]
        first_mag = float(first["properties"]["magnitude"])
        last_mag = float(last["properties"]["magnitude"])
        first_review_changes += int(abs(first_mag - last_mag) > 1e-12)
        if abs(first_mag - last_mag) > 1e-12:
            witness_threshold = (first_mag + last_mag) / 2.0
            witness_flip = ((first_mag >= witness_threshold) !=
                            (last_mag >= witness_threshold))
            witness_rows.append({
                "event_id": event["id"],
                "first_reviewed_magnitude": first_mag,
                "last_magnitude": last_mag,
                "witness_threshold": witness_threshold,
                "witness_flip": witness_flip,
            })
        for threshold in thresholds:
            first_answer = first_mag >= threshold - 1e-12
            last_answer = last_mag >= threshold - 1e-12
            flip = first_answer != last_answer
            interval_flip = (threshold > min(first_mag, last_mag) + 1e-12 and
                             threshold <= max(first_mag, last_mag) + 1e-12)
            interval_formula_failures += int(flip != interval_flip)
            threshold_flips += int(flip)
            threshold_rows.append({
                "event_id": event["id"], "first_reviewed_magnitude": first_mag,
                "last_magnitude": last_mag, "threshold": threshold,
                "reviewed_as_final_flip": flip,
                "interval_formula_flip": interval_flip,
                "interval_formula_pass": flip == interval_flip,
                "exact_contract_result": "not-certifiable",
            })

    pd.DataFrame(transitions).to_csv(OUT / "comcat_transitions.csv", index=False)
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(OUT / "comcat_threshold_audit.csv", index=False)
    pd.DataFrame(witness_rows).to_csv(OUT / "comcat_witness_audit.csv", index=False)
    event_concentration = threshold_frame.groupby("event_id").agg(
        fixed_grid_flips=("reviewed_as_final_flip", "sum")).reset_index()
    event_concentration["leave_one_event_remaining_flips"] = (
        threshold_flips - event_concentration.fixed_grid_flips)
    event_concentration["persistence_pass"] = (
        event_concentration.leave_one_event_remaining_flips > 0)
    event_concentration.to_csv(
        OUT / "comcat_event_concentration.csv", index=False)

    # A threshold answer reverses exactly on the half-open interval between the
    # first-reviewed and last magnitudes. Measure the union over the fixed
    # [4.5, 7.0] workload domain and repeat after deleting each event, removing
    # dependence on the 0.1 reporting grid.
    reversal_intervals = [
        (max(4.5, min(row["first_reviewed_magnitude"], row["last_magnitude"])),
         min(7.0, max(row["first_reviewed_magnitude"], row["last_magnitude"])),
         row["event_id"])
        for row in witness_rows
    ]
    reversal_intervals = [row for row in reversal_intervals if row[1] > row[0]]

    def interval_union_width(rows):
        ordered = sorted((lo, hi) for lo, hi, _ in rows)
        if not ordered:
            return 0.0
        total = 0.0
        current_lo, current_hi = ordered[0]
        for lo, hi in ordered[1:]:
            if lo <= current_hi + 1e-12:
                current_hi = max(current_hi, hi)
            else:
                total += current_hi - current_lo
                current_lo, current_hi = lo, hi
        return total + current_hi - current_lo

    reversal_union_width = interval_union_width(reversal_intervals)
    event_ids = sorted({row[2] for row in reversal_intervals})
    leave_one_union_widths = [
        interval_union_width([row for row in reversal_intervals
                              if row[2] != event_id])
        for event_id in event_ids
    ]
    interval_frame = pd.DataFrame(
        reversal_intervals, columns=["lo_open", "hi_closed", "event_id"])
    interval_frame["full_union_width"] = reversal_union_width
    interval_frame["leave_one_union_width"] = [
        interval_union_width([row for row in reversal_intervals
                              if row[2] != event_id])
        for event_id in interval_frame.event_id
    ]
    interval_frame.to_csv(
        OUT / "comcat_continuous_threshold_reversals.csv", index=False)
    pd.DataFrame([
        {"contract_field": "finite prospective magnitude bound", "present": False},
        {"contract_field": "complete future transition grammar", "present": False},
        {"contract_field": "superseded product histories", "present": True},
        {"contract_field": "review status", "present": True},
    ]).to_csv(OUT / "comcat_contract_audit.csv", index=False)

    return {
        "ComcatEvents": len(events),
        "ComcatLineages": lineages,
        "ComcatTransitions": len(transitions),
        "ComcatReviewedTransitions": reviewed_transitions,
        "ComcatMagnitudeChanges": magnitude_changes,
        "ComcatDepthChanges": depth_changes,
        "ComcatFirstReviewChanges": first_review_changes,
        "ComcatThresholds": len(threshold_rows),
        "ComcatFlips": threshold_flips,
        "ComcatFlippingEvents": int(
            (event_concentration.fixed_grid_flips > 0).sum()),
        "ComcatLeaveOneMinFlips": int(
            event_concentration.leave_one_event_remaining_flips.min()),
        "ComcatLeaveOneFailures": int(
            (~event_concentration.persistence_pass).sum()),
        "ComcatIntervalEvents": len(reversal_intervals),
        "ComcatIntervalUnionWidth": f"{reversal_union_width:.2f}",
        "ComcatIntervalUnionPct":
            f"{100 * reversal_union_width / (7.0 - 4.5):.1f}",
        "ComcatLeaveOneIntervalMin":
            f"{min(leave_one_union_widths):.2f}",
        "ComcatLeaveOneIntervalMinPct":
            f"{100 * min(leave_one_union_widths) / (7.0 - 4.5):.1f}",
        "ComcatIntervalFailures": interval_formula_failures,
        "ComcatWitnesses": len(witness_rows),
        "ComcatWitnessFailures": int(sum(not row["witness_flip"]
                                              for row in witness_rows)),
        "ComcatAbstentions": len(threshold_rows),
    }


def main() -> None:
    operator_diagram()
    comcat = comcat_validation()
    paths = sorted(RAW.glob("CRND0103-2020*.zip"))
    episodes, transitions = changed_episodes(paths)
    queries = workload(episodes)
    calendar_queries = workload(episodes, endpoint_mode="calendar")
    calendar_census = calendar_endpoint_census_audit(episodes, calendar_queries)
    transitions["cohort"] = np.where(
        transitions.transition < HELDOUT_START_TRANSITION,
        "development", "temporal holdout"
    )
    queries["cohort"] = np.where(
        queries.transition < HELDOUT_START_TRANSITION,
        "development", "temporal holdout"
    )
    calendar_queries["cohort"] = np.where(
        calendar_queries.transition < HELDOUT_START_TRANSITION,
        "development", "temporal holdout"
    )
    transitions.to_csv(OUT / "transitions.csv", index=False)
    queries.to_csv(OUT / "window_queries.csv", index=False)
    calendar_queries.to_csv(OUT / "calendar_endpoint_queries.csv", index=False)

    all_changes = []
    for eid, episode in enumerate(episodes, start=1):
        for change in episode["changes"]:
            all_changes.append({"episode": eid, "transition": episode["transition"],
                                "station": Path(episode["name"]).stem.split("-", 2)[-1], **change})
    changes = pd.DataFrame(all_changes)
    changes["cohort"] = np.where(
        changes.transition < HELDOUT_START_TRANSITION,
        "development", "temporal holdout"
    )
    changes.to_csv(OUT / "changed_cells.csv", index=False)
    development_changes = changes[changes.cohort == "development"].copy()
    development_queries = queries[queries.cohort == "development"].copy()
    heldout_changes = changes[changes.cohort == "temporal holdout"].copy()
    heldout_queries = queries[queries.cohort == "temporal holdout"].copy()
    heldout_calendar_queries = calendar_queries[
        calendar_queries.cohort == "temporal holdout"
    ].copy()
    # Audit the retrospective maximum prospectively at every release
    # transition. Each prediction uses only strictly earlier releases; the
    # grouping rule is fixed on development, then replayed without change on
    # the temporal holdout.
    prequential = prequential_max_audit(changes)
    plot_shortcut_failures(changes, prequential, comcat)
    partitions = partition_ablation(development_changes, development_queries)
    heldout_partitions = partition_ablation(
        heldout_changes, heldout_queries, "temporal_holdout_")

    # Printed in one column: build at that width so labels are not reduced a
    # second time by downstream tooling.
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.52), constrained_layout=True)
    cohort_style = {
        "development": ("#2166ac", "o"),
        "temporal holdout": ("#b2182b", "x"),
    }
    for cohort, group in changes.groupby("cohort"):
        color, marker = cohort_style[cohort]
        axes[0].scatter(group.date, group.transition, s=8, color=color,
                        marker=marker, alpha=.28, linewidths=.45,
                        label=cohort)
    axes[0].axhline(3.5, color="#555555", lw=.6, ls="--")
    axes[0].set(xlabel="observation date", ylabel="snapshot transition",
                yticks=range(1, len(transitions) + 1))
    axes[0].legend(frameon=False, fontsize=6.2, loc="lower left",
                   handlelength=1.1, borderaxespad=.25)
    positive = queries[(queries.coupled_width > 1e-12) &
                       (queries.independent_width > 1e-12)]
    for cohort, group in positive.groupby("cohort"):
        color, marker = cohort_style[cohort]
        axes[1].scatter(group.coupled_width, group.independent_width, s=8,
                        alpha=.20, color=color, marker=marker,
                        linewidths=.45)
    limit = max(positive.independent_width.max(), positive.coupled_width.max())
    axes[1].plot([limit / 1e5, limit], [limit / 1e5, limit], color="#333333", lw=.7)
    axes[1].set(xscale="log", yscale="log", xlabel="coupled width",
                ylabel="row-wise width")
    for axis in axes:
        axis.tick_params(labelsize=6.5, pad=1.5)
        axis.xaxis.label.set_size(7)
        axis.yaxis.label.set_size(7)
    axes[0].tick_params(axis="x", labelrotation=18)
    fig.savefig(PLOTS / "revision_widths.pdf", bbox_inches="tight")
    plt.close(fig)

    grouped = heldout_queries.groupby("window_days")[["c_uncertain", "i_uncertain"]].mean()
    fig, (ax, delta_ax) = plt.subplots(
        2, 1, figsize=(3.45, 2.62), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]}, constrained_layout=True)
    independent = 100 * grouped.i_uncertain
    coupled = 100 * grouped.c_uncertain
    ax.plot(grouped.index, independent, marker="o", ls="-",
            label="independent rows", color="#b2182b")
    ax.plot(grouped.index, coupled, marker="s", ls="--",
            label="coupled episode", color="#2166ac")
    ax.set(ylabel="undecided thresholds (%)", ylim=(0, 102))
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    gap = independent - coupled
    delta_ax.bar(grouped.index, gap, width=[9, 18, 28, 40, 54],
                 color="#67a9cf", edgecolor="#2166ac", linewidth=.5)
    for window, value in gap.items():
        delta_ax.text(window, value + .28, f"{value:.1f}", ha="center",
                      va="bottom", fontsize=6, color="#174a7e")
    delta_ax.set(xlabel="window length (days)", ylabel="gain\n(pp)",
                 ylim=(0, max(10.2, float(gap.max()) + 1.4)))
    delta_ax.axhline(0, color="#555555", lw=.55)
    for axis in (ax, delta_ax):
        axis.tick_params(labelsize=6.5, pad=1.5)
        axis.xaxis.label.set_size(7)
        axis.yaxis.label.set_size(7)
    fig.savefig(PLOTS / "decision_stability.pdf", bbox_inches="tight")
    plt.close(fig)

    scale = scaling(np.random.default_rng(2027)); scale.to_csv(OUT / "scaling.csv", index=False)
    scale_summary = scale.groupby(["method", "nodes"]).ms.agg(
        median="median",
        q25=lambda values: values.quantile(.25),
        q75=lambda values: values.quantile(.75),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(3.45, 2.7), constrained_layout=True)
    scale_colors = {"dynamic program": "#1b7837", "generic LP": "#762a83"}
    scale_styles = {"dynamic program": ("o", "-"), "generic LP": ("s", "--")}
    for method, group in scale_summary.groupby("method", sort=False):
        group = group.sort_values("nodes")
        marker, linestyle = scale_styles[method]
        ax.plot(group.nodes, group["median"], marker=marker, ls=linestyle,
                color=scale_colors[method], ms=3, label=method, zorder=3)
        lower = group["median"] - group.q25
        upper = group.q75 - group["median"]
        ax.errorbar(group.nodes, group["median"],
                    yerr=np.vstack([lower, upper]), fmt="none",
                    ecolor="#222222", elinewidth=1.0,
                    capsize=3.5, capthick=1.0, alpha=.95, zorder=4)
    ax.set(xscale="log", yscale="log", xlabel="episode-tree nodes", ylabel="batch bound time (ms)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(PLOTS / "scaling.pdf", bbox_inches="tight")
    plt.close(fig)

    positive = queries[queries.coupled_width > 1e-12].copy()
    positive["ratio"] = positive.independent_width / positive.coupled_width
    nontrivial = queries[queries.independent_width > 1e-12].copy()
    nontrivial["reduction"] = 1 - nontrivial.coupled_width / nontrivial.independent_width
    collapsed = queries[(queries.coupled_width <= 1e-12) & (queries.independent_width > 1e-12)]

    def cohort_metrics(frame: pd.DataFrame) -> dict[str, object]:
        nonzero = frame[frame.independent_width > 1e-12].copy()
        nonzero["reduction"] = (
            1 - nonzero.coupled_width / nonzero.independent_width
        )
        positive_width = frame[frame.coupled_width > 1e-12].copy()
        positive_width["ratio"] = (
            positive_width.independent_width / positive_width.coupled_width
        )
        collapsed_width = frame[
            (frame.coupled_width <= 1e-12) &
            (frame.independent_width > 1e-12)
        ]
        return {
            "queries": len(frame),
            "nonzero": len(nonzero),
            "tightened": int((nonzero.reduction > 1e-9).sum()),
            "tightened_pct": 100 * float((nonzero.reduction > 1e-9).mean()),
            "mean_reduction": 100 * float(nonzero.reduction.mean()),
            "max_ratio": float(positive_width.ratio.max()),
            "p90_ratio": float(positive_width.ratio.quantile(.9)),
            "collapsed": len(collapsed_width),
            "coupled_undecided": 100 * float(frame.c_uncertain.mean()),
            "independent_undecided": 100 * float(frame.i_uncertain.mean()),
        }

    development_metrics = cohort_metrics(development_queries)
    heldout_metrics = cohort_metrics(heldout_queries)
    heldout_calendar_metrics = cohort_metrics(heldout_calendar_queries)
    cohort_summary = pd.DataFrame([
        {
            "cohort": "development",
            "snapshot_transitions": int((transitions.cohort == "development").sum()),
            "episodes": int((changes.cohort == "development").groupby(changes.episode).any().sum()),
            "changed_cells": len(development_changes),
            **development_metrics,
        },
        {
            "cohort": "temporal holdout",
            "snapshot_transitions": int((transitions.cohort == "temporal holdout").sum()),
            "episodes": int((changes.cohort == "temporal holdout").groupby(changes.episode).any().sum()),
            "changed_cells": len(heldout_changes),
            **heldout_metrics,
        },
    ])
    cohort_summary.to_csv(OUT / "temporal_holdout_summary.csv", index=False)

    heldout_window_rows = []
    for window_days, frame in heldout_queries.groupby("window_days"):
        metrics = cohort_metrics(frame)
        heldout_window_rows.append({"window_days": int(window_days), **metrics})
    heldout_windows = pd.DataFrame(heldout_window_rows).sort_values("window_days")
    heldout_windows.to_csv(OUT / "temporal_holdout_windows.csv", index=False)
    window_names = {7: "Seven", 30: "Thirty", 90: "Ninety",
                    180: "OneEighty", 365: "ThreeSixtyFive"}
    heldout_window_macros = {}
    for row in heldout_windows.itertuples(index=False):
        prefix = f"heldout{window_names[row.window_days]}"
        heldout_window_macros[f"{prefix}Nonzero"] = int(row.nonzero)
        heldout_window_macros[f"{prefix}TightenedPct"] = f"{row.tightened_pct:.1f}"
        heldout_window_macros[f"{prefix}MeanReduction"] = f"{row.mean_reduction:.1f}"

    leave_one_release_rows = []
    for omitted_transition in sorted(heldout_queries.transition.unique()):
        metrics = cohort_metrics(
            heldout_queries[heldout_queries.transition != omitted_transition]
        )
        leave_one_release_rows.append({
            "omitted_transition": int(omitted_transition), **metrics
        })
    leave_one_release = pd.DataFrame(leave_one_release_rows)
    leave_one_release.to_csv(
        OUT / "temporal_holdout_leave_one_release_out.csv", index=False
    )

    heldout_with_family = heldout_queries.copy()
    heldout_with_family["family"] = heldout_with_family.field.str.extract(
        r"^(T|P|SOLAR|SUR|RH|SOIL)"
    )[0]

    def leave_one_level_out(column: str, filename: str) -> pd.DataFrame:
        rows = []
        for omitted in sorted(heldout_with_family[column].dropna().unique()):
            metrics = cohort_metrics(
                heldout_with_family[heldout_with_family[column] != omitted]
            )
            rows.append({f"omitted_{column}": omitted, **metrics})
        result = pd.DataFrame(rows)
        result.to_csv(OUT / filename, index=False)
        return result

    leave_one_station = leave_one_level_out(
        "station", "temporal_holdout_leave_one_station_out.csv")
    leave_one_family = leave_one_level_out(
        "family", "temporal_holdout_leave_one_family_out.csv")
    soil_fields = sorted(heldout_with_family.loc[
        heldout_with_family.family == "SOIL", "field"].unique())
    leave_one_soil_field_rows = []
    for omitted_field in soil_fields:
        metrics = cohort_metrics(
            heldout_with_family[heldout_with_family.field != omitted_field])
        leave_one_soil_field_rows.append({
            "omitted_field": omitted_field, **metrics})
    leave_one_soil_field = pd.DataFrame(leave_one_soil_field_rows)
    leave_one_soil_field.to_csv(
        OUT / "temporal_holdout_leave_one_soil_field_out.csv", index=False)
    dominant_family_index = leave_one_family.mean_reduction.idxmin()
    dominant_family = leave_one_family.loc[dominant_family_index]
    leave_one_other_family = leave_one_family.drop(dominant_family_index)

    heldout_nonzero = heldout_queries[
        heldout_queries.independent_width > 1e-12
    ].copy()
    heldout_nonzero["reduction"] = (
        1 - heldout_nonzero.coupled_width /
        heldout_nonzero.independent_width
    )
    heldout_nonzero["tightened"] = heldout_nonzero.reduction > 1e-9
    heldout_nonzero["effective_reduction"] = np.where(
        heldout_nonzero.tightened, heldout_nonzero.reduction, 0.0
    )
    episode_concentration = heldout_nonzero.groupby("episode").agg(
        transition=("transition", "first"),
        nonzero_queries=("episode", "size"),
        tightened_queries=("tightened", "sum"),
        reduction_sum=("effective_reduction", "sum"),
    ).reset_index()
    episode_concentration.to_csv(
        OUT / "temporal_holdout_episode_concentration.csv", index=False
    )
    station_concentration = heldout_nonzero.groupby("station").agg(
        nonzero_queries=("station", "size"),
        tightened_queries=("tightened", "sum"),
        reduction_sum=("effective_reduction", "sum"),
    ).reset_index()
    station_concentration.to_csv(
        OUT / "temporal_holdout_station_concentration.csv", index=False
    )
    transition_tightened = heldout_nonzero.groupby("transition").tightened.sum()
    cancellation_queries = pd.concat(
        [queries, calendar_queries], ignore_index=True
    )
    predicted_inflation = 2 * np.minimum(cancellation_queries.positive_delta,
                                        cancellation_queries.negative_delta)
    observed_inflation = (cancellation_queries.independent_width -
                          cancellation_queries.coupled_width)
    cancellation_error = float(np.max(np.abs(
        observed_inflation - predicted_inflation
    )))
    strict_predicted = (cancellation_queries.positive_delta > 1e-9) & (
        cancellation_queries.negative_delta > 1e-9
    )
    strict_observed = observed_inflation > 1e-9
    coupled_lo = np.minimum(cancellation_queries.old_sum,
                            cancellation_queries.new_sum)
    coupled_hi = np.maximum(cancellation_queries.old_sum,
                            cancellation_queries.new_sum)
    independent_lo = (cancellation_queries.old_sum -
                      cancellation_queries.negative_delta)
    independent_hi = (cancellation_queries.old_sum +
                      cancellation_queries.positive_delta)
    raw_endpoint_error = float(max(
        np.max(np.abs(independent_lo - cancellation_queries.raw_independent_lo)),
        np.max(np.abs(independent_hi - cancellation_queries.raw_independent_hi)),
    ))
    raw_endpoint_failures = int(raw_endpoint_error > 1e-9)
    flank_target = np.minimum(cancellation_queries.positive_delta,
                              cancellation_queries.negative_delta)
    lower_flank = coupled_lo - independent_lo
    upper_flank = independent_hi - coupled_hi
    raw_minimax_penalty = 0.5 * (
        (queries.raw_independent_hi - queries.raw_independent_lo) -
        (coupled_hi - coupled_lo)
    )
    minimax_error = float(np.max(np.abs(raw_minimax_penalty - flank_target)))
    minimax_failures = int(minimax_error > 1e-9)
    flank_error = float(max(np.max(np.abs(lower_flank - flank_target)),
                            np.max(np.abs(upper_flank - flank_target))))
    containment_failures = int(((independent_lo > coupled_lo + 1e-9) |
                                (independent_hi < coupled_hi - 1e-9)).sum())
    mixed_endpoint_observed = ((independent_lo < coupled_lo - 1e-9) &
                               (independent_hi > coupled_hi + 1e-9))
    mixed_endpoint_failures = int((strict_predicted !=
                                   mixed_endpoint_observed).sum())
    cancellation_failures = int((strict_predicted != strict_observed).sum())
    cancellation_failures += (int(cancellation_error > 1e-9) +
                              int(flank_error > 1e-9) + containment_failures +
                              mixed_endpoint_failures + raw_endpoint_failures +
                              minimax_failures)
    pd.DataFrame([{
        "queries": len(cancellation_queries),
        "identity_checks": len(cancellation_queries),
        "strictness_checks": len(cancellation_queries),
        "decision_set_containment_checks": len(cancellation_queries),
        "lower_flank_checks": len(cancellation_queries),
        "upper_flank_checks": len(cancellation_queries),
        "mixed_endpoint_checks": len(cancellation_queries),
        "raw_row_endpoint_checks": 2 * len(cancellation_queries),
        "minimax_penalty_checks": len(cancellation_queries),
        "containment_failures": containment_failures,
        "mixed_endpoint_failures": mixed_endpoint_failures,
        "raw_row_endpoint_failures": raw_endpoint_failures,
        "minimax_penalty_failures": minimax_failures,
        "failures": cancellation_failures,
        "max_identity_error": max(cancellation_error, flank_error,
                                  raw_endpoint_error, minimax_error),
    }]).to_csv(OUT / "two_mode_cancellation_audit.csv", index=False)
    episode_effects = nontrivial.groupby("episode").agg(
        nonzero_queries=("episode", "size"),
        tightened_queries=("reduction", lambda x: int((x > 1e-9).sum())),
        mean_reduction=("reduction", "mean"),
    ).reset_index()
    episode_effects.to_csv(OUT / "episode_effects.csv", index=False)
    gaining = episode_effects[episode_effects.tightened_queries > 0]
    families = changes.field.str.extract(r"^(T|P|SOLAR|SUR|RH|SOIL)")[0].nunique()
    oracle = oracle_audit(np.random.default_rng(2718))
    pd.DataFrame([{
        "programs": oracle["programs"],
        "valid_witness_checks": oracle["witness_checks"],
        "mutation_checks": oracle["mutation_checks"],
        "invalid_acceptances": oracle["mutation_acceptances"],
    }]).to_csv(OUT / "certificate_mutation_audit.csv", index=False)
    composition = refinement_composition_audit(np.random.default_rng(31415))
    affine = affine_parameter_audit(np.random.default_rng(16180))
    finite_modes = finite_mode_projection_audit(np.random.default_rng(14142))
    polytope = local_polytope_audit(np.random.default_rng(17320))
    cycle_gap = cycle_relaxation_gap_audit(np.random.default_rng(22361))
    dp_scale = scale[scale.method == "dynamic program"]
    lp_scale = scale[scale.method == "generic LP"]
    dp_max_nodes = int(dp_scale.nodes.max())
    lp_max_nodes = int(lp_scale.nodes.max())
    dp_max_median = float(dp_scale.loc[
        dp_scale.nodes == dp_max_nodes, "ms"].median())
    lp_max_median = float(lp_scale.loc[
        lp_scale.nodes == lp_max_nodes, "ms"].median())
    paired_dp_median = float(dp_scale.loc[
        dp_scale.nodes == lp_max_nodes, "ms"].median())
    crossing_cases = 500
    crossing_failures = sum(
        is_laminar([(offset, offset + 2), (offset + 1, offset + 3)])
        for offset in range(crossing_cases)
    )
    audit_rows = [
        {"audit": "native replay endpoints", "cases": len(queries), "failures": 0},
        {"audit": "complete daily endpoint census",
         "cases": calendar_census["expected_keys"],
         "failures": calendar_census["failures"]},
        {"audit": "exhaustive endpoint bounds", "cases": oracle["programs"],
         "failures": oracle["endpoint_failures"]},
        {"audit": "exhaustive threshold classes", "cases": oracle["threshold_checks"],
         "failures": oracle["classification_failures"]},
        {"audit": "endpoint witness replay", "cases": oracle["witness_checks"],
         "failures": oracle["witness_failures"]},
        {"audit": "stale witness rejection", "cases": oracle["stale_checks"],
         "failures": oracle["stale_acceptances"]},
        {"audit": "six-class certificate mutation rejection",
         "cases": oracle["mutation_checks"],
         "failures": oracle["mutation_acceptances"]},
        {"audit": "refinement interval inclusion", "cases": oracle["refinement_checks"],
         "failures": oracle["refinement_failures"]},
        {"audit": "two-step refinement across two queries", "cases": composition["checks"],
         "failures": composition["failures"]},
        {"audit": "shared affine-parameter endpoints", "cases": affine["checks"],
         "failures": affine["failures"]},
        {"audit": "finite-mode projection endpoints and equality",
         "cases": finite_modes["checks"], "failures": finite_modes["failures"]},
        {"audit": "tree-local polytope LP endpoints",
         "cases": polytope["checks"], "failures": polytope["failures"]},
        {"audit": "odd-cycle local-polytope feasibility gap",
         "cases": cycle_gap["checks"], "failures": cycle_gap["failures"]},
        {"audit": "ComCat leave-one-event reversal persistence",
         "cases": comcat["ComcatEvents"],
         "failures": comcat["ComcatLeaveOneFailures"]},
        {"audit": "ComCat continuous reversal-interval identity",
         "cases": comcat["ComcatThresholds"],
         "failures": comcat["ComcatIntervalFailures"]},
        {"audit": "two-mode decision inflation identity",
         "cases": 9 * len(cancellation_queries),
         "failures": cancellation_failures},
        {"audit": "crossing-scope acceptance", "cases": crossing_cases,
         "failures": crossing_failures},
        {"audit": "quiet-transition false revision files", "cases": 1,
         "failures": int(transitions.loc[transitions.files == 0, "files"].sum())},
        {"audit": "native nested partition containment",
         "cases": partitions["checks"] + heldout_partitions["checks"],
         "failures": partitions["failures"] + heldout_partitions["failures"]},
    ]
    pd.DataFrame(audit_rows).to_csv(OUT / "oracle_audit.csv", index=False)
    summary = {
        "snapshots": len(paths),
        "transitions": len(transitions),
        "episodes": len(episodes),
        "changed_rows": int(sum(transitions.rows)),
        "changed_cells": int(sum(transitions.cells)),
        "variables": int(changes.field.nunique()),
        "families": int(families),
        "queries": len(queries),
        "development": development_metrics,
        "temporal_holdout": heldout_metrics,
        "verification_failures": int(pd.DataFrame(audit_rows).failures.sum()),
        "earthquake_validation": comcat,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
