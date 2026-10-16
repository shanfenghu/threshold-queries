from pathlib import Path

import pandas as pd
import pytest

from threshold_queries.workflow import (
    check_endpoint_witness,
    dp_bounds,
    dp_bounds_with_witness,
    quiet_transition_failures,
)


def small_program():
    children = [[1, 2], [], []]
    modes = [
        [(0, 0, {0}), (0, 0, {0})],
        [(1, 2, {0}), (5, 6, {1})],
        [(3, 4, {0}), (7, 8, {1})],
    ]
    return children, modes


def test_dynamic_program_respects_shared_parent_mode():
    children, modes = small_program()
    assert dp_bounds(children, modes) == (4, 14)


def test_endpoint_witnesses_replay_independently():
    children, modes = small_program()
    lower, upper, low_assignment, high_assignment = dp_bounds_with_witness(
        children, modes
    )
    assert check_endpoint_witness(children, modes, low_assignment, lower, "low")
    assert check_endpoint_witness(children, modes, high_assignment, upper, "high")


def test_stale_witness_is_rejected():
    children, modes = small_program()
    lower, _, assignment, _ = dp_bounds_with_witness(children, modes)
    assignment[1] = 1
    assert not check_endpoint_witness(children, modes, assignment, lower, "low")


def quiet_control():
    paths = [Path("CRND0103-202010260450.zip"), Path("CRND0103-202011020550.zip")]
    transitions = pd.DataFrame([{"transition": 1, "files": 0, "rows": 0, "cells": 0}])
    return paths, transitions


def test_declared_quiet_transition_is_unchanged():
    paths, transitions = quiet_control()
    assert quiet_transition_failures(paths, transitions, []) == 0


@pytest.mark.parametrize("column", ["files", "rows", "cells"])
def test_quiet_transition_rejects_unexpected_revisions(column):
    paths, transitions = quiet_control()
    transitions.loc[0, column] = 1
    assert quiet_transition_failures(paths, transitions, []) == 1


def test_quiet_transition_rejects_episode_even_when_counts_claim_zero():
    paths, transitions = quiet_control()
    assert quiet_transition_failures(paths, transitions, [{"transition": 1}]) == 1


def test_quiet_transition_rejects_missing_control():
    paths, transitions = quiet_control()
    assert quiet_transition_failures(paths[:1], transitions, []) == 1
    assert quiet_transition_failures(paths, transitions.iloc[:0], []) == 1
