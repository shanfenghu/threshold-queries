from threshold_queries.workflow import (
    check_endpoint_witness,
    dp_bounds,
    dp_bounds_with_witness,
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
