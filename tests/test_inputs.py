import hashlib

from threshold_queries.inputs import InputSpec, inspect_inputs, snapshot_paths


def test_input_inspection_reports_missing_file(tmp_path):
    specs = [InputSpec("source/value.bin", "unused")]
    assert inspect_inputs(tmp_path, specs) == ["missing: source/value.bin"]


def test_input_inspection_accepts_expected_digest(tmp_path):
    payload = b"immutable scientific input"
    path = tmp_path / "source" / "value.bin"
    path.parent.mkdir()
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert inspect_inputs(tmp_path, [InputSpec("source/value.bin", expected)]) == []


def test_input_inspection_rejects_changed_content(tmp_path):
    path = tmp_path / "source" / "value.bin"
    path.parent.mkdir()
    path.write_bytes(b"changed")
    diagnostics = inspect_inputs(
        tmp_path,
        [InputSpec("source/value.bin", hashlib.sha256(b"original").hexdigest())],
    )
    assert diagnostics == ["digest mismatch: source/value.bin"]


def test_snapshot_selection_ignores_unlisted_archives(tmp_path):
    directory = tmp_path / "raw" / "uscrn"
    directory.mkdir(parents=True)
    (directory / "CRND0103-202001010000.zip").write_bytes(b"unrelated archive")
    paths = snapshot_paths(tmp_path)
    assert len(paths) == 8
    assert paths[0].name == "CRND0103-202010120450.zip"
    assert paths[-1].name == "CRND0103-202011300550.zip"
    assert all(path.name != "CRND0103-202001010000.zip" for path in paths)
