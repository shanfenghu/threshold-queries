"""Validate the immutable inputs used by the evaluation workflow."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(os.environ.get("THRESHOLD_QUERIES_ROOT", Path.cwd())).resolve()
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class InputSpec:
    """Expected relative path and SHA-256 digest for one input."""

    path: str
    sha256: str


EXPECTED_INPUTS = (
    InputSpec(
        "raw/uscrn/CRND0103-202010120450.zip",
        "a0225a61ab0686603bd4b5ecfc71be40bc53788cdb26ebd4d3c1050cdb09d247",
    ),
    InputSpec(
        "raw/uscrn/CRND0103-202010190450.zip",
        "0f258dbb8127113528a77937096be61ba4a10535e5c50917010a7552c2d7a3a2",
    ),
    InputSpec(
        "raw/uscrn/CRND0103-202010260450.zip",
        "d3fda53966b34e093d7a7a76e959428f687eac1f7ee317dda461cb8703cc3e96",
    ),
    InputSpec(
        "raw/uscrn/CRND0103-202011020550.zip",
        "4c234c9d6228638badfbc744dc1c450403d054ba9eb2ba748b0711ebdc25ce19",
    ),
    InputSpec(
        "raw/uscrn/CRND0103-202011090550.zip",
        "981346b548faa729e92a7e07c3e60b9ce8e5d3249b04ed735357a9c892ae3e0d",
    ),
    InputSpec(
        "raw/uscrn/CRND0103-202011160550.zip",
        "e53b0203565d39177183bc34646d4487a2a88863e251040756946c248bb66961",
    ),
    InputSpec(
        "raw/uscrn/CRND0103-202011230550.zip",
        "5351daf1947407dde3c5f5a97e462f9ec99d78f4b1a3871436ac36cb7f3e1112",
    ),
    InputSpec(
        "raw/uscrn/CRND0103-202011300550.zip",
        "994b06402263437a2ea2412b44e77e4cc64eba331449bed681ae8b087566f400",
    ),
    InputSpec(
        "input/comcat/comcat_2025-01-01_08_details.json.gz",
        "aade84e828e7f2ce4043d8a67c2dadaec89d5fac7e2be0d8362087cd6037bf73",
    ),
)


def snapshot_paths(root: Path = DATA_ROOT) -> list[Path]:
    """Return only the eight declared climate archives, in publication order."""

    return [
        root / spec.path
        for spec in EXPECTED_INPUTS
        if spec.path.startswith("raw/uscrn/")
    ]


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_inputs(
    root: Path = DATA_ROOT,
    specs: Iterable[InputSpec] = EXPECTED_INPUTS,
) -> list[str]:
    """Return actionable diagnostics for missing or altered inputs."""

    diagnostics = []
    for spec in specs:
        path = root / spec.path
        if not path.is_file():
            diagnostics.append(f"missing: {spec.path}")
        elif file_digest(path) != spec.sha256:
            diagnostics.append(f"digest mismatch: {spec.path}")
    return diagnostics


def validate_inputs(root: Path = DATA_ROOT) -> None:
    """Raise a concise error unless every expected input is present and exact."""

    diagnostics = inspect_inputs(root)
    if diagnostics:
        details = "\n  - ".join(diagnostics)
        raise RuntimeError(
            "Input validation failed. See data/README.md for acquisition "
            f"instructions.\n  - {details}"
        )


def main() -> int:
    """Command-line input validation entry point."""

    diagnostics = inspect_inputs()
    if diagnostics:
        print("Input validation failed:")
        for diagnostic in diagnostics:
            print(f"  - {diagnostic}")
        return 1
    print(f"Validated {len(EXPECTED_INPUTS)} inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
