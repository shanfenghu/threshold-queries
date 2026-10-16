#!/usr/bin/env python3
"""Download the fixed US Climate Reference Network release archives."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/snapshots"
ARCHIVES = (
    "CRND0103-202010120450.zip",
    "CRND0103-202010190450.zip",
    "CRND0103-202010260450.zip",
    "CRND0103-202011020550.zip",
    "CRND0103-202011090550.zip",
    "CRND0103-202011160550.zip",
    "CRND0103-202011230550.zip",
    "CRND0103-202011300550.zip",
)


def main() -> int:
    """Download absent archives atomically; leave existing files unchanged."""

    destination = Path(__file__).resolve().parent / "raw" / "uscrn"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        target = destination / name
        if target.exists():
            print(f"present: {name}")
            continue
        temporary = target.with_suffix(".download")
        print(f"fetching: {name}")
        try:
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", temporary)
            temporary.replace(target)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            print(f"failed: {name}: {error}", file=sys.stderr)
            return 1
    print("Download complete. Run `make check-data` to verify every input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
