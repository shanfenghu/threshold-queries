# Public inputs

The expected SHA-256 digests live in `threshold_queries.inputs`; run
`make check-data` before a full evaluation.

## Climate releases

The workflow uses eight archived daily releases from the
[US Climate Reference Network snapshot service](https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/snapshots/).
The first four releases form the development cohort and the last four form a
temporal holdout. Retrieve the exact archives with:

```bash
python data/fetch_uscrn.py
```

The script writes only to `data/raw/uscrn/`, leaves existing files unchanged,
and removes incomplete downloads. The digest check detects missing, altered, or
newly replaced archives before evaluation.

## Earthquake histories

`data/input/comcat/` contains a compressed detail response for events first
reported between 1 and 8 January 2025 with magnitude at least 4.5. It was
collected from the [USGS FDSN Event Web Service](https://earthquake.usgs.gov/fdsnws/event/1/query)
with superseded products included. The workflow groups versions by authoritative
network and event code, then reconstructs magnitude, depth, and status histories.

The workflow validates all input digests first, then checks release order,
identifiers, and value types during evaluation.
