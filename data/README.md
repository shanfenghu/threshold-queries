# Public inputs

The evaluation expects two local directories under `data/raw/`.

`uscrn/` contains eight archived US Climate Reference Network daily releases
from October and November 2020. The workflow reads the release archives without
unpacking them and reconstructs every changed station-day-field episode.

`comcat/` contains a compressed US Geological Survey ComCat detail response for
events first reported on 1 January 2025. The workflow groups product versions by
authoritative network and event code, then reconstructs magnitude, depth, and
status histories.

These inputs are excluded from version control because they are redistributed
by their source agencies and can be downloaded independently. The workflow
validates release order, identifiers, and value types before analysis.
