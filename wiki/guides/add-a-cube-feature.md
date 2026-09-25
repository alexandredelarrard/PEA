---
title: Add a cube feature
description: Place a feature in the correct part, preserve point-in-time and incremental contracts, and validate its economics.
type: guide
tags:
  - wiki
  - guide
---
# Add a cube feature

## Goal

Add a modelling feature without leaking future data, breaking the part schema, colliding with an existing name, or making incremental builds disagree with full builds.

## Steps

1. Identify the owning domain part in [parts.py](../../src/data_aggregate/utils/common/parts.py) and read its transformer under [data_aggregate/transformers](../../src/data_aggregate/transformers/).
2. Confirm the source table, publication date, projection, and [source availability](../concepts/source-availability.md).
3. Implement the narrow calculation in the appropriate domain helper under [data_aggregate/utils](../../src/data_aggregate/utils/).
4. Return a date-by-ticker panel and add it through `PanelMerger` so duplicate feature names fail loudly.
5. Apply raw, cross-sectional, peer-relative, or self-history variants through the shared helpers rather than reimplementing them.
6. Add tunable windows or thresholds to [configs/build_cube.yml](../../configs/build_cube.yml); stable facts belong in constants.
7. Declare any binding daily look-back on the part and raise its warm-up if necessary.
8. Update the relevant feature catalogue generator under [scripts](../../scripts/).
9. Add known-truth mathematical coverage where needed and a bounded real-data economic test with a printed conclusion.
10. Run the part-registry, projection, incremental-equivalence, catalogue, and targeted feature tests.
11. Build the affected part and finish with read-only [validation](../modules/validation.md).

## Relevant code

- Part registry: [parts.py](../../src/data_aggregate/utils/common/parts.py)
- Incremental lifecycle: [incremental.py](../../src/data_aggregate/utils/common/incremental.py)
- Cross-sectional transforms: [xs.py](../../src/data_aggregate/utils/common/xs.py)
- Peer panels: [panel.py](../../src/data_aggregate/utils/common/panel.py)
- Collision checks: [panel_merge.py](../../src/data_aggregate/utils/common/panel_merge.py)
- Regression guard: [test_aggregate_regression.py](../../tests/data_aggregate/test_aggregate_regression.py)

## Gotchas

A source with quarterly or filing grain can still produce a feature with a long daily look-back. A new column changes an unmanaged part schema and therefore forces a full rebuild. Do not regenerate the aggregate fingerprint baseline to hide an unexplained numeric change.

## Related

- [Cube aggregation](../modules/data-aggregate.md)
- [Cube parts](../concepts/cube-parts.md)
- [Cube build](../flows/cube-build.md)
- [Validate a change](./validate-a-change.md)
