---
title: Peer deduction
description: Hybrid business-similarity and return-correlation peer baskets persisted for cube construction.
type: module
tags:
  - wiki
  - module
---
# Peer deduction

## Summary

`src/data_peers` creates the peer basket used by peer-relative feature engineering. It combines return information with business-description embeddings, removes redundant share classes, and persists a dictionary that later cube builders load rather than recompute.

## Responsibilities

- Load the investable universe and market-adjusted return history.
- Fetch and cache business descriptions and embeddings only for missing tickers.
- Build correlation-only or hybrid similarity matrices according to configuration.
- Select, weight, persist, and reload a fixed-size peer basket per ticker.

## Public API / entry points

- `StepDeducePeers.run()` in [step_deduce_peers.py](../../src/data_peers/step_deduce_peers.py).
- The `deduce-peers` command in [data_peers/cli.py](../../src/data_peers/cli.py).
- `build_peer_dict_hybrid()`, `save_peer_dict()`, and `load_peer_dict()` in [sector_peers.py](../../src/data_peers/utils/sector_peers.py).

## Key files

- [src/data_peers/step_deduce_peers.py](../../src/data_peers/step_deduce_peers.py) coordinates the calculation.
- [src/data_peers/utils/sector_peers.py](../../src/data_peers/utils/sector_peers.py) owns similarity, dual-class filtering, basket construction, and persistence.
- [src/data_peers/utils/embeddings.py](../../src/data_peers/utils/embeddings.py) caches descriptions and embeddings.
- [configs/peers.yml](../../configs/peers.yml) selects basket size, observations, weighting, and correlation/embedding blend.

## Dependencies

The module reads prices and the universe through [DataStore](./data-store.md), uses shared market-series utilities, and consumes the embedding client exposed by [GPT extraction](./gpt-extract.md).

## Participates in

Peer output is built before aggregation in the [nightly data refresh](../flows/nightly-data-refresh.md) and consumed by [cube aggregation](./data-aggregate.md) to create [peer-relative features](../concepts/peer-relative-features.md).

## Related

- [Feature-to-portfolio architecture](../architecture/feature-to-portfolio.md)
- [Cube build](../flows/cube-build.md)
- [Peer configuration](./constants-and-configuration.md)
