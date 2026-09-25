---
title: Peer-relative features
description: Features expressed relative to economically similar company baskets.
type: concept
tags:
  - wiki
  - concept
---
# Peer-relative features

## Definition

A peer-relative feature compares a company's characteristic with a small basket of related companies rather than the entire universe. Peer baskets are derived from business-description embedding similarity and optionally return correlation, then loaded by the cube's panel builders.

## Why it matters

Global cross-sectional ranks can mix structurally different businesses, while sector labels can be too coarse. A peer basket aims to compare a company with firms that are both economically and statistically similar. Dual-class tickers are removed so the same business does not become its own strongest peer.

Peer-relative values remain point-in-time only if both the underlying feature and the trading-date panel are point-in-time safe.

## Where it lives

- Peer orchestration: [step_deduce_peers.py](../../src/data_peers/step_deduce_peers.py)
- Basket construction: [sector_peers.py](../../src/data_peers/utils/sector_peers.py)
- Embedding cache: [data_peers/utils/embeddings.py](../../src/data_peers/utils/embeddings.py)
- Panel construction: [data_aggregate/utils/common/panel.py](../../src/data_aggregate/utils/common/panel.py)
- Configuration: [configs/peers.yml](../../configs/peers.yml)

## Related

- [Peer deduction](../modules/data-peers.md)
- [Cube aggregation](../modules/data-aggregate.md)
- [Point-in-time data](./point-in-time-data.md)
