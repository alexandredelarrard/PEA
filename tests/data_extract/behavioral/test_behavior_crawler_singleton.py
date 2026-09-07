"""
The shared Motley Fool crawler accessor (utils_behavior._crawler / _get).

Every fool request -- quote-page discovery, the index crawl and each transcript download --
reaches the network through `_get` -> `_crawler()`. The rest of the fool test suite
monkeypatches `_get` itself, so a defect INSIDE `_crawler()` is invisible to it: a missing
`global _CRAWLER` made the lazy-init assignment shadow the module global, so `_crawler()`
raised UnboundLocalError on every call and silently killed the whole last-resort source for
the recent-quarter gap. These tests call it for real (with the Crawler class stubbed) so the
accessor itself is covered.
"""
from __future__ import annotations

import pytest

from src.data_extract.utils.behavioral import utils_behavior as ub


class _StubCrawler:
    made = 0

    def __init__(self, **kw):
        _StubCrawler.made += 1
        self.kw = kw

    def get_text(self, url, log_missing=True):
        return f"<html>{url}</html>"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(ub, "Crawler", _StubCrawler)
    monkeypatch.setattr(ub, "_CRAWLER", None)
    _StubCrawler.made = 0
    yield


def test_crawler_accessor_returns_a_crawler_and_does_not_raise():
    c = ub._crawler()                       # UnboundLocalError before the `global` fix
    assert isinstance(c, _StubCrawler)
    print("\n=== SANITY CHECK: _crawler() accessor ===")
    print(f"  returned a live crawler ({type(c).__name__}) instead of raising "
          "UnboundLocalError. Validated.")


def test_crawler_is_built_once_and_reused():
    """It must be a SINGLETON: a per-request crawler would throw away the rolling
    fingerprint/proxy state the fool crawl depends on."""
    first, second, third = ub._crawler(), ub._crawler(), ub._crawler()
    assert first is second is third, "the crawler must be cached in the module global"
    assert _StubCrawler.made == 1, f"built {_StubCrawler.made} times, expected exactly 1"
    assert ub._CRAWLER is first, "the module global must actually be populated"
    print(f"  3 calls -> 1 construction, same object reused, module global set. Validated.")


def test_get_reaches_the_network_through_the_shared_crawler():
    """`_get` is the single HTTP door for the fool path -- prove it opens."""
    html = ub._get("https://www.fool.com/quote/nyse/ed/", log_missing=False)
    assert html == "<html>https://www.fool.com/quote/nyse/ed/</html>"
    assert _StubCrawler.made == 1
    print("  _get() returned HTML via the shared crawler (the fool fallback is reachable). "
          "Validated.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
