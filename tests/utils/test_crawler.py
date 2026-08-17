"""
Rolling-fingerprint crawler with authorized-proxy rotation (src/utils/crawler.py).

Verifies: on a detected block (429) it ROTATES to the next configured proxy and retries fast to a
200; a terminal 404 returns immediately (no retry); rolling headers carry a real-browser UA; proxy
credentials are masked in logs; and it works direct (no proxies). The transport is mocked — no
network, no proxies, no real IPs. This tool rotates only the operator's OWN authorized proxies; it
never acquires anonymous proxy pools.
"""
from __future__ import annotations

import pytest

from src.utils import crawler as cw
from src.utils import polite_http as ph


class _Resp:
    def __init__(self, status, text="", headers=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}

    def json(self):
        import json
        return json.loads(self.text)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(cw.time, "sleep", lambda *_: None)   # keep the fast-retry test instant


def test_rotates_ip_on_block_then_succeeds(monkeypatch):
    c = cw.Crawler(retries=3, backoff=0.0, proxies=["http://a:1", "http://b:2"])
    seq = iter([429, 200])                                    # blocked on IP #1, ok on IP #2
    used, uas = [], []

    def fake(url, params, headers, proxy):
        used.append(proxy)
        uas.append(headers.get("User-Agent"))
        return _Resp(next(seq), "OK")

    monkeypatch.setattr(c, "_raw_get", fake)
    out = c.get_text("https://example.com/data")

    assert out == "OK"
    assert len(used) == 2, "should have retried exactly once after the block"
    assert used[0] != used[1], "must MOVE to a different proxy after detection"
    assert all(ua in ph._UA_POOL for ua in uas), "rolling headers must carry a real-browser UA"

    print("\n=== SANITY CHECK: rotate IP on block ===")
    print(f"  request 1 via {cw._mask(used[0])} -> 429 (detected); rotated -> {cw._mask(used[1])} -> 200")
    print(f"  rolling UAs distinct-capable: {len(set(uas))} seen across 2 calls")
    print("  CONCLUSION: detection -> immediate IP rotation over the CONFIGURED (authorized) pool + "
          "fast retry to success. Validated.")


def test_terminal_404_no_retry(monkeypatch):
    c = cw.Crawler(retries=5, proxies=["http://a:1"])
    calls = []

    def fake(url, params, headers, proxy):
        calls.append(proxy)
        return _Resp(404, "not found")

    monkeypatch.setattr(c, "_raw_get", fake)
    assert c.get_text("https://example.com/missing") is None
    assert len(calls) == 1, "a 404 must NOT be retried / rotated"
    print("\n=== SANITY CHECK: terminal 404 ===")
    print(f"  404 -> {len(calls)} request, no retry/rotation. Validated.")


def test_direct_when_no_proxies(monkeypatch):
    c = cw.Crawler(retries=2, proxies=[])
    assert c.n_proxies == 0
    seen = []

    def fake(url, params, headers, proxy):
        seen.append(proxy)
        return _Resp(200, "OK")

    monkeypatch.setattr(c, "_raw_get", fake)
    assert c.get_text("https://example.com") == "OK"
    assert seen == [None], "no proxies -> direct connection (proxy=None)"
    print("\n=== SANITY CHECK: no proxies -> direct ===")
    print("  with no PEA_SCRAPE_PROXIES the crawler goes direct (can't rotate IPs you don't have). Validated.")


def test_gives_up_after_retries(monkeypatch):
    c = cw.Crawler(retries=2, backoff=0.0, proxies=["http://a:1", "http://b:2"])
    used = []

    def fake(url, params, headers, proxy):
        used.append(proxy)
        return _Resp(429, "blocked")

    monkeypatch.setattr(c, "_raw_get", fake)
    assert c.get_text("https://example.com") is None
    assert len(used) == 3, "retries=2 -> 1 initial + 2 retries"
    print("\n=== SANITY CHECK: exhaust retries ===")
    print(f"  persistent 429 across {len(used)} attempts -> None (asks for more authorized proxies). Validated.")


def test_mask_strips_credentials():
    assert cw._mask("http://user:secret@host.example.com:8080") == "http://host.example.com:8080"
    assert cw._mask(None) == "direct"
    print("\n=== SANITY CHECK: proxy credential masking ===")
    print("  user:pass stripped from proxy URL before logging. Validated.")


def test_load_proxy_pool_from_env(monkeypatch):
    monkeypatch.setenv("PEA_SCRAPE_PROXIES", "http://a:1 , http://b:2 ,, http://c:3")
    assert cw.load_proxy_pool() == ["http://a:1", "http://b:2", "http://c:3"]
    monkeypatch.delenv("PEA_SCRAPE_PROXIES", raising=False)
    print("\n=== SANITY CHECK: proxy pool env parse ===")
    print("  PEA_SCRAPE_PROXIES parsed to 3 authorized proxies (blanks dropped). Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
