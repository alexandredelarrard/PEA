"""
Shared anti-429 HTTP toolkit (src/utils/polite_http.py) used by the earnings-call, Wikipedia
(and, for the proxy resolver, Google Trends) extractors. Tests the transport-agnostic policy:
retry + backoff honouring Retry-After, a PER-HOST run-wide slowdown ratcheted on 429 (so one
host's throttle can't slow another), get_text/get_json, and the BYO-proxy env resolver.
"""
from __future__ import annotations

from src.utils import polite_http as ph


class _Resp:
    def __init__(self, code, text="", headers=None, js=None):
        self.status_code, self.text, self.headers, self._js = code, text, headers or {}, js

    def json(self):
        if self._js is None:
            raise ValueError("no json")
        return self._js


def test_http_get_honours_retry_after_and_ratchets_host_pace(monkeypatch):
    ph._PACE.clear()
    seq = [_Resp(429, headers={"Retry-After": "2"}), _Resp(200, "OK")]
    monkeypatch.setattr(ph, "_raw_get", lambda url, **k: seq.pop(0))
    waits = []
    monkeypatch.setattr(ph.time, "sleep", lambda s: waits.append(s))

    r = ph.http_get("https://www.fool.com/x", retries=3, backoff=1.0)
    assert r is not None and r.status_code == 200 and r.text == "OK", "should retry past 429"
    assert waits and waits[0] >= 2.0, f"must honour Retry-After=2s (waited {waits})"
    assert ph.pace_mult("https://www.fool.com/y") > 1.0, "429 must ratchet the host's pace"

    print("\n=== SANITY CHECK: polite_http 429 handling ===")
    print(f"  429 -> honoured Retry-After ({waits[0]:.1f}s), ratcheted fool.com pace to "
          f"x{ph.pace_mult('https://www.fool.com/y'):.1f}, then 200 on retry.")


def test_pace_is_per_host(monkeypatch):
    ph._PACE.clear()
    ph.note_throttle("https://api-a.com/x")
    ph.note_throttle("https://api-a.com/y")
    assert ph.pace_mult("https://api-a.com/z") > 1.0, "throttled host should be slowed"
    assert ph.pace_mult("https://api-b.com/z") == 1.0, "a different host must NOT be slowed"
    print("  per-host isolation: api-a slowed to "
          f"x{ph.pace_mult('https://api-a.com/z'):.1f}, api-b still x1.0 (Google's throttle "
          "won't slow Wikimedia). Validated.")


def test_get_text_json_and_terminal_none(monkeypatch):
    ph._PACE.clear()
    monkeypatch.setattr(ph.time, "sleep", lambda s: None)
    monkeypatch.setattr(ph, "_raw_get", lambda url, **k: _Resp(200, "hi", js={"a": 1}))
    assert ph.get_text("https://x") == "hi"
    assert ph.get_json("https://x") == {"a": 1}
    # a persistent 404 -> None (not retried); a transport error -> None
    monkeypatch.setattr(ph, "_raw_get", lambda url, **k: _Resp(404))
    assert ph.http_get("https://x", retries=2, log_missing=False) is None
    monkeypatch.setattr(ph, "_raw_get", lambda url, **k: None)
    assert ph.http_get("https://x", retries=1) is None
    print("  get_text/get_json OK; 404 and transport-error both -> None. Validated.")


def test_resolve_proxy_env(monkeypatch):
    for k in ("PEA_SCRAPE_PROXY", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(k, raising=False)
    assert ph.resolve_proxy() is None
    monkeypatch.setenv("PEA_SCRAPE_PROXY", "http://corp-proxy:8080")
    assert ph.resolve_proxy() == {"http": "http://corp-proxy:8080", "https": "http://corp-proxy:8080"}
    print("  resolve_proxy: None without env; {'http','https'} from PEA_SCRAPE_PROXY. "
          "BYO-proxy shared by all extractors. Validated.")


if __name__ == "__main__":
    import types
    mp = types.SimpleNamespace(setattr=lambda o, n, v: setattr(o, n, v),
                               setenv=lambda k, v: __import__("os").environ.__setitem__(k, v),
                               delenv=lambda k, raising=True: __import__("os").environ.pop(k, None))
    test_pace_is_per_host(mp)
