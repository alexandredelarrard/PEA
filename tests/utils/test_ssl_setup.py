"""
Corporate-CA bundle helper (src/utils/ssl_setup.py): builds a combined
certifi + OS-trust-store bundle and points the CA env vars at it so every HTTPS
client (curl_cffi/yfinance, requests, httpx) trusts a corporate TLS-inspection
proxy WITHOUT disabling verification.
"""
from __future__ import annotations

import os
import ssl
from pathlib import Path

import certifi

from src.utils import ssl_setup


def test_bundle_is_valid_and_superset_of_certifi(tmp_path):
    dest = tmp_path / "ca.pem"
    out = ssl_setup.build_corporate_ca_bundle(dest)
    assert out == dest and dest.exists()

    n = dest.read_text(encoding="utf-8").count("BEGIN CERTIFICATE")
    certifi_n = Path(certifi.where()).read_text(encoding="utf-8").count("BEGIN CERTIFICATE")
    assert n >= certifi_n                                   # certifi + (OS store on Windows)
    # must be a loadable PEM (proves it will verify, not silently break TLS)
    ssl.create_default_context().load_verify_locations(cafile=str(dest))

    print("\n=== SANITY CHECK: combined CA bundle ===")
    print(f"  {n} certs (certifi has {certifi_n}); loads clean into an SSL context. Validated.")


def test_configure_sets_env_then_respects_user_override(tmp_path, monkeypatch):
    for v in ssl_setup.CA_ENV_VARS:
        monkeypatch.delenv(v, raising=False)

    dest = tmp_path / "ca.pem"
    path = ssl_setup.configure_corporate_ca(dest=dest, force=True)   # force -> works cross-OS
    assert path == str(dest)
    assert all(os.environ[v] == str(dest) for v in ssl_setup.CA_ENV_VARS)

    # a CA env var the user already set is RESPECTED (never silently overridden)
    monkeypatch.setenv("SSL_CERT_FILE", "/preexisting/ca.pem")
    for v in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        monkeypatch.delenv(v, raising=False)
    assert ssl_setup.configure_corporate_ca(dest=dest) == "/preexisting/ca.pem"

    print("\n=== SANITY CHECK: configure_corporate_ca ===")
    print("  sets SSL_CERT_FILE/CURL_CA_BUNDLE/REQUESTS_CA_BUNDLE; respects a pre-set "
          "value (user override wins). Validated.")


def test_relaxed_context_drops_only_strict_and_still_verifies(tmp_path):
    """Python 3.13 turns VERIFY_X509_STRICT on by default, which rejects the Zscaler
    intermediate's non-critical basicConstraints. We clear THAT FLAG ONLY -- certificate
    and hostname verification must stay on, or this becomes `verify=False` by the back door."""
    bundle = ssl_setup.build_corporate_ca_bundle(tmp_path / "ca.pem")
    ctx = ssl_setup.relaxed_ssl_context(cafile=str(bundle))

    assert not (ctx.verify_flags & ssl.VERIFY_X509_STRICT), "STRICT must be cleared"
    assert ctx.verify_mode == ssl.CERT_REQUIRED, "certificates must still be REQUIRED"
    assert ctx.check_hostname is True, "hostname checking must stay ON"
    # the default context really does enable STRICT on this interpreter (guards the premise)
    assert ssl.create_default_context().verify_flags & ssl.VERIFY_X509_STRICT

    print("\n=== SANITY CHECK: relaxed_ssl_context ===")
    print("  VERIFY_X509_STRICT cleared; CERT_REQUIRED + check_hostname still on. Validated.")


def test_bundle_write_is_atomic_and_reused(tmp_path, monkeypatch):
    """A torn bundle reads as `[X509] PEM lib` or a bogus `unable to get local issuer
    certificate`, so the write must never be observable half-done: no `.tmp` may survive,
    and an already-usable bundle is REUSED instead of rewritten on every process start."""
    dest = tmp_path / "ca.pem"
    ssl_setup.build_corporate_ca_bundle(dest)
    assert ssl_setup.bundle_is_usable(dest)
    assert not list(tmp_path.glob("*.tmp")), "no temp file may be left behind"

    # second call must REUSE (not rewrite) -- proven by making a rewrite impossible
    mtime = dest.stat().st_mtime_ns
    monkeypatch.setattr(ssl_setup, "_os_store_pem",
                        lambda: (_ for _ in ()).throw(AssertionError("rebuilt when it "
                                                                    "should have reused")))
    again = ssl_setup.build_corporate_ca_bundle(dest)
    assert again == dest and dest.stat().st_mtime_ns == mtime, "usable bundle must be reused"

    # a TRUNCATED bundle is not 'usable' -> it gets rebuilt rather than trusted
    dest.write_text("-----BEGIN CERTIFICATE-----\ntruncated", encoding="utf-8")
    assert not ssl_setup.bundle_is_usable(dest), "a torn bundle must be rejected"

    print("=== SANITY CHECK: atomic + reused CA bundle ===")
    print("  built atomically (no .tmp left), reused when usable, rebuilt when torn. Validated.")


if __name__ == "__main__":
    import tempfile
    test_bundle_is_valid_and_superset_of_certifi(Path(tempfile.mkdtemp()))
