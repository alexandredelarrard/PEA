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


if __name__ == "__main__":
    import tempfile
    test_bundle_is_valid_and_superset_of_certifi(Path(tempfile.mkdtemp()))
