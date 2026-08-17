"""
ssl_setup.py  (src/utils/ssl_setup.py)
--------------------------------------
Make Python's HTTPS clients trust a CORPORATE TLS-inspection proxy.

On a managed / corporate network, outbound HTTPS is intercepted by a proxy that
re-signs traffic with a corporate root CA. That CA lives in the OS trust store
(installed by IT) but Python's HTTP libs default to the `certifi` bundle, which
does NOT contain it -> `curl: (60) SSL certificate problem: unable to get local
issuer certificate` from curl_cffi (yfinance / Google Trends), and the equivalent
from requests (SEC / FRED / Wikipedia / Dataroma) and httpx (OpenAI).

`configure_corporate_ca()` builds a COMBINED bundle = certifi + the OS trust store
(via `ssl.enum_certificates` on Windows) and points the standard CA env vars at it:

    SSL_CERT_FILE, CURL_CA_BUNDLE, REQUESTS_CA_BUNDLE

which curl_cffi (its `_default_cacert()` checks exactly these three before certifi),
requests, urllib and httpx all honour. This only ADDS the roots the OS already
trusts (exactly what your browser trusts) — it does NOT disable verification.

IMPORTANT ordering: yfinance imports curl_cffi at module load and freezes its
default CA path THEN, so this must run BEFORE the first HTTP-client import — call
it as the first line of `main.py`, or set the same vars persistently so every
process inherits them:  `python -m src.utils.ssl_setup`  prints the `setx` commands.
"""
from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path

import certifi

# curl_cffi._default_cacert() checks these in order before falling back to certifi;
# requests honours REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE; ssl/urllib/httpx use SSL_CERT_FILE.
CA_ENV_VARS = ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE")
DEFAULT_BUNDLE = Path.home() / ".stock_pick_strat" / "corporate_ca_bundle.pem"


def _os_store_pem() -> list[str]:
    """PEM blocks for the OS trust store (Windows ROOT + intermediate CA stores),
    which on a managed machine includes the corporate proxy CA. Empty off Windows."""
    if sys.platform != "win32":
        return []
    pems: list[str] = []
    for store in ("ROOT", "CA"):
        try:
            certs = ssl.enum_certificates(store)
        except Exception:
            continue
        for der, _enc, _trust in certs:
            try:
                pems.append(ssl.DER_cert_to_PEM_cert(der))
            except Exception:
                continue
    return pems


def build_corporate_ca_bundle(dest: Path | None = None) -> Path:
    """Write `certifi + OS trust store` to `dest` (default: under the user home) and
    return the path. Idempotent (rewrites the file each call)."""
    dest = Path(dest) if dest else DEFAULT_BUNDLE
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts = [Path(certifi.where()).read_text(encoding="utf-8")]
    parts.extend(_os_store_pem())
    dest.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return dest


def configure_corporate_ca(dest: Path | None = None, force: bool = False) -> str | None:
    """Point the CA env vars at a combined certifi + OS-store bundle so every HTTPS
    client trusts the corporate proxy CA.

    No-op (returns the existing value) when a CA env var is ALREADY set — the user's
    own config wins — unless `force=True`. Only builds a bundle where there is an OS
    store to add (Windows); elsewhere it leaves certifi as the default. Returns the
    bundle path in effect, or None if nothing was configured.
    """
    already = next((os.environ[v] for v in CA_ENV_VARS if os.environ.get(v)), None)
    if already and not force:
        return already
    if sys.platform != "win32" and not force:
        return already
    bundle = str(build_corporate_ca_bundle(dest))
    for v in CA_ENV_VARS:
        os.environ[v] = bundle
    return bundle


if __name__ == "__main__":
    path = build_corporate_ca_bundle()
    n_os = len(_os_store_pem())
    print(f"Combined CA bundle (certifi + {n_os} OS-store certs) written to:\n  {path}\n")
    print("Make it permanent for ALL future shells, then restart your terminal:")
    for v in CA_ENV_VARS:
        print(f'  setx {v} "{path}"')
