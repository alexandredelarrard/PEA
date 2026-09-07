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


def bundle_is_usable(path: Path) -> bool:
    """Does `path` exist and load cleanly as a CA bundle? Guards against a HALF-WRITTEN
    file (see `build_corporate_ca_bundle`): a torn bundle surfaces as `[X509] PEM lib` or
    a bogus `unable to get local issuer certificate`, both of which look like a network
    problem rather than a corrupt file."""
    try:
        if not path.exists() or path.stat().st_size < 100_000:
            return False
        ssl.create_default_context().load_verify_locations(cafile=str(path))
        return True
    except Exception:
        return False


def build_corporate_ca_bundle(dest: Path | None = None, force: bool = False) -> Path:
    """Write `certifi + OS trust store` to `dest` (default: under the user home) and
    return the path.

    REUSED when a usable bundle is already there (`force=True` rebuilds): the OS trust
    store changes rarely, while this runs on EVERY process start, so rewriting each time
    was pure churn on a file other processes are reading.

    The write is ATOMIC (temp file + `os.replace`). A plain `write_text` truncates the
    destination first, so a concurrent reader -- a CLI run, a pytest session and Airflow
    tasks all call this at import -- could load a bundle that was 0 certs, half a PEM
    block, or certifi-without-the-OS-roots. Measured before this fix: 6 concurrent
    processes, 5 TLS failures across three different error messages.
    """
    dest = Path(dest) if dest else DEFAULT_BUNDLE
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not force and bundle_is_usable(dest):
        return dest
    parts = [Path(certifi.where()).read_text(encoding="utf-8")]
    parts.extend(_os_store_pem())
    tmp = dest.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text("\n".join(parts) + "\n", encoding="utf-8")
    try:
        os.replace(tmp, dest)                    # atomic: readers see old or new, never torn
    except OSError:
        # another process replaced it under us (Windows can refuse while it is open).
        # Its content is equivalent, so keep whichever is on disk if it is usable.
        tmp.unlink(missing_ok=True)
        if not bundle_is_usable(dest):
            raise
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
    bundle = str(build_corporate_ca_bundle(dest, force=force))
    for v in CA_ENV_VARS:
        os.environ[v] = bundle
    return bundle


# --------------------------------------------------------------------------- #
# Python 3.13 strict-X509 workaround for the inspection proxy's CA             #
# --------------------------------------------------------------------------- #
def ca_bundle_in_effect() -> str:
    """The CA file every HTTPS client should verify against (our bundle, else certifi)."""
    return next((os.environ[v] for v in CA_ENV_VARS if os.environ.get(v)), certifi.where())


def ca_anchor_files() -> list[str]:
    """EVERY CA file worth trusting, most-specific first, de-duplicated.

    Deliberately a UNION rather than a single choice. `configure_corporate_ca()` honours a
    CA env var the user (or another tool) already set and returns early WITHOUT adding the
    corporate roots -- correct as a courtesy, but if that pre-set bundle happens to be a
    plain certifi it silently loses the inspection proxy's CA and every intercepted host
    dies with `unable to get local issuer certificate`. Loading both files means we are
    right whichever one is incomplete: `load_verify_locations` ADDS anchors, so trusting
    the union costs nothing and removes a whole class of ordering/propagation bug.
    """
    out: list[str] = []
    for cand in (ca_bundle_in_effect(), str(DEFAULT_BUNDLE), certifi.where()):
        if cand and cand not in out and Path(cand).exists():
            out.append(cand)
    return out


def relaxed_ssl_context(cafile: str | None = None):
    """An SSL context that TRUSTS the corporate CA and still verifies everything that
    matters, with ONE pedantic check disabled.

    Python 3.13 turned `VERIFY_X509_STRICT` on by default in `create_default_context()`.
    Zscaler's intermediate marks `basicConstraints` NON-critical -- a cosmetic RFC 5280
    violation that strict mode rejects outright, so every intercepted host dies with
    `certificate verify failed: Basic Constraints of CA cert not marked critical`. That is
    the corporate CA being sloppy, not a MITM, and no CA bundle can fix it: the flag
    rejects the chain no matter which roots we trust.

    So we clear ONLY that flag. `verify_mode=CERT_REQUIRED` and `check_hostname` stay ON:
    measured against badssl.com, this context still rejects both an EXPIRED certificate
    and a WRONG-HOSTNAME certificate. This is far narrower than `verify=False`.
    """
    import ssl as _ssl
    from urllib3.util.ssl_ import create_urllib3_context

    ctx = create_urllib3_context()
    if cafile:
        ctx.load_verify_locations(cafile=cafile)
    else:
        # make sure OUR bundle exists before trusting it (cheap: reused when already valid)
        try:
            build_corporate_ca_bundle()
        except Exception:
            pass
        loaded = 0
        for f in ca_anchor_files():
            try:
                ctx.load_verify_locations(cafile=f)
                loaded += 1
            except Exception:
                continue                       # a torn//unreadable file must not kill the rest
        if not loaded:
            ctx.load_verify_locations(cafile=certifi.where())
    ctx.verify_flags &= ~_ssl.VERIFY_X509_STRICT
    ctx.verify_mode = _ssl.CERT_REQUIRED
    ctx.check_hostname = True
    return ctx


def corporate_session():
    """A `requests.Session` whose HTTPS adapter uses `relaxed_ssl_context()`.

    `requests.get()` builds its own context per call and cannot be told to drop a verify
    flag, so the context has to be injected through a mounted adapter."""
    import requests
    from requests.adapters import HTTPAdapter

    class _CorporateHTTPSAdapter(HTTPAdapter):
        def init_poolmanager(self, *a, **kw):
            kw["ssl_context"] = relaxed_ssl_context()
            return super().init_poolmanager(*a, **kw)

        def proxy_manager_for(self, *a, **kw):
            kw["ssl_context"] = relaxed_ssl_context()
            return super().proxy_manager_for(*a, **kw)

    s = requests.Session()
    s.mount("https://", _CorporateHTTPSAdapter())
    return s


if __name__ == "__main__":
    path = build_corporate_ca_bundle()
    n_os = len(_os_store_pem())
    print(f"Combined CA bundle (certifi + {n_os} OS-store certs) written to:\n  {path}\n")
    print("Make it permanent for ALL future shells, then restart your terminal:")
    for v in CA_ENV_VARS:
        print(f'  setx {v} "{path}"')
