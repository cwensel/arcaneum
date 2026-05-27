"""SSL configuration utilities for corporate proxy/VPN environments.

SSL verification is enabled by default. Corporate proxy environments with
self-signed certificates can explicitly opt out with ARC_SSL_VERIFY=false.

Usage:
    # Default: strict SSL verification
    arc search semantic "query" --collection MyCollection

    # To disable SSL verification for a trusted corporate proxy:
    export ARC_SSL_VERIFY=false
    arc search semantic "query" --collection MyCollection

Security Warning:
    When SSL verification is disabled, this module globally monkey-patches
    `requests.Session.request`, `httpx.Client.__init__`, and
    `httpx.AsyncClient.__init__` so that ALL HTTP connections made by the
    process skip certificate verification — including third-party libraries
    unrelated to Qdrant/MeiliSearch. This creates a broad MITM attack surface
    and should only be used on trusted networks (corporate VPNs, controlled
    dev environments). To avoid this, callers should not import this module
    from library code; disable_ssl_verification() should be invoked only from
    the CLI entry point. Strict validation is the default; only set
    ARC_SSL_VERIFY=false on trusted networks.
"""

import os
import warnings
import ssl
import logging

logger = logging.getLogger(__name__)

# Suppress noisy warnings from third-party libraries
warnings.filterwarnings("ignore", message=".*optimum is not installed.*")

_SSL_DISABLED = False


def disable_ssl_verification(quiet: bool = False) -> bool:
    """Disable SSL certificate verification globally.

    Sets environment variables and monkey-patches libraries to disable SSL
    verification. Works even if libraries are already imported.

    Args:
        quiet: If True, suppress the warning message about disabled SSL.

    Returns:
        True if SSL verification was disabled, False if already disabled.

    Security Warning:
        Only use on trusted networks with self-signed certificates (e.g., corporate VPNs).
    """
    global _SSL_DISABLED

    if _SSL_DISABLED:
        return False

    # === PHASE 1: Environment variables (for libraries not yet imported) ===
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""

    # HuggingFace Hub specific settings
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

    # === PHASE 2: Python SSL module patch ===
    # This affects all Python SSL connections including urllib3 and httpx
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    # === PHASE 3: Suppress warnings ===
    warnings.filterwarnings("ignore", message=".*Unverified HTTPS request.*")
    warnings.filterwarnings("ignore", category=Warning, module="urllib3")

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except (ImportError, AttributeError):
        pass

    try:
        import requests
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    except (ImportError, AttributeError):
        pass

    # === PHASE 4: Monkey-patch requests library (for already-imported sessions) ===
    try:
        import requests
        _original_request = requests.Session.request

        def _patched_request(self, method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return _original_request(self, method, url, **kwargs)

        requests.Session.request = _patched_request
    except (ImportError, AttributeError):
        pass

    # === PHASE 5: Patch huggingface_hub if already imported ===
    try:
        import huggingface_hub
        # HF Hub uses its own HTTP backend, try to patch it
        if hasattr(huggingface_hub, "constants"):
            huggingface_hub.constants.HF_HUB_DISABLE_SSL_VERIFICATION = True
        # Also try the newer configuration approach
        try:
            from huggingface_hub import configure_http_backend
            import requests
            session = requests.Session()
            session.verify = False
            configure_http_backend(backend_factory=lambda: session)
        except (ImportError, AttributeError, TypeError):
            pass
    except ImportError:
        pass

    # === PHASE 6: Patch httpx if present (used by some HF libraries) ===
    try:
        import httpx
        _original_httpx_client_init = httpx.Client.__init__

        def _patched_httpx_client_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            return _original_httpx_client_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_httpx_client_init

        _original_httpx_async_init = httpx.AsyncClient.__init__

        def _patched_httpx_async_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            return _original_httpx_async_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = _patched_httpx_async_init
    except (ImportError, AttributeError):
        pass

    _SSL_DISABLED = True

    if not quiet:
        logger.warning("SSL certificate verification disabled. Use only on trusted networks.")

    return True


def configure_ssl_from_env() -> bool:
    """Configure SSL based on ARC_SSL_VERIFY environment variable.

    SSL verification is enabled by default. Set ARC_SSL_VERIFY=false to opt
    into the global corporate-proxy bypass.

    Returns:
        True if SSL verification was disabled, False otherwise.

    Example:
        # Default: strict SSL verification
        arc search semantic "query" --collection MyCollection

        # To disable SSL verification for a trusted corporate proxy:
        export ARC_SSL_VERIFY=false
        arc search semantic "query" --collection MyCollection
    """
    ssl_verify = os.environ.get("ARC_SSL_VERIFY")

    if ssl_verify is None:
        return False

    if ssl_verify.lower() in ("false", "0", "no", "off"):
        return disable_ssl_verification(quiet=True)

    return False
