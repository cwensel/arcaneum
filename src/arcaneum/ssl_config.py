"""SSL configuration utilities for corporate proxy/VPN environments.

SSL verification is DISABLED by default for compatibility with corporate
proxies that use self-signed certificates.

Usage:
    # Default: SSL verification disabled (works with corporate proxies)
    arc search semantic "query" --collection MyCollection

    # To enable strict SSL verification:
    export ARC_SSL_VERIFY=true
    arc search semantic "query" --collection MyCollection

Security Note:
    SSL verification is disabled by default to support corporate environments.
    Set ARC_SSL_VERIFY=true if you need strict certificate validation.
"""

import os
import warnings
import ssl
import logging

logger = logging.getLogger(__name__)

# Suppress noisy warnings from third-party libraries
warnings.filterwarnings("ignore", message=".*optimum is not installed.*")

_SSL_DISABLED = False


def is_ssl_verification_disabled() -> bool:
    """Check if SSL verification is disabled.

    Returns:
        True if SSL verification has been disabled, False otherwise.
    """
    return _SSL_DISABLED


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

    SSL verification is DISABLED by default for compatibility with corporate
    proxies. Set ARC_SSL_VERIFY=true to enable strict SSL verification.

    Returns:
        True if SSL verification was disabled, False otherwise.

    Example:
        # Default: SSL verification disabled (works with corporate proxies)
        arc search semantic "query" --collection MyCollection

        # To enable SSL verification:
        export ARC_SSL_VERIFY=true
        arc search semantic "query" --collection MyCollection
    """
    ssl_verify = os.environ.get("ARC_SSL_VERIFY", "false").lower()

    if ssl_verify in ("true", "1", "yes", "on"):
        # User explicitly wants SSL verification enabled
        return False

    # Default: disable SSL verification for corporate proxy compatibility
    return disable_ssl_verification(quiet=True)


def check_and_configure_ssl() -> dict:
    """Check and configure SSL, returning status information.

    This is the recommended entry point for CLI commands. It checks the
    environment variable and returns information about the SSL configuration.

    SSL verification is DISABLED by default for corporate proxy compatibility.

    Returns:
        Dictionary with:
            - ssl_verify: bool - Whether SSL verification is enabled
            - configured_by: str - How SSL was configured ('environment', 'default')
            - warning: str or None - Warning message if SSL is disabled
    """
    ssl_verify_env = os.environ.get("ARC_SSL_VERIFY", "false").lower()

    if ssl_verify_env in ("true", "1", "yes", "on"):
        return {
            "ssl_verify": True,
            "configured_by": "environment",
            "warning": None
        }

    # Default: disable SSL verification
    disable_ssl_verification(quiet=True)
    return {
        "ssl_verify": False,
        "configured_by": "default",
        "warning": None  # No warning since this is the default
    }
