"""Setup verification and diagnostics command (RDR-006 enhancement)."""

import sys
import os
import importlib.util
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table

from arcaneum.cli.output import print_json, print_info, print_error
from arcaneum.cli.errors import EXIT_SUCCESS, EXIT_ERROR


def check_python_version() -> Tuple[bool, str]:
    """Check Python version meets minimum requirement."""
    min_version = (3, 12)
    current = sys.version_info[:2]

    if current >= min_version:
        return True, f"Python {current[0]}.{current[1]} (>= {min_version[0]}.{min_version[1]} required)"
    else:
        return False, f"Python {current[0]}.{current[1]} (need >= {min_version[0]}.{min_version[1]})"


def check_dependency(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a Python dependency is installed."""
    if import_name is None:
        import_name = package_name

    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            return True, f"{package_name} {version}"
        except Exception as e:
            return True, f"{package_name} (version unknown)"
    else:
        return False, f"{package_name} not installed"


def check_qdrant_connection(verbose: bool = False) -> Tuple[bool, str]:
    """Check Qdrant server connectivity."""
    try:
        from qdrant_client import QdrantClient
        from arcaneum.config import get_qdrant_config

        config = get_qdrant_config()
        client = QdrantClient(**config)

        # Try to get collections to verify connectivity
        collections = client.get_collections()
        return True, f"Qdrant connected ({len(collections.collections)} collections)"
    except ImportError:
        return False, "qdrant-client not installed"
    except Exception as e:
        if verbose:
            return False, f"Qdrant connection failed: {str(e)}"
        else:
            return False, "Qdrant connection failed (check server is running)"


def check_meilisearch_connection(verbose: bool = False) -> Tuple[bool, str]:
    """Check MeiliSearch server connectivity (optional)."""
    try:
        from meilisearch import Client
        from arcaneum.config import get_meilisearch_config

        config = get_meilisearch_config()
        client = Client(**config)

        # Try to get health to verify connectivity
        health = client.health()
        if health.get('status') == 'available':
            indexes = client.get_indexes()
            return True, f"MeiliSearch connected ({len(indexes['results'])} indexes)"
        else:
            return False, f"MeiliSearch status: {health.get('status', 'unknown')}"
    except ImportError:
        return None, "meilisearch not installed (optional)"
    except Exception as e:
        if verbose:
            return None, f"MeiliSearch connection failed: {str(e)} (optional)"
        else:
            return None, "MeiliSearch connection failed (optional, not required)"


def check_embedding_models(verbose: bool = False) -> Tuple[bool, str]:
    """Check if embedding models can be loaded."""
    try:
        from arcaneum.models.factory import EmbeddingModelFactory

        # Try to get available models
        factory = EmbeddingModelFactory()
        available = factory.get_available_models()

        if len(available) > 0:
            return True, f"Embedding models available: {', '.join(available)}"
        else:
            return False, "No embedding models configured"
    except ImportError as e:
        return False, f"Model factory import failed: {str(e)}"
    except Exception as e:
        if verbose:
            return False, f"Model check failed: {str(e)}"
        else:
            return False, "Model initialization failed"


def check_temp_dir_writable() -> Tuple[bool, str]:
    """Check if temporary directory is writable."""
    import tempfile

    try:
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, '.arcaneum_test')

        with open(test_file, 'w') as f:
            f.write('test')

        os.remove(test_file)
        return True, f"Temp directory writable: {temp_dir}"
    except Exception as e:
        return False, f"Temp directory not writable: {str(e)}"


def check_environment_vars(verbose: bool = False) -> Tuple[bool, str]:
    """Check important environment variables."""
    important_vars = [
        'QDRANT_URL',
        'QDRANT_API_KEY',
        'MEILISEARCH_URL',
        'MEILISEARCH_API_KEY',
    ]

    set_vars = []
    for var in important_vars:
        if os.getenv(var):
            set_vars.append(var)

    if verbose and len(set_vars) > 0:
        return True, f"Environment vars set: {', '.join(set_vars)}"
    elif len(set_vars) > 0:
        return True, f"{len(set_vars)} environment variable(s) configured"
    else:
        return None, "Using default configuration (no env vars set)"


def doctor_command(verbose: bool = False, output_json: bool = False):
    """Run system diagnostics and report setup status.

    Args:
        verbose: Show detailed information
        output_json: Output results in JSON format
    """
    if not output_json:
        print_info("Running Arcaneum diagnostics...")
        print()

    # Run all checks
    checks: Dict[str, Tuple[bool | None, str]] = {
        "Python Version": check_python_version(),
        "qdrant-client": check_dependency("qdrant-client", "qdrant_client"),
        "sentence-transformers": check_dependency("sentence-transformers", "sentence_transformers"),
        "PyMuPDF": check_dependency("PyMuPDF", "fitz"),
        "pytesseract": check_dependency("pytesseract"),
        "Qdrant Connection": check_qdrant_connection(verbose),
        "MeiliSearch Connection": check_meilisearch_connection(verbose),
        "Embedding Models": check_embedding_models(verbose),
        "Temp Directory": check_temp_dir_writable(),
        "Environment": check_environment_vars(verbose),
    }

    # Count results
    passed = sum(1 for status, _ in checks.values() if status is True)
    failed = sum(1 for status, _ in checks.values() if status is False)
    optional = sum(1 for status, _ in checks.values() if status is None)
    total_required = len(checks) - optional

    if output_json:
        # JSON output
        results = []
        for name, (status, message) in checks.items():
            results.append({
                "check": name,
                "status": "pass" if status is True else ("fail" if status is False else "optional"),
                "message": message
            })

        overall_status = "success" if failed == 0 else "error"
        print_json(
            overall_status,
            f"Diagnostics complete: {passed}/{total_required} required checks passed",
            data={
                "checks": results,
                "summary": {
                    "total": len(checks),
                    "passed": passed,
                    "failed": failed,
                    "optional": optional
                }
            }
        )
    else:
        # Text output with table
        console = Console()
        table = Table(title="Arcaneum System Diagnostics")

        table.add_column("Check", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")

        for name, (status, message) in checks.items():
            if status is True:
                status_icon = "✅"
                status_style = "green"
            elif status is False:
                status_icon = "❌"
                status_style = "red"
            else:  # None = optional
                status_icon = "⚠️"
                status_style = "yellow"

            table.add_row(
                name,
                f"[{status_style}]{status_icon}[/{status_style}]",
                message
            )

        console.print(table)
        print()

        # Summary
        if failed == 0:
            print_info(f"✅ All required checks passed ({passed}/{total_required})")
            if optional > 0:
                print_info(f"⚠️  {optional} optional check(s) not configured")
            return EXIT_SUCCESS
        else:
            print_error(f"❌ {failed} check(s) failed, {passed} passed", json_output=False)
            print()
            print_info("Fix the failed checks above to use Arcaneum")
            print_info("Run with --verbose for more details")
            return EXIT_ERROR

    return EXIT_SUCCESS if failed == 0 else EXIT_ERROR
