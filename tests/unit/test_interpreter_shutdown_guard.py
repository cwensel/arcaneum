"""The shutdown guard must forward pytest's real exit status, never mask it.

Co-loading PyMuPDF's SWIG types with PyTorch segfaults CPython's `finalize_modules`
after every test has already run and reported, turning a green suite into exit 139.
`tests/conftest.py` contains the containment; these tests pin its contract, because
a guard that calls `os._exit` is exactly the kind of code that can silently turn a
failing suite green.

Each case runs pytest in a subprocess so the real hook executes.
"""

import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def _run_pytest(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Run pytest over a generated test file, with the project's conftest active."""
    (tmp_path / "conftest.py").write_text(CONFTEST.read_text())
    (tmp_path / "test_generated.py").write_text(body)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "test_generated.py", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_passing_run_exits_zero(tmp_path):
    result = _run_pytest(tmp_path, "def test_ok():\n    assert True\n")

    assert result.returncode == 0
    assert "1 passed" in result.stdout


def test_failing_run_still_fails(tmp_path):
    """The guard must not turn red into green - the whole point of pinning it."""
    result = _run_pytest(tmp_path, "def test_bad():\n    assert False\n")

    assert result.returncode == 1
    assert "1 failed" in result.stdout


def test_summary_is_printed_before_the_guard_exits(tmp_path):
    """Exiting at sessionfinish would truncate pytest's own reporting."""
    result = _run_pytest(
        tmp_path,
        "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n",
    )

    assert result.returncode == 0
    assert "2 passed" in result.stdout


def test_collection_errors_are_forwarded(tmp_path):
    result = _run_pytest(tmp_path, "import a_module_that_does_not_exist\n")

    assert result.returncode != 0


@pytest.mark.parametrize("report", ["term-missing", "xml:cov.xml"])
def test_coverage_reports_survive_the_guard(tmp_path, report):
    """pytest-cov writes during unconfigure, so the guard must run after it."""
    (tmp_path / "conftest.py").write_text(CONFTEST.read_text())
    (tmp_path / "test_generated.py").write_text("def test_ok():\n    assert True\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_generated.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--cov=test_generated",
            f"--cov-report={report}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0
    if report.startswith("xml"):
        assert (tmp_path / "cov.xml").exists()
    else:
        assert "test_generated" in result.stdout
