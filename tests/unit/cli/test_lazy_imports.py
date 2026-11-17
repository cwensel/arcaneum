"""Start-time guard: the CLI imports neither qdrant_client nor fastembed (RDR-026 Q11).

Those two libraries cost most of `import arcaneum.cli.main`. Every `arc`
command and the `arc mcp` server start through that import, and the server must
answer `initialize` within its start budget, so the modules on the start path
import them only inside the functions that need them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"

HEAVY = ("qdrant_client", "fastembed")


def _modules_after(statement: str) -> dict[str, bool]:
    # A fresh interpreter: this test process has long since imported both.
    probe = (
        f"{statement}\n"
        "import json, sys\n"
        f"print(json.dumps({{name: name in sys.modules for name in {HEAVY!r}}}))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_cli_main_import_defers_qdrant_client_and_fastembed():
    loaded = _modules_after("import arcaneum.cli.main")
    assert loaded == {name: False for name in HEAVY}


def test_mcp_server_import_defers_qdrant_client_and_fastembed():
    loaded = _modules_after("import arcaneum.mcp_server")
    assert loaded == {name: False for name in HEAVY}
