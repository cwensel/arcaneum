"""Code fingerprint shared by the workstation, its workers, and deploy (RDR-025).

Unreleased trees report the last release's version string, so the version alone
cannot tell a worker whether it runs the workstation's code. The fingerprint is a
hash of the ``arcaneum`` package's Python sources and its declared requirements
(``Requires-Dist``), computed the same way from an installed package and from a
wheel, so a wheel built from a tree has that tree's fingerprint.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Iterable
from email.parser import HeaderParser
from functools import lru_cache
from importlib import metadata
from pathlib import Path

PACKAGE = "arcaneum"


def _digest(sources: Iterable[tuple[str, bytes]], requires: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for path, content in sorted(sources):
        digest.update(f"source {path} {hashlib.sha256(content).hexdigest()}\n".encode())
    for requirement in sorted({value.strip() for value in requires if value.strip()}):
        digest.update(f"requires {requirement}\n".encode())
    return digest.hexdigest()


def tree_fingerprint(package_dir: Path, requires: Iterable[str]) -> str:
    """Fingerprint a package directory such as ``src/arcaneum`` or site-packages."""
    package_dir = Path(package_dir)
    sources = (
        (f"{PACKAGE}/{path.relative_to(package_dir).as_posix()}", path.read_bytes())
        for path in package_dir.rglob("*.py")
        if "__pycache__" not in path.relative_to(package_dir).parts
    )
    return _digest(sources, requires)


def wheel_fingerprint(wheel_path: Path) -> str:
    """Fingerprint an arcaneum wheel without installing it."""
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        sources = [
            (name, wheel.read(name))
            for name in names
            if name.startswith(f"{PACKAGE}/") and name.endswith(".py")
        ]
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA") and name.count("/") == 1
        ]
        if len(metadata_names) != 1:
            raise ValueError(f"{wheel_path} is not a single-distribution wheel")
        headers = HeaderParser().parsestr(wheel.read(metadata_names[0]).decode())
    return _digest(sources, headers.get_all("Requires-Dist") or [])


def installed_requires() -> list[str]:
    """Declared requirements of the installed arcaneum distribution."""
    try:
        return list(metadata.distribution(PACKAGE).requires or [])
    except metadata.PackageNotFoundError:
        return []


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Fingerprint of the arcaneum code this process imported."""
    import arcaneum

    return tree_fingerprint(Path(arcaneum.__file__).parent, installed_requires())
