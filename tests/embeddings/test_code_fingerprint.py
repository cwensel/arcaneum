"""Code fingerprint of the arcaneum package (RDR-025)."""

import zipfile
from importlib import metadata
from pathlib import Path

import arcaneum
from arcaneum.embeddings.remote.fingerprint import (
    code_fingerprint,
    tree_fingerprint,
    wheel_fingerprint,
)

REQUIRES = ["click>=8.3.0", "numpy>=1.26.0", 'torch>=2.12.0; extra == "sentence-transformers"']


def make_tree(root: Path) -> Path:
    package = root / "arcaneum"
    (package / "embeddings").mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.10.0"\n')
    (package / "embeddings" / "__init__.py").write_text("")
    (package / "embeddings" / "client.py").write_text("VALUE = 1\n")
    # Namespace packages (no __init__.py) are packaged too.
    (package / "utils").mkdir()
    (package / "utils" / "memory.py").write_text("LIMIT = 2\n")
    return package


def make_wheel(package: Path, path: Path, requires: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as wheel:
        for source in sorted(package.rglob("*.py")):
            wheel.write(source, f"arcaneum/{source.relative_to(package).as_posix()}")
        wheel.writestr("arcaneum/embeddings/capabilities-v1.json", "{}")
        lines = ["Metadata-Version: 2.4", "Name: arcaneum", "Version: 0.10.0"]
        lines += [f"Requires-Dist: {requirement}" for requirement in requires]
        wheel.writestr("arcaneum-0.10.0.dist-info/METADATA", "\n".join(lines) + "\n")
    return path


def test_tree_fingerprint_is_stable_and_tracks_source_changes(tmp_path):
    package = make_tree(tmp_path)
    first = tree_fingerprint(package, REQUIRES)

    assert first == tree_fingerprint(package, REQUIRES)
    assert len(first) == 64

    (package / "utils" / "memory.py").write_text("LIMIT = 3\n")
    assert tree_fingerprint(package, REQUIRES) != first


def test_tree_fingerprint_ignores_bytecode_and_non_python_files(tmp_path):
    package = make_tree(tmp_path)
    first = tree_fingerprint(package, REQUIRES)

    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "client.cpython-312.pyc").write_bytes(b"\0")
    (package / "embeddings" / "capabilities-v1.json").write_text("{}")

    assert tree_fingerprint(package, REQUIRES) == first


def test_tree_fingerprint_tracks_declared_requirements_but_not_their_order(tmp_path):
    package = make_tree(tmp_path)
    first = tree_fingerprint(package, REQUIRES)

    assert tree_fingerprint(package, list(reversed(REQUIRES))) == first
    assert tree_fingerprint(package, [*REQUIRES[:-1], "torch>=2.13.0"]) != first


def test_tree_fingerprint_tracks_renamed_files(tmp_path):
    package = make_tree(tmp_path)
    first = tree_fingerprint(package, REQUIRES)

    (package / "utils" / "memory.py").rename(package / "utils" / "memo.py")

    assert tree_fingerprint(package, REQUIRES) != first


def test_wheel_fingerprint_matches_the_tree_it_was_built_from(tmp_path):
    package = make_tree(tmp_path / "src")
    wheel = make_wheel(package, tmp_path / "arcaneum-0.10.0-py3-none-any.whl", REQUIRES)

    assert wheel_fingerprint(wheel) == tree_fingerprint(package, REQUIRES)


def test_wheel_fingerprint_tracks_requirements(tmp_path):
    package = make_tree(tmp_path / "src")
    wheel = make_wheel(package, tmp_path / "a.whl", REQUIRES)
    changed = make_wheel(package, tmp_path / "b.whl", REQUIRES[:-1])

    assert wheel_fingerprint(wheel) != wheel_fingerprint(changed)


def test_code_fingerprint_is_the_installed_package_tree():
    requires = metadata.distribution("arcaneum").requires or []
    package = Path(arcaneum.__file__).parent

    assert code_fingerprint() == tree_fingerprint(package, requires)
