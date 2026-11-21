"""Upgrade test from Qdrant 1.18.2 (release compatibility policy, RDR-027 Q12).

A collection written by 1.18.2 is served in place by each step of
`arc container upgrade` (`_QDRANT_UPGRADE_STEPS`), and keeps identical point
ids, vectors, payloads and payload schema after every step.

It runs its own throwaway container and named volume on a spare port, never
the `arcaneum` compose stack, and is skipped without Docker.
"""

import hashlib
import json
import shutil
import subprocess
import time
import uuid

import pytest
import requests

from arcaneum.cli.docker import _QDRANT_IMAGE, _QDRANT_UPGRADE_STEPS

pytestmark = [pytest.mark.integration, pytest.mark.slow]

FROM_TAG = "v1.18.2"
PORT = 16353
URL = f"http://localhost:{PORT}"
COLLECTION = "UpgradeProbe"
POINTS = 300


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=600)


def _http(method: str, path: str, body=None) -> dict:
    response = requests.request(method, f"{URL}{path}", json=body, timeout=60)
    response.raise_for_status()
    return response.json()["result"]


def _serve(tag: str, name: str, volume: str, *, loaded: bool = True) -> None:
    started = _docker(
        "run", "-d", "--name", name, "-p", f"{PORT}:6333",
        "-v", f"{volume}:/qdrant/storage", f"{_QDRANT_IMAGE}:{tag}",
    )  # fmt: skip
    assert started.returncode == 0, started.stderr
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            status = requests.get(f"{URL}/collections/{COLLECTION}", timeout=2).status_code
            if status == 200 or (status == 404 and not loaded):
                if requests.get(f"{URL}/", timeout=2).json()["version"] == tag.lstrip("v"):
                    return
        except (requests.RequestException, ValueError, KeyError):
            pass
        time.sleep(1)
    pytest.fail(f"Qdrant {tag} did not start: {_docker('logs', name).stderr[-2000:]}")


def _retire(name: str) -> None:
    # Graceful, so Qdrant flushes before the next image opens the storage.
    _docker("stop", "-t", "60", name)
    _docker("rm", "-f", "-v", name)


def _load() -> None:
    _http(
        "PUT",
        f"/collections/{COLLECTION}",
        {"vectors": {"dense": {"size": 8, "distance": "Cosine"}}, "on_disk_payload": True},
    )
    for field, schema in (("file_path", "keyword"), ("chunk_index", "integer")):
        _http(
            "PUT",
            f"/collections/{COLLECTION}/index?wait=true",
            {"field_name": field, "field_schema": schema},
        )
    points = [
        {
            "id": str(uuid.UUID(int=i + 1)),
            "vector": {"dense": [((i * 7 + d) % 11) / 11 + 0.01 for d in range(8)]},
            "payload": {
                "file_path": f"/corpus/doc-{i // 10}.md",
                "chunk_index": i % 10,
                "text": f"chunk {i} of the upgrade probe",
            },
        }
        for i in range(POINTS)
    ]
    _http("PUT", f"/collections/{COLLECTION}/points?wait=true", {"points": points})


def _fingerprint() -> dict:
    info = _http("GET", f"/collections/{COLLECTION}")
    rows, offset = [], None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        page = _http("POST", f"/collections/{COLLECTION}/points/scroll", body)
        rows += [(str(p["id"]), p["vector"], p["payload"]) for p in page["points"]]
        offset = page.get("next_page_offset")
        if offset is None:
            break
    rows.sort(key=lambda row: row[0])

    def digest(values) -> str:
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    return {
        "points": info["points_count"],
        # Each index's type and configuration; the indexed-point tally is left
        # to "points" and the payload digest.
        "payload_schema": {
            field: (schema.get("data_type"), schema.get("params"))
            for field, schema in info["payload_schema"].items()
        },
        "ids": digest([row[0] for row in rows]),
        "vectors": digest([row[1] for row in rows]),
        "payloads": digest([row[2] for row in rows]),
    }


@pytest.mark.skipif(not _docker_available(), reason="Docker is not available")
def test_collection_from_1_18_2_survives_each_upgrade_step():
    suffix = uuid.uuid4().hex[:8]
    name, volume = f"arc-qdrant-upgrade-{suffix}", f"arc-qdrant-upgrade-{suffix}"
    assert _docker("volume", "create", volume).returncode == 0
    try:
        _serve(FROM_TAG, name, volume, loaded=False)
        _load()
        baseline = _fingerprint()
        _retire(name)
        assert baseline["points"] == POINTS
        assert {"file_path", "chunk_index"} <= set(baseline["payload_schema"])

        for tag in _QDRANT_UPGRADE_STEPS:
            _serve(tag, name, volume)
            assert _fingerprint() == baseline, f"collection changed on {tag}"
            _retire(name)
    finally:
        _docker("rm", "-f", "-v", name)
        _docker("volume", "rm", volume)
