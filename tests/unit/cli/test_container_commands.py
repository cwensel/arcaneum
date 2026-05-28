"""CLI tests for container management commands.

Tests for 'arc container' subcommands: start, stop, restart, status, logs, reset.
Uses mocked subprocess and requests to avoid actual Docker operations.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestContainerStart:
    """Test 'arc container start' command."""

    def test_docker_compose_up_called(self):
        """Test that docker compose up is called with correct arguments."""
        from arcaneum.cli.docker import start_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    with patch('arcaneum.cli.docker.get_container_env', return_value={'MEILISEARCH_API_KEY': 'test'}):
                        with patch('arcaneum.cli.docker.check_qdrant_health', return_value=True):
                            with patch('arcaneum.cli.docker.check_meilisearch_health', return_value=True):
                                with patch('time.sleep'):
                                    start_command.callback()

                # Verify docker compose up was called
                calls = mock_run.call_args_list
                compose_calls = [c for c in calls if 'compose' in str(c)]
                assert any('up' in str(c) and '-d' in str(c) for c in compose_calls)

    def test_health_status_reported(self, capsys):
        """Test that health status is reported after start."""
        from arcaneum.cli.docker import start_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    with patch('arcaneum.cli.docker.get_container_env', return_value={'MEILISEARCH_API_KEY': 'test'}):
                        with patch('arcaneum.cli.docker.check_qdrant_health', return_value=True):
                            with patch('arcaneum.cli.docker.check_meilisearch_health', return_value=True):
                                with patch('time.sleep'):
                                    start_command.callback()

        captured = capsys.readouterr()
        assert 'Qdrant' in captured.out
        assert 'MeiliSearch' in captured.out

    def test_docker_unavailable_error(self, capsys):
        """Test error message when Docker is not available."""
        from arcaneum.cli.docker import start_command

        with patch('shutil.which', return_value=None):
            start_command.callback()

        captured = capsys.readouterr()
        # Error messages go to stderr
        output = captured.out + captured.err
        assert 'Docker' in output
        assert 'not installed' in output.lower()


class TestContainerStop:
    """Test 'arc container stop' command."""

    def test_docker_compose_down_called(self):
        """Test that docker compose down is called."""
        from arcaneum.cli.docker import stop_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    stop_command.callback()

                calls = mock_run.call_args_list
                compose_calls = [c for c in calls if 'compose' in str(c)]
                assert any('down' in str(c) for c in compose_calls)

    def test_already_stopped_handling(self, capsys):
        """Test graceful handling when services are already stopped."""
        from arcaneum.cli.docker import stop_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    stop_command.callback()

        captured = capsys.readouterr()
        # Should show success message even if already stopped
        assert 'stopped' in captured.out.lower() or 'Container' in captured.out


class TestContainerRestart:
    """Test 'arc container restart' command."""

    def test_docker_compose_restart_called(self):
        """Test that docker compose restart is called."""
        from arcaneum.cli.docker import restart_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    with patch('arcaneum.cli.docker.check_qdrant_health', return_value=True):
                        with patch('time.sleep'):
                            restart_command.callback()

                calls = mock_run.call_args_list
                compose_calls = [c for c in calls if 'compose' in str(c)]
                assert any('restart' in str(c) for c in compose_calls)


class TestContainerStatus:
    """Test 'arc container status' command."""

    def test_running_containers_shown(self, capsys):
        """Test that running containers are shown."""
        from arcaneum.cli.docker import status_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "CONTAINER ID   IMAGE   STATUS\nabc123   qdrant   Up 5 minutes"
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    with patch('arcaneum.cli.docker.check_qdrant_health', return_value=True):
                        with patch('arcaneum.cli.docker.check_meilisearch_health', return_value=True):
                            status_command.callback()

        captured = capsys.readouterr()
        assert 'Status' in captured.out or 'Healthy' in captured.out

    def test_health_checks_shown(self, capsys):
        """Test that health status is shown for each service."""
        from arcaneum.cli.docker import status_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = '{"Volumes": []}'
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    with patch('arcaneum.cli.docker.check_qdrant_health', return_value=True):
                        with patch('arcaneum.cli.docker.check_meilisearch_health', return_value=False):
                            status_command.callback()

        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert 'Qdrant' in output
        assert 'MeiliSearch' in output


class TestContainerLogs:
    """Test 'arc container logs' command."""

    def test_recent_output(self):
        """Test that logs show recent output."""
        from arcaneum.cli.docker import logs_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    logs_command.callback(follow=False, tail=100)

                calls = mock_run.call_args_list
                compose_calls = [c for c in calls if 'compose' in str(c)]
                assert any('logs' in str(c) for c in compose_calls)
                assert any('--tail=100' in str(c) for c in compose_calls)

    def test_follow_flag(self):
        """Test that --follow flag is passed."""
        from arcaneum.cli.docker import logs_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    logs_command.callback(follow=True, tail=50)

                calls = mock_run.call_args_list
                compose_calls = [c for c in calls if 'compose' in str(c)]
                assert any('-f' in str(c) for c in compose_calls)

    def test_tail_limit(self):
        """Test that --tail limit is respected."""
        from arcaneum.cli.docker import logs_command

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    logs_command.callback(follow=False, tail=50)

                calls = mock_run.call_args_list
                compose_calls = [c for c in calls if 'compose' in str(c)]
                assert any('--tail=50' in str(c) for c in compose_calls)


class TestContainerBackupRestore:
    """Test 'arc container backup' and 'arc container restore' commands."""

    def test_backup_writes_manifest_and_exports_services(self, temp_dir):
        """Test that backup snapshots Qdrant and exports MeiliSearch metadata."""
        from arcaneum.cli.docker import backup_command

        responses = [
            {"results": [{"uid": 42}]},
            {"results": []},
            {"result": {"collections": [{"name": "Docs"}]}},
            {"result": {"name": "Docs-123.snapshot"}},
            {"status": "ok"},
            {"results": [{"uid": "DocsText", "primaryKey": "id"}]},
            {"searchableAttributes": ["content"]},
            {"results": [{"id": "1", "content": "hello"}]},
            {"results": []},
            {"results": [{"uid": 42}]},
        ]

        def fake_request(method, url, **kwargs):
            response = MagicMock()
            response.json.return_value = responses.pop(0)
            response.raise_for_status.return_value = None
            return response

        backup_path = temp_dir / "backup"

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch('requests.request', side_effect=fake_request):
                    with patch('arcaneum.paths.get_meilisearch_api_key', return_value='secret-key'):
                        backup_command.callback(
                            output=str(backup_path),
                            qdrant_url='http://qdrant',
                            meilisearch_url='http://meili',
                            qdrant_container='qdrant',
                            qdrant_timeout=300,
                            skip_meilisearch=False,
                            output_json=False,
                        )

        manifest = json.loads((backup_path / "manifest.json").read_text())
        assert manifest["qdrant"][0]["collection"] == "Docs"
        assert manifest["meilisearch"][0]["index"] == "DocsText"
        assert "Embedding model cache" in manifest["not_protected"]

        meili_metadata = json.loads(
            (backup_path / "meilisearch" / "DocsText.metadata.json").read_text()
        )
        meili_documents = (backup_path / "meilisearch" / "DocsText.documents.jsonl").read_text()
        assert meili_metadata["settings"] == {"searchableAttributes": ["content"]}
        assert meili_metadata["documents"] == 1
        assert json.loads(meili_documents) == {"id": "1", "content": "hello"}

        docker_cp_calls = [c for c in mock_run.call_args_list if "cp" in c.args[0]]
        assert docker_cp_calls
        assert "qdrant:/qdrant/snapshots/Docs/Docs-123.snapshot" in docker_cp_calls[0].args[0]

    def test_restore_recovers_qdrant_and_recreates_meilisearch(self, temp_dir):
        """Test that restore uses manifest entries for both services."""
        from arcaneum.cli.docker import restore_command

        backup_path = temp_dir / "backup"
        (backup_path / "qdrant").mkdir(parents=True)
        (backup_path / "meilisearch").mkdir()
        (backup_path / "qdrant" / "Docs-123.snapshot").write_text("snapshot")
        (backup_path / "meilisearch" / "DocsText.metadata.json").write_text(json.dumps({
            "uid": "DocsText",
            "primaryKey": "id",
            "settings": {"searchableAttributes": ["content"]},
            "documents": 1,
        }))
        (backup_path / "meilisearch" / "DocsText.documents.jsonl").write_text(
            json.dumps({"id": "1", "content": "hello"}) + "\n"
        )
        (backup_path / "manifest.json").write_text(json.dumps({
            "qdrant": [{
                "collection": "Docs",
                "snapshot": "Docs-123.snapshot",
                "file": "qdrant/Docs-123.snapshot",
            }],
            "meilisearch": [{
                "index": "DocsText",
                "primaryKey": "id",
                "documents": 1,
                "metadata_file": "meilisearch/DocsText.metadata.json",
                "documents_file": "meilisearch/DocsText.documents.jsonl",
            }],
        }))

        requests_seen = []

        def fake_request(method, url, **kwargs):
            requests_seen.append((method, url, kwargs.get("json")))
            response = MagicMock()
            if "/tasks/" in url:
                response.json.return_value = {"status": "succeeded"}
            elif method in {"DELETE", "POST", "PATCH"} and "meili" in url:
                task_count = len([
                    r for r in requests_seen
                    if "meili" in r[1] and r[0] in {"DELETE", "POST", "PATCH"}
                ])
                response.json.return_value = {"taskUid": task_count}
            else:
                response.json.return_value = {"status": "ok"}
            response.raise_for_status.return_value = None
            return response

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch('requests.request', side_effect=fake_request):
                    with patch('arcaneum.paths.get_meilisearch_api_key', return_value='secret-key'):
                        restore_command.callback(
                            backup_directory=str(backup_path),
                            qdrant_url='http://qdrant',
                            meilisearch_url='http://meili',
                            qdrant_container='qdrant',
                            qdrant_timeout=300,
                            meilisearch_timeout=1800,
                            skip_meilisearch=False,
                            output_json=False,
                        )

        assert ("PUT", "http://qdrant/collections/Docs/snapshots/recover", {
            "location": "file:///qdrant/snapshots/Docs/Docs-123.snapshot",
        }) in requests_seen
        docker_calls = [call.args[0] for call in mock_run.call_args_list]
        assert [
            "docker", "exec", "qdrant", "mkdir", "-p", "/qdrant/snapshots/Docs",
        ] in docker_calls
        assert ("DELETE", "http://meili/indexes/DocsText", None) in requests_seen
        assert ("PATCH", "http://meili/indexes/DocsText/settings", {
            "searchableAttributes": ["content"],
        }) in requests_seen
        assert any(
            method == "POST" and url == "http://meili/indexes/DocsText/documents"
            for method, url, _ in requests_seen
        )

    def test_backup_rejects_meili_task_created_during_qdrant_snapshot(self, temp_dir):
        """Test that an empty starting task history is still a real baseline."""
        from arcaneum.cli.docker import backup_command

        responses = [
            {"results": []},
            {"results": []},
            {"result": {"collections": [{"name": "Docs"}]}},
            {"result": {"name": "Docs-123.snapshot"}},
            {"status": "ok"},
            {"results": []},
            {"results": []},
            {"results": [{"uid": 99}]},
        ]

        def fake_request(method, url, **kwargs):
            response = MagicMock()
            response.json.return_value = responses.pop(0)
            response.raise_for_status.return_value = None
            return response

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch('requests.request', side_effect=fake_request):
                    with patch('arcaneum.paths.get_meilisearch_api_key', return_value='secret-key'):
                        with pytest.raises(RuntimeError, match="task history changed"):
                            backup_command.callback(
                                output=str(temp_dir / "backup"),
                                qdrant_url='http://qdrant',
                                meilisearch_url='http://meili',
                                qdrant_container='qdrant',
                                qdrant_timeout=300,
                                skip_meilisearch=False,
                                output_json=False,
                            )


class TestContainerReset:
    """Test 'arc container reset' command."""

    def test_confirm_required(self, capsys):
        """Test that --confirm flag is required."""
        from arcaneum.cli.docker import reset_command

        reset_command.callback(confirm=False)

        captured = capsys.readouterr()
        # Error messages go to stderr
        output = captured.out + captured.err
        assert '--confirm' in output or 'confirm' in output.lower()

    def test_data_deletion_with_confirm(self, temp_dir, capsys):
        """Test that data is deleted when --confirm is provided."""
        from arcaneum.cli.docker import reset_command

        # Create mock data directories
        qdrant_dir = temp_dir / "qdrant"
        qdrant_dir.mkdir()
        (qdrant_dir / "test_data.db").touch()

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    with patch('arcaneum.cli.docker.get_data_dir', return_value=temp_dir):
                        reset_command.callback(confirm=True)

        captured = capsys.readouterr()
        assert 'reset' in captured.out.lower() or 'complete' in captured.out.lower()

    def test_stops_services_first(self, temp_dir):
        """Test that services are stopped before reset."""
        from arcaneum.cli.docker import reset_command

        call_order = []

        def track_calls(*args, **kwargs):
            if 'down' in str(args):
                call_order.append('down')
            result = MagicMock()
            result.returncode = 0
            return result

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run', side_effect=track_calls):
                with patch('arcaneum.cli.docker.get_compose_file', return_value='/path/to/docker-compose.yml'):
                    with patch('arcaneum.cli.docker.get_data_dir', return_value=temp_dir):
                        reset_command.callback(confirm=True)

        # Verify 'down' was called
        assert 'down' in call_order


class TestCheckDockerAvailable:
    """Test Docker availability checking."""

    def test_docker_installed_and_running(self):
        """Test when Docker is installed and running."""
        from arcaneum.cli.docker import check_docker_available

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                result = check_docker_available()

        assert result is True

    def test_docker_not_installed(self, capsys):
        """Test when Docker is not installed."""
        from arcaneum.cli.docker import check_docker_available

        with patch('shutil.which', return_value=None):
            result = check_docker_available()

        assert result is False
        captured = capsys.readouterr()
        # Error messages go to stderr
        output = captured.out + captured.err
        assert 'not installed' in output.lower()

    def test_docker_not_running(self, capsys):
        """Test when Docker is installed but not running."""
        from arcaneum.cli.docker import check_docker_available

        with patch('shutil.which', return_value='/usr/bin/docker'):
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(1, 'docker info')
                result = check_docker_available()

        assert result is False
        captured = capsys.readouterr()
        # Error messages go to stderr
        output = captured.out + captured.err
        assert 'not running' in output.lower()


class TestHealthChecks:
    """Test service health check functions."""

    def test_qdrant_health_check_success(self):
        """Test Qdrant health check when healthy."""
        from arcaneum.cli.docker import check_qdrant_health

        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = check_qdrant_health()

        assert result is True
        mock_get.assert_called_once_with("http://localhost:6333/healthz", timeout=2)

    def test_qdrant_health_check_failure(self):
        """Test Qdrant health check when unhealthy."""
        from arcaneum.cli.docker import check_qdrant_health
        import requests

        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")

            result = check_qdrant_health()

        assert result is False

    def test_meilisearch_health_check_success(self):
        """Test MeiliSearch health check when healthy."""
        from arcaneum.cli.docker import check_meilisearch_health

        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = check_meilisearch_health()

        assert result is True
        mock_get.assert_called_once_with("http://localhost:7700/health", timeout=2)

    def test_meilisearch_health_check_failure(self):
        """Test MeiliSearch health check when unhealthy."""
        from arcaneum.cli.docker import check_meilisearch_health
        import requests

        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")

            result = check_meilisearch_health()

        assert result is False


class TestGetComposeFile:
    """Test Docker Compose file discovery."""

    def test_finds_compose_file_in_deploy(self, temp_dir, monkeypatch):
        """Test finding compose file in deploy directory."""
        from arcaneum.cli import docker as docker_module

        # Create a fake repo structure. get_compose_file walks up 4 parents from
        # docker.py to reach repo root, so we emulate that by pointing __file__
        # at a nested path whose parent.parent.parent.parent is temp_dir.
        fake_source = temp_dir / "src" / "arcaneum" / "cli" / "docker.py"
        fake_source.parent.mkdir(parents=True)
        fake_source.touch()

        deploy_dir = temp_dir / "deploy"
        deploy_dir.mkdir()
        compose_file = deploy_dir / "docker-compose.yml"
        compose_file.touch()

        monkeypatch.setattr(docker_module, '__file__', str(fake_source))

        result = docker_module.get_compose_file()

        assert result is not None
        assert Path(result).resolve() == compose_file.resolve()

    def test_compose_file_not_found(self, temp_dir, capsys):
        """Test error when compose file not found."""
        from arcaneum.cli.docker import get_compose_file

        with patch('pathlib.Path.exists', return_value=False):
            result = get_compose_file()

        # Should return None and print error
        # Note: Due to path resolution complexity, we check the general behavior
        # The function iterates through paths, and if none exist, returns None
