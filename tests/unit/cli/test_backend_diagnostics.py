from arcaneum.cli.sync import _report_embedding_backend


class Client:
    def get_backend_diagnostics(self, model):
        return {
            "backend": "fastembed-cpu",
            "state": "stable",
            "device": "cpu",
            "evidence_version": "v1",
            "fallback_reason": "experimental denied",
            "worker_restart_count": 0,
        }


def test_verbose_backend_diagnostics_expose_state_and_fallback(monkeypatch):
    lines = []
    monkeypatch.setattr("arcaneum.cli.sync.console.print", lines.append)
    _report_embedding_backend(Client(), ["arctic-m"], True, False)
    assert "backend=fastembed-cpu" in lines[0]
    assert "state=stable" in lines[0]
    assert "fallback=experimental denied" in lines[0]
    assert "worker_restarts=0" in lines[0]


def test_json_mode_suppresses_human_diagnostics(monkeypatch):
    lines = []
    monkeypatch.setattr("arcaneum.cli.sync.console.print", lines.append)
    _report_embedding_backend(Client(), ["arctic-m"], True, True)
    assert lines == []
