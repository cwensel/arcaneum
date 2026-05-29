"""Tests for markdown injection persistence."""


def test_get_agent_memory_dir_uses_xdg_data_home(monkeypatch, tmp_path):
    from arcaneum.indexing.markdown.injection import get_agent_memory_dir

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    memory_dir = get_agent_memory_dir("Memory")

    assert memory_dir == tmp_path / "arcaneum" / "agent-memory" / "Memory"
    assert memory_dir.exists()
