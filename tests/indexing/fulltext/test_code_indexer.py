"""Unit tests for SourceCodeFullTextIndexer (RDR-011).

Tests the source code full-text indexing functionality for MeiliSearch.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from arcaneum.indexing.fulltext.code_indexer import SourceCodeFullTextIndexer
from arcaneum.indexing.fulltext.ast_extractor import CodeDefinition
from arcaneum.indexing.types import GitMetadata


class TestSourceCodeFullTextIndexerInit:
    """Tests for indexer initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        mock_client = Mock()
        indexer = SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index"
        )

        assert indexer.meili_client is mock_client
        assert indexer.index_name == "test-index"
        assert indexer.batch_size == 1000

    def test_init_custom_batch_size(self):
        """Test initialization with custom batch size."""
        mock_client = Mock()
        indexer = SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index",
            batch_size=500
        )

        assert indexer.batch_size == 500


class TestSourceCodeFullTextIndexerDocumentBuilding:
    """Tests for document building functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MeiliSearch client."""
        client = Mock()
        client.health_check.return_value = True
        client.search.return_value = {"hits": [], "estimatedTotalHits": 0}
        return client

    @pytest.fixture
    def indexer(self, mock_client):
        """Create indexer with mocked client."""
        return SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-code-index"
        )

    @pytest.fixture
    def sample_git_metadata(self):
        """Create sample git metadata."""
        return GitMetadata(
            project_name="test-project",
            branch="main",
            commit_hash="abc123def456",
            remote_url="https://github.com/test/project.git",
            project_root="/path/to/project"
        )

    def test_build_document_function(self, indexer, sample_git_metadata):
        """Test document building for function definition."""
        defn = CodeDefinition(
            name="my_function",
            qualified_name="my_function",
            code_type="function",
            start_line=10,
            end_line=20,
            content="def my_function():\n    pass",
            file_path="/path/to/project/src/module.py"
        )

        doc = indexer._build_document(
            defn, sample_git_metadata, "test-project#main", "/path/to/project/src/module.py"
        )

        assert "id" in doc
        assert doc["content"] == "def my_function():\n    pass"
        assert doc["function_name"] == "my_function"
        assert doc["class_name"] is None
        assert doc["qualified_name"] == "my_function"
        assert doc["filename"] == "module.py"
        assert doc["git_project_identifier"] == "test-project#main"
        assert doc["git_project_name"] == "test-project"
        assert doc["git_branch"] == "main"
        assert doc["git_commit_hash"] == "abc123def456"
        assert doc["start_line"] == 10
        assert doc["end_line"] == 20
        assert doc["line_count"] == 11
        assert doc["code_type"] == "function"
        assert doc["programming_language"] == "python"
        assert doc["file_extension"] == ".py"

    def test_build_document_class(self, indexer, sample_git_metadata):
        """Test document building for class definition."""
        defn = CodeDefinition(
            name="MyClass",
            qualified_name="MyClass",
            code_type="class",
            start_line=1,
            end_line=50,
            content="class MyClass:\n    pass",
            file_path="/path/to/project/src/module.py"
        )

        doc = indexer._build_document(
            defn, sample_git_metadata, "test-project#main", "/path/to/project/src/module.py"
        )

        assert doc["function_name"] is None
        assert doc["class_name"] == "MyClass"
        assert doc["code_type"] == "class"

    def test_build_document_method(self, indexer, sample_git_metadata):
        """Test document building for method definition."""
        defn = CodeDefinition(
            name="my_method",
            qualified_name="MyClass.my_method",
            code_type="method",
            start_line=10,
            end_line=15,
            content="def my_method(self):\n    pass",
            file_path="/path/to/project/src/module.py"
        )

        doc = indexer._build_document(
            defn, sample_git_metadata, "test-project#main", "/path/to/project/src/module.py"
        )

        assert doc["function_name"] == "my_method"
        assert doc["qualified_name"] == "MyClass.my_method"
        assert doc["code_type"] == "method"

    def test_build_document_unique_ids(self, indexer, sample_git_metadata):
        """Test that document IDs are unique."""
        defn1 = CodeDefinition(
            name="func",
            qualified_name="func",
            code_type="function",
            start_line=1,
            end_line=5,
            content="def func(): pass",
            file_path="/path/to/project/src/a.py"
        )

        defn2 = CodeDefinition(
            name="func",
            qualified_name="func",
            code_type="function",
            start_line=1,
            end_line=5,
            content="def func(): pass",
            file_path="/path/to/project/src/b.py"
        )

        doc1 = indexer._build_document(
            defn1, sample_git_metadata, "test-project#main", "/path/to/project/src/a.py"
        )
        doc2 = indexer._build_document(
            defn2, sample_git_metadata, "test-project#main", "/path/to/project/src/b.py"
        )

        assert doc1["id"] != doc2["id"]

    def test_build_document_different_branches(self, indexer):
        """Test document IDs differ for same file on different branches."""
        metadata_main = GitMetadata(
            project_name="project",
            branch="main",
            commit_hash="abc",
            remote_url="https://github.com/test/project.git",
            project_root="/path/to/project"
        )

        metadata_dev = GitMetadata(
            project_name="project",
            branch="develop",
            commit_hash="def",
            remote_url="https://github.com/test/project.git",
            project_root="/path/to/project"
        )

        defn = CodeDefinition(
            name="func",
            qualified_name="func",
            code_type="function",
            start_line=1,
            end_line=5,
            content="def func(): pass",
            file_path="/path/to/project/src/module.py"
        )

        doc_main = indexer._build_document(
            defn, metadata_main, "project#main", "/path/to/project/src/module.py"
        )
        doc_dev = indexer._build_document(
            defn, metadata_dev, "project#develop", "/path/to/project/src/module.py"
        )

        assert doc_main["id"] != doc_dev["id"]
        assert doc_main["git_branch"] == "main"
        assert doc_dev["git_branch"] == "develop"

    def test_build_document_language_detection(self, indexer, sample_git_metadata):
        """Test language detection from file extension."""
        test_cases = [
            ("/test.py", "python"),
            ("/test.js", "javascript"),
            ("/test.ts", "typescript"),
            ("/test.java", "java"),
            ("/test.go", "go"),
            ("/test.rs", "rust"),
        ]

        for file_path, expected_lang in test_cases:
            defn = CodeDefinition(
                name="func",
                qualified_name="func",
                code_type="function",
                start_line=1,
                end_line=5,
                content="code",
                file_path=file_path
            )

            doc = indexer._build_document(
                defn, sample_git_metadata, "test#main", file_path
            )

            assert doc["programming_language"] == expected_lang, \
                f"Expected {expected_lang} for {file_path}"


class TestSourceCodeFullTextIndexerCodeExtensions:
    """Tests for code file extension filtering."""

    def test_code_extensions_includes_common(self):
        """Test that common extensions are included."""
        expected = ['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c']
        for ext in expected:
            assert ext in SourceCodeFullTextIndexer.CODE_EXTENSIONS

    def test_code_extensions_count(self):
        """Test reasonable number of extensions supported."""
        # Should support many languages
        assert len(SourceCodeFullTextIndexer.CODE_EXTENSIONS) > 20


class TestSourceCodeFullTextIndexerBatchUpload:
    """Tests for batch upload functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MeiliSearch client."""
        client = Mock()
        client.add_documents_sync.return_value = {"taskUid": 123}
        return client

    @pytest.fixture
    def indexer(self, mock_client):
        """Create indexer with mocked client."""
        return SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index",
            batch_size=100
        )

    def test_upload_batch_empty(self, indexer):
        """Test upload with empty document list does nothing."""
        indexer._upload_batch([])
        indexer.meili_client.add_documents_sync.assert_not_called()

    def test_upload_batch_success(self, indexer):
        """Test successful batch upload."""
        documents = [{"id": "1", "content": "test"}]
        indexer._upload_batch(documents)

        indexer.meili_client.add_documents_sync.assert_called_once_with(
            index_name="test-index",
            documents=documents,
            timeout_ms=120000
        )

    def test_upload_batch_error(self, indexer):
        """Test batch upload error handling."""
        indexer.meili_client.add_documents_sync.side_effect = Exception("Upload failed")

        with pytest.raises(Exception, match="Upload failed"):
            indexer._upload_batch([{"id": "1"}])


class TestSourceCodeFullTextIndexerWithMockedGit:
    """Tests for indexing with mocked git operations."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MeiliSearch client."""
        client = Mock()
        client.health_check.return_value = True
        client.search.return_value = {"hits": [], "estimatedTotalHits": 0}
        client.add_documents_sync.return_value = {"taskUid": 123}
        return client

    @pytest.fixture
    def sample_project(self):
        """Create a sample git project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project structure
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            # Create Python files
            (src_dir / "main.py").write_text('''
def main():
    print("Hello, World!")

class App:
    def run(self):
        pass
''')

            (src_dir / "utils.py").write_text('''
def helper():
    return 42
''')

            # Create JavaScript file
            (src_dir / "index.js").write_text('''
function greet(name) {
    console.log("Hello, " + name);
}
''')

            yield tmpdir

    def test_get_code_files(self, mock_client, sample_project):
        """Test getting code files from project."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index"
        )

        # Mock git_discovery to return our test files
        src_dir = Path(sample_project) / "src"
        mock_files = [
            str(src_dir / "main.py"),
            str(src_dir / "utils.py"),
            str(src_dir / "index.js"),
            str(src_dir / "README.md"),  # Should be filtered out
        ]

        with patch.object(indexer.git_discovery, 'get_tracked_files', return_value=mock_files):
            code_files = indexer._get_code_files(sample_project)

        assert len(code_files) == 3
        assert str(src_dir / "main.py") in code_files
        assert str(src_dir / "utils.py") in code_files
        assert str(src_dir / "index.js") in code_files
        assert str(src_dir / "README.md") not in code_files


class TestGitCodeMetadataSync:
    """Tests for GitCodeMetadataSync functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MeiliSearch client."""
        client = Mock()
        return client

    def test_should_reindex_not_indexed(self, mock_client):
        """Test should_reindex returns True for new project."""
        from arcaneum.indexing.fulltext.sync import GitCodeMetadataSync

        mock_client.search.return_value = {"hits": [], "estimatedTotalHits": 0}
        sync = GitCodeMetadataSync(mock_client)

        result = sync.should_reindex_project(
            "test-index", "project#main", "abc123"
        )

        assert result is True

    def test_should_reindex_commit_changed(self, mock_client):
        """Test should_reindex returns True when commit changed."""
        from arcaneum.indexing.fulltext.sync import GitCodeMetadataSync

        mock_client.search.return_value = {
            "hits": [{
                "git_project_identifier": "project#main",
                "git_commit_hash": "old_commit"
            }],
            "estimatedTotalHits": 1
        }
        sync = GitCodeMetadataSync(mock_client)

        result = sync.should_reindex_project(
            "test-index", "project#main", "new_commit"
        )

        assert result is True

    def test_should_reindex_unchanged(self, mock_client):
        """Test should_reindex returns False when unchanged."""
        from arcaneum.indexing.fulltext.sync import GitCodeMetadataSync

        mock_client.search.return_value = {
            "hits": [{
                "git_project_identifier": "project#main",
                "git_commit_hash": "same_commit"
            }],
            "estimatedTotalHits": 1
        }
        sync = GitCodeMetadataSync(mock_client)

        result = sync.should_reindex_project(
            "test-index", "project#main", "same_commit"
        )

        assert result is False

    def test_cache_behavior(self, mock_client):
        """Test that indexed projects are cached."""
        from arcaneum.indexing.fulltext.sync import GitCodeMetadataSync

        mock_client.search.return_value = {
            "hits": [{
                "git_project_identifier": "project#main",
                "git_commit_hash": "abc123"
            }],
            "estimatedTotalHits": 1
        }
        sync = GitCodeMetadataSync(mock_client)

        # First call
        sync.get_indexed_projects("test-index")
        call_count_1 = mock_client.search.call_count

        # Second call should use cache
        sync.get_indexed_projects("test-index")
        call_count_2 = mock_client.search.call_count

        assert call_count_1 == call_count_2

    def test_clear_cache(self, mock_client):
        """Test cache clearing."""
        from arcaneum.indexing.fulltext.sync import GitCodeMetadataSync

        mock_client.search.return_value = {
            "hits": [],
            "estimatedTotalHits": 0
        }
        sync = GitCodeMetadataSync(mock_client)

        # First call
        sync.get_indexed_projects("test-index")
        call_count_1 = mock_client.search.call_count

        # Clear cache
        sync.clear_cache()

        # Next call should query again
        sync.get_indexed_projects("test-index")
        call_count_2 = mock_client.search.call_count

        assert call_count_2 > call_count_1


class TestSourceCodeFullTextIndexerParallel:
    """Tests for parallel processing functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock MeiliSearch client."""
        client = Mock()
        client.health_check.return_value = True
        client.search.return_value = {"hits": [], "estimatedTotalHits": 0}
        client.add_documents_sync.return_value = {"taskUid": 123}
        return client

    def test_init_workers_default(self, mock_client):
        """Test default workers is auto-calculated."""
        from multiprocessing import cpu_count
        indexer = SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index"
        )
        expected_workers = max(1, cpu_count() // 2)
        assert indexer.workers == expected_workers

    def test_init_workers_explicit(self, mock_client):
        """Test explicit workers setting."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index",
            workers=4
        )
        assert indexer.workers == 4

    def test_init_workers_zero_means_sequential(self, mock_client):
        """Test workers=0 means sequential (1 worker)."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index",
            workers=0
        )
        assert indexer.workers == 1

    def test_init_workers_one_is_sequential(self, mock_client):
        """Test workers=1 means sequential."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index",
            workers=1
        )
        assert indexer.workers == 1

    def test_init_workers_negative_means_sequential(self, mock_client):
        """Test negative workers means sequential (1 worker)."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index",
            workers=-5
        )
        assert indexer.workers == 1

    def test_init_workers_large(self, mock_client):
        """Test large worker count is respected."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=mock_client,
            index_name="test-index",
            workers=16
        )
        assert indexer.workers == 16


class TestExtractDefinitionsWorker:
    """Tests for the worker function used in parallel processing."""

    def test_worker_extracts_definitions(self, tmp_path):
        """Test worker function extracts definitions from a file."""
        from arcaneum.indexing.fulltext.code_indexer import _extract_definitions_worker

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        file_path, defn_dicts, error = _extract_definitions_worker(str(test_file))

        assert file_path == str(test_file)
        assert error is None
        assert len(defn_dicts) >= 1
        # Should find the function
        function_found = any(d['name'] == 'hello' for d in defn_dicts)
        assert function_found

    def test_worker_handles_missing_file(self, tmp_path):
        """Test worker handles missing file gracefully."""
        from arcaneum.indexing.fulltext.code_indexer import _extract_definitions_worker

        missing_file = str(tmp_path / "does_not_exist.py")
        file_path, defn_dicts, error = _extract_definitions_worker(missing_file)

        assert file_path == missing_file
        assert defn_dicts == []
        assert error is not None

    def test_worker_returns_serializable_dicts(self, tmp_path):
        """Test worker returns dictionaries (not CodeDefinition objects)."""
        from arcaneum.indexing.fulltext.code_indexer import _extract_definitions_worker

        test_file = tmp_path / "test.py"
        test_file.write_text("class MyClass:\n    def method(self):\n        pass\n")

        file_path, defn_dicts, error = _extract_definitions_worker(str(test_file))

        assert error is None
        assert isinstance(defn_dicts, list)
        for d in defn_dicts:
            assert isinstance(d, dict)
            # Check expected keys
            assert 'name' in d
            assert 'qualified_name' in d
            assert 'code_type' in d
            assert 'start_line' in d
            assert 'end_line' in d
            assert 'content' in d
            assert 'file_path' in d
