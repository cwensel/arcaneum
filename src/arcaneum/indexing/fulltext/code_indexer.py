"""Source code full-text indexer for MeiliSearch (RDR-011).

Indexes source code to MeiliSearch at function/class-level granularity
for exact phrase and keyword search, complementary to Qdrant's semantic
search (RDR-005).

Key features:
- Function/class-level granularity with line ranges
- Git-aware multi-branch support (composite identifiers)
- Metadata-based sync (MeiliSearch as source of truth)
- Filter-based branch-specific deletion
- Parallel file processing for improved throughput
"""

import gc
import hashlib
import logging
import os
import re
from concurrent.futures import as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from ..git_operations import GitProjectDiscovery
from ..common.multiprocessing import create_process_pool
from ..types import GitMetadata
from .ast_extractor import ASTFunctionExtractor, CodeDefinition
from .sync import GitCodeMetadataSync
from ...fulltext.client import FullTextClient

logger = logging.getLogger(__name__)


def _extract_definitions_worker(
    file_path: str,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Process a single file: read and extract AST definitions.

    Module-level function for ProcessPoolExecutor pickling.

    Args:
        file_path: Path to source file

    Returns:
        Tuple of (file_path, list of definition dicts, error message or None)
    """
    # Set low priority for background processing
    if os.environ.get('ARCANEUM_DISABLE_WORKER_NICE') != '1':
        try:
            if hasattr(os, 'nice'):
                os.nice(10)
        except Exception:
            pass

    try:
        # Import inside worker to avoid pickling issues
        from .ast_extractor import ASTFunctionExtractor
        extractor = ASTFunctionExtractor()

        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()

        # Extract definitions
        definitions = extractor.extract_definitions(file_path, code)

        # Convert CodeDefinition objects to dicts for pickling
        defn_dicts = [
            {
                'name': d.name,
                'qualified_name': d.qualified_name,
                'code_type': d.code_type,
                'start_line': d.start_line,
                'end_line': d.end_line,
                'content': d.content,
                'file_path': d.file_path,
            }
            for d in definitions
        ]

        return (file_path, defn_dicts, None)

    except Exception as e:
        return (file_path, [], str(e))


class SourceCodeFullTextIndexer:
    """Index source code to MeiliSearch at function/class level.

    This class orchestrates the full-text indexing pipeline:
    1. Discovers git projects using RDR-005's GitProjectDiscovery
    2. Checks MeiliSearch for already indexed projects (source of truth)
    3. Extracts function/class definitions using ASTFunctionExtractor
    4. Builds MeiliSearch documents with rich metadata
    5. Uploads documents in batches with progress tracking

    Attributes:
        meili_client: MeiliSearch client instance
        index_name: Target MeiliSearch index name
        batch_size: Documents per batch upload (default: 1000)
    """

    # File extensions to index (same as ast_chunker.py for consistency)
    CODE_EXTENSIONS = {
        '.py', '.java', '.js', '.jsx', '.ts', '.tsx',
        '.cs', '.go', '.rs', '.c', '.h', '.cpp', '.cc', '.cxx', '.hpp',
        '.php', '.rb', '.kt', '.kts', '.scala', '.sc', '.swift',
        '.sh', '.bash', '.zsh', '.r', '.R',
        '.lua', '.vim', '.el', '.clj', '.ex', '.exs',
        '.erl', '.hrl', '.hs', '.ml', '.nim', '.pl', '.pm',
    }

    def __init__(
        self,
        meili_client: FullTextClient,
        index_name: str,
        batch_size: int = 1000,
        workers: Optional[int] = None
    ):
        """Initialize source code full-text indexer.

        Args:
            meili_client: MeiliSearch client instance
            index_name: Target MeiliSearch index name
            batch_size: Documents per batch upload (default: 1000)
            workers: Parallel workers for AST extraction.
                     None = auto (cpu_count // 2)
                     0 or 1 = sequential (no parallelization)
                     N > 1 = use N workers
        """
        self.meili_client = meili_client
        self.index_name = index_name
        self.batch_size = batch_size

        # Configure parallel workers
        if workers is None:
            self.workers = max(1, cpu_count() // 2)
        elif workers <= 1:
            self.workers = 1  # Sequential mode
        else:
            self.workers = workers

        # Reuse RDR-005 components
        self.git_discovery = GitProjectDiscovery()
        self.ast_extractor = ASTFunctionExtractor()
        self.sync = GitCodeMetadataSync(meili_client)

    def index_directory(
        self,
        input_path: str,
        depth: Optional[int] = None,
        force: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Index all git repositories in directory to MeiliSearch.

        Discovers git projects, checks for changes against MeiliSearch,
        and indexes new/modified projects at function/class level.

        Args:
            input_path: Directory to search for git repositories
            depth: Maximum directory depth for git discovery (None = unlimited)
            force: Force reindex all projects (ignore change detection)
            verbose: Show detailed progress

        Returns:
            Dict with indexing statistics
        """
        stats = {
            'total_projects': 0,
            'indexed_projects': 0,
            'skipped_projects': 0,
            'failed_projects': 0,
            'total_files': 0,
            'indexed_files': 0,
            'total_definitions': 0,
            'errors': [],
        }

        # Query MeiliSearch for indexed projects (unless forcing)
        if not force:
            indexed_projects = self.sync.get_indexed_projects(self.index_name)
            logger.info(f"Found {len(indexed_projects)} indexed project branches in MeiliSearch")
        else:
            indexed_projects = {}

        # Discover git projects (RDR-005)
        git_projects = self.git_discovery.find_git_projects(input_path, depth)
        stats['total_projects'] = len(git_projects)

        if not git_projects:
            logger.info("No git projects found to index")
            return stats

        # Process each project with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task(
                "[cyan]Indexing git projects to MeiliSearch",
                total=len(git_projects)
            )

            for project_root in git_projects:
                try:
                    # Extract git metadata
                    git_metadata = self.git_discovery.extract_metadata(project_root)
                    if not git_metadata:
                        stats['failed_projects'] += 1
                        stats['errors'].append({
                            'project': project_root,
                            'error': 'Failed to extract git metadata'
                        })
                        progress.update(task, advance=1)
                        continue

                    identifier = git_metadata.identifier

                    # Check if needs reindexing
                    needs_indexing = force or self.sync.should_reindex_project(
                        self.index_name, identifier, git_metadata.commit_hash
                    )

                    if not needs_indexing:
                        if verbose:
                            progress.console.print(
                                f"  [dim]Skipped:[/dim] {identifier} [dim](up to date)[/dim]"
                            )
                        stats['skipped_projects'] += 1
                        progress.update(task, advance=1)
                        continue

                    # Delete old documents if reindexing
                    if identifier in indexed_projects:
                        self.sync.delete_project_documents(self.index_name, identifier)
                        self.sync.clear_cache()

                    # Index project
                    project_stats = self._index_project(
                        project_root, git_metadata, identifier, verbose, progress
                    )

                    stats['indexed_projects'] += 1
                    stats['total_files'] += project_stats['total_files']
                    stats['indexed_files'] += project_stats['indexed_files']
                    stats['total_definitions'] += project_stats['total_definitions']

                    if verbose:
                        progress.console.print(
                            f"  [green]Indexed:[/green] {identifier} "
                            f"[dim]({project_stats['indexed_files']} files, "
                            f"{project_stats['total_definitions']} definitions)[/dim]"
                        )

                except Exception as e:
                    stats['failed_projects'] += 1
                    stats['errors'].append({
                        'project': project_root,
                        'error': str(e)
                    })
                    if verbose:
                        progress.console.print(f"  [red]Failed:[/red] {project_root}: {e}")

                progress.update(task, advance=1)

        return stats

    def _index_project(
        self,
        project_root: str,
        git_metadata: GitMetadata,
        identifier: str,
        verbose: bool,
        progress
    ) -> Dict[str, Any]:
        """Index all files in a git project.

        Uses parallel processing if workers > 1, otherwise sequential.

        Args:
            project_root: Path to git repository root
            git_metadata: Git metadata for this project
            identifier: Composite identifier (project#branch)
            verbose: Show detailed progress
            progress: Rich progress instance for output

        Returns:
            Dict with project indexing statistics
        """
        stats = {
            'total_files': 0,
            'indexed_files': 0,
            'total_definitions': 0,
            'errors': [],
        }

        # Get tracked files (RDR-005 pattern)
        code_files = self._get_code_files(project_root)
        stats['total_files'] = len(code_files)

        if not code_files:
            return stats

        # Choose processing method based on worker count
        if self.workers > 1 and len(code_files) > 1:
            documents, stats = self._process_files_parallel(
                code_files, git_metadata, identifier, verbose, progress, stats
            )
        else:
            documents, stats = self._process_files_sequential(
                code_files, git_metadata, identifier, verbose, progress, stats
            )

        # Upload remaining documents
        if documents:
            self._upload_batch(documents)

        return stats

    def _process_files_sequential(
        self,
        code_files: List[str],
        git_metadata: GitMetadata,
        identifier: str,
        verbose: bool,
        progress,
        stats: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process files sequentially (original implementation).

        Args:
            code_files: List of file paths to process
            git_metadata: Git metadata for this project
            identifier: Composite identifier (project#branch)
            verbose: Show detailed progress
            progress: Rich progress instance for output
            stats: Statistics dict to update

        Returns:
            Tuple of (remaining documents, updated stats)
        """
        documents: List[Dict[str, Any]] = []

        for file_path in code_files:
            try:
                # Read file content
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        code = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    continue

                # Extract function/class definitions
                definitions = self.ast_extractor.extract_definitions(file_path, code)

                # Build MeiliSearch documents
                for defn in definitions:
                    doc = self._build_document(defn, git_metadata, identifier, file_path)
                    documents.append(doc)
                    stats['total_definitions'] += 1

                stats['indexed_files'] += 1

                # Upload in batches
                if len(documents) >= self.batch_size:
                    self._upload_batch(documents)
                    documents = []

            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                stats['errors'].append({'file': file_path, 'error': str(e)})

        return documents, stats

    def _process_files_parallel(
        self,
        code_files: List[str],
        git_metadata: GitMetadata,
        identifier: str,
        verbose: bool,
        progress,
        stats: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process files in parallel using ProcessPoolExecutor.

        Uses worker processes for CPU-bound AST extraction.
        Batch uploads remain sequential to avoid MeiliSearch contention.

        Args:
            code_files: List of file paths to process
            git_metadata: Git metadata for this project
            identifier: Composite identifier (project#branch)
            verbose: Show detailed progress
            progress: Rich progress instance for output
            stats: Statistics dict to update

        Returns:
            Tuple of (remaining documents, updated stats)
        """
        documents: List[Dict[str, Any]] = []

        # Use shared process pool with proper fork context and signal handling
        executor = None
        future_to_file = {}
        try:
            executor = create_process_pool(max_workers=self.workers)
            # Submit all file processing jobs
            future_to_file = {
                executor.submit(_extract_definitions_worker, file_path): file_path
                for file_path in code_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]

                try:
                    result_path, defn_dicts, error = future.result()

                    if error:
                        logger.warning(f"Error processing {result_path}: {error}")
                        stats['errors'].append({'file': result_path, 'error': error})
                        continue

                    if defn_dicts:
                        # Convert dicts back to CodeDefinition and build documents
                        for defn_dict in defn_dicts:
                            defn = CodeDefinition(**defn_dict)
                            doc = self._build_document(
                                defn, git_metadata, identifier, file_path
                            )
                            documents.append(doc)
                            stats['total_definitions'] += 1

                        stats['indexed_files'] += 1

                    # Batch upload
                    if len(documents) >= self.batch_size:
                        self._upload_batch(documents)
                        documents = []

                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    stats['errors'].append({'file': file_path, 'error': str(e)})

        except KeyboardInterrupt:
            logger.warning("Interrupted - shutting down workers...")
            raise
        finally:
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
            # Cleanup
            del future_to_file
            gc.collect()

        return documents, stats

    def _get_code_files(self, project_root: str) -> List[str]:
        """Get list of code files from git project.

        Uses git ls-files to respect .gitignore patterns.

        Args:
            project_root: Path to git repository root

        Returns:
            List of absolute paths to code files
        """
        # Get all tracked files from git
        all_files = self.git_discovery.get_tracked_files(project_root)

        # Filter by code extensions
        code_files = [
            f for f in all_files
            if Path(f).suffix.lower() in self.CODE_EXTENSIONS
        ]

        return code_files

    def _build_document(
        self,
        defn: CodeDefinition,
        git_metadata: GitMetadata,
        identifier: str,
        file_path: str
    ) -> Dict[str, Any]:
        """Build MeiliSearch document from CodeDefinition.

        Creates a document with the full metadata schema defined in RDR-011.

        Args:
            defn: Code definition (function/class/method/module)
            git_metadata: Git metadata for this project
            identifier: Composite identifier (project#branch)
            file_path: Absolute path to source file

        Returns:
            MeiliSearch document dictionary
        """
        # Generate unique ID: identifier:file_path:qualified_name:start_line
        # Sanitize for MeiliSearch ID constraints (alphanumeric, hyphen, underscore)
        # Note: identifier contains '#' which needs to be replaced
        sanitized_identifier = re.sub(r'[^a-zA-Z0-9_-]', '_', identifier)[:100]
        file_rel = os.path.relpath(file_path, git_metadata.project_root)
        sanitized_path = re.sub(r'[^a-zA-Z0-9_-]', '_', file_rel)[:200]
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', defn.qualified_name)[:100]
        doc_id = f"{sanitized_identifier}_{sanitized_path}_{sanitized_name}_{defn.start_line}"
        doc_id = re.sub(r'_+', '_', doc_id)  # Collapse multiple underscores
        doc_id = doc_id[:511]  # MeiliSearch ID limit

        # Detect programming language
        file_ext = Path(file_path).suffix.lower()
        language = self.ast_extractor.LANGUAGE_MAP.get(file_ext, 'unknown')

        return {
            # Primary key
            'id': doc_id,

            # Searchable content
            'content': defn.content,
            'function_name': defn.name if defn.code_type in ['function', 'method'] else None,
            'class_name': defn.name if defn.code_type == 'class' else None,
            'qualified_name': defn.qualified_name,
            'filename': os.path.basename(file_path),

            # Git metadata (from RDR-005 pattern)
            'git_project_identifier': identifier,
            'git_project_name': git_metadata.project_name,
            'git_branch': git_metadata.branch,
            'git_commit_hash': git_metadata.commit_hash,
            'git_remote_url': git_metadata.remote_url,
            'file_path': file_path,

            # Location metadata (function/class-level)
            'start_line': defn.start_line,
            'end_line': defn.end_line,
            'line_count': defn.line_count,
            'code_type': defn.code_type,

            # Language
            'programming_language': language,
            'file_extension': file_ext,
        }

    def _upload_batch(self, documents: List[Dict[str, Any]]):
        """Upload a batch of documents to MeiliSearch.

        Uses synchronous upload to wait for completion and enable
        early error detection.

        Args:
            documents: List of document dictionaries to upload
        """
        if not documents:
            return

        try:
            self.meili_client.add_documents_sync(
                index_name=self.index_name,
                documents=documents,
                timeout_ms=120000  # 2 minutes for large batches
            )
            logger.debug(f"Uploaded batch of {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to upload batch: {e}")
            raise

    def index_single_project(
        self,
        project_root: str,
        force: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Index a single git repository.

        Convenience method for indexing a specific project without
        directory discovery.

        Args:
            project_root: Path to git repository root
            force: Force reindex (ignore change detection)
            verbose: Show detailed progress

        Returns:
            Dict with indexing statistics
        """
        stats = {
            'indexed_projects': 0,
            'skipped_projects': 0,
            'failed_projects': 0,
            'total_files': 0,
            'indexed_files': 0,
            'total_definitions': 0,
            'errors': [],
        }

        try:
            # Extract git metadata
            git_metadata = self.git_discovery.extract_metadata(project_root)
            if not git_metadata:
                stats['failed_projects'] = 1
                stats['errors'].append({
                    'project': project_root,
                    'error': 'Failed to extract git metadata'
                })
                return stats

            identifier = git_metadata.identifier

            # Check if needs reindexing
            needs_indexing = force or self.sync.should_reindex_project(
                self.index_name, identifier, git_metadata.commit_hash
            )

            if not needs_indexing:
                stats['skipped_projects'] = 1
                return stats

            # Delete old documents if reindexing
            indexed_projects = self.sync.get_indexed_projects(self.index_name)
            if identifier in indexed_projects:
                self.sync.delete_project_documents(self.index_name, identifier)
                self.sync.clear_cache()

            # Index project (without progress bar for single project)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                project_stats = self._index_project(
                    project_root, git_metadata, identifier, verbose, progress
                )

            stats['indexed_projects'] = 1
            stats['total_files'] = project_stats['total_files']
            stats['indexed_files'] = project_stats['indexed_files']
            stats['total_definitions'] = project_stats['total_definitions']
            stats['errors'].extend(project_stats.get('errors', []))

        except Exception as e:
            stats['failed_projects'] = 1
            stats['errors'].append({
                'project': project_root,
                'error': str(e)
            })

        return stats

    def delete_project(self, project_identifier: str) -> int:
        """Delete all documents for a specific project/branch.

        Args:
            project_identifier: Composite identifier (e.g., "project#branch")

        Returns:
            Number of documents deleted
        """
        return self.sync.delete_project_documents(self.index_name, project_identifier)

    def get_indexed_projects(self) -> Dict[str, str]:
        """Get all indexed projects with their commit hashes.

        Returns:
            Dict mapping git_project_identifier -> git_commit_hash
        """
        return self.sync.get_indexed_projects(self.index_name)
