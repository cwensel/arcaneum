"""
Main orchestration pipeline for git-aware source code indexing.

This module integrates all components to provide end-to-end source code
indexing with metadata-based sync (RDR-005 with RDR-006 progress enhancements).
"""

import logging
import os
from typing import List, Optional, Set
from pathlib import Path

import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from fastembed import TextEmbedding

from .git_operations import GitProjectDiscovery
from .git_metadata_sync import GitMetadataSync
from .ast_chunker import ASTCodeChunker
from .qdrant_indexer import QdrantIndexer
from .types import CodeChunk, CodeChunkMetadata, GitMetadata

logger = logging.getLogger(__name__)
console = Console()


class SourceCodeIndexer:
    """Orchestrates end-to-end source code indexing with metadata-based sync.

    Workflow:
    1. Query Qdrant for already-indexed projects (source of truth)
    2. Discover git projects in input directory
    3. For each project:
       - Check if re-indexing needed (commit changed)
       - If changed: delete old chunks (filter-based), re-index
       - If unchanged: skip
    4. Process files: AST chunk → embed → upload
    5. Report statistics

    Follows RDR-005 design with Qdrant as single source of truth.
    """

    def __init__(
        self,
        qdrant_indexer: QdrantIndexer,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 400,
        extensions: Optional[List[str]] = None,
        vector_name: Optional[str] = None
    ):
        """Initialize source code indexer.

        Args:
            qdrant_indexer: Configured QdrantIndexer
            embedding_model: FastEmbed model name
            chunk_size: Target chunk size in tokens (400 for 8K, 2K-4K for 32K models)
            extensions: File extensions to index (None = default list)
            vector_name: Name of vector if using named vectors (e.g., "stella")
        """
        self.qdrant_indexer = qdrant_indexer
        self.git_discovery = GitProjectDiscovery()
        self.chunker = ASTCodeChunker(chunk_size=chunk_size)
        self.sync = GitMetadataSync(qdrant_indexer.client)

        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embedder = TextEmbedding(model_name=embedding_model)
        self.vector_name = vector_name

        # Default extensions (15+ languages from RDR-005)
        self.extensions = extensions or [
            ".py", ".java", ".js", ".jsx", ".ts", ".tsx",
            ".cs", ".go", ".rs", ".c", ".h", ".cpp", ".hpp",
            ".php", ".rb", ".kt", ".scala", ".swift"
        ]

        # Statistics
        self.stats = {
            "projects_discovered": 0,
            "projects_indexed": 0,
            "projects_skipped": 0,
            "files_processed": 0,
            "chunks_created": 0,
            "chunks_uploaded": 0,
        }

    def index_directory(
        self,
        input_path: str,
        collection_name: str,
        depth: Optional[int] = None,
        force: bool = False,
        show_progress: bool = True,
        verbose: bool = False
    ) -> dict:
        """Index all git repositories in directory with metadata-based sync.

        Args:
            input_path: Directory containing git repositories
            collection_name: Qdrant collection name
            depth: Maximum depth to search for repos (None = unlimited)
            force: If True, bypass incremental sync and reindex all
            show_progress: If True, show progress with rich
            verbose: If True, show detailed output

        Returns:
            Dictionary with indexing statistics

        Example:
            >>> indexer = SourceCodeIndexer(qdrant_indexer)
            >>> stats = indexer.index_directory(
            ...     "/home/code",
            ...     "my-code",
            ...     depth=2
            ... )
            >>> print(f"Indexed {stats['projects_indexed']} projects")
        """
        # Show configuration at start
        console.print(f"\n[bold blue]Source Code Indexing Configuration[/bold blue]")
        console.print(f"  Collection: {collection_name} (type: code)")
        console.print(f"  Embedding: {self.embedding_model_name}")
        if self.vector_name:
            console.print(f"  Vector: {self.vector_name}")
        console.print(f"  Pipeline: Git Discover → AST Chunk → Embed (batched) → Upload")
        if depth is not None:
            console.print(f"  Depth: {depth}")
        if force:
            console.print(f"  Mode: Force reindex")
        # Check if offline mode set via environment
        if os.environ.get('HF_HUB_OFFLINE') == '1':
            console.print(f"  [yellow]Mode: Offline (HF_HUB_OFFLINE=1)[/yellow]")
        console.print()

        # Step 1: Query Qdrant for indexed projects (source of truth)
        if not force:
            if verbose:
                console.print("[cyan]Querying indexed projects...[/cyan]")
            indexed_projects = self.sync.get_indexed_projects(collection_name)
            if verbose:
                console.print(f"Found {len(indexed_projects)} indexed combinations\n")
        else:
            if verbose:
                console.print("[yellow]Force mode: bypassing incremental sync[/yellow]\n")
            indexed_projects = {}

        # Step 2: Discover git projects
        if verbose:
            console.print(f"[cyan]Discovering git projects...[/cyan]")
        git_projects = self.git_discovery.find_git_projects(input_path, depth)
        self.stats["projects_discovered"] = len(git_projects)

        if not verbose:
            console.print(f"[INFO] Found {len(git_projects)} projects")
        else:
            console.print(f"Found {len(git_projects)} git project(s)\n")

        if not git_projects:
            console.print("[INFO] No git projects found")
            return self.stats

        # Step 3: Process each project
        projects_to_index = []

        if verbose:
            console.print("[cyan]Analyzing projects...[/cyan]")

        for project_root in git_projects:
            git_metadata = self.git_discovery.extract_metadata(project_root)

            if not git_metadata:
                logger.warning(f"Could not extract metadata from {project_root}, skipping")
                continue

            identifier = git_metadata.identifier

            # Check if needs indexing (query Qdrant metadata)
            needs_indexing = self.sync.should_reindex_project(
                collection_name,
                identifier,
                git_metadata.commit_hash
            )

            if not needs_indexing and not force:
                if verbose:
                    console.print(
                        f"  [green]✓[/green] {identifier} "
                        f"(commit {git_metadata.commit_hash[:12]} already indexed)"
                    )
                self.stats["projects_skipped"] += 1
                continue

            # Project needs indexing
            if identifier in indexed_projects:
                old_commit = indexed_projects[identifier].commit_hash
                if verbose:
                    console.print(
                        f"  [yellow]↻[/yellow] {identifier} "
                        f"(commit changed: {old_commit[:12]} → {git_metadata.commit_hash[:12]})"
                    )
                # Delete old chunks (filter-based, fast)
                self.qdrant_indexer.delete_branch_chunks(collection_name, identifier)
            else:
                if verbose:
                    console.print(f"  [blue]➕[/blue] {identifier} (new branch)")

            projects_to_index.append((project_root, git_metadata, identifier))

        if verbose:
            console.print()

        if not projects_to_index:
            console.print("[INFO] All projects up to date")
            return self.stats

        # Step 4: Index projects
        if not verbose:
            console.print(f"[INFO] Indexing {len(projects_to_index)} projects...")
        else:
            console.print(f"[cyan]Indexing {len(projects_to_index)} project(s)...[/cyan]\n")

        # Process projects with appropriate output level
        total_projects = len(projects_to_index)

        for idx, (project_root, git_metadata, identifier) in enumerate(projects_to_index, 1):
            if not verbose:
                # Progress with percentage (RDR-006 format)
                percentage = (idx / total_projects * 100)
                console.print(f"[INFO] Processing {idx}/{total_projects} ({percentage:.0f}%) {identifier}", end="")

            self._index_project(
                project_root,
                git_metadata,
                identifier,
                collection_name,
                verbose=verbose,
                project_num=idx,
                total_projects=total_projects
            )

            # Stats are now shown in _index_project final line

        # Report final statistics
        if verbose:
            console.print(f"\n[bold green]✓ Indexing complete![/bold green]")
            console.print(f"\nStatistics:")
            console.print(f"  Projects discovered: {self.stats['projects_discovered']}")
            console.print(f"  Projects indexed: {self.stats['projects_indexed']}")
            console.print(f"  Projects skipped: {self.stats['projects_skipped']}")
            console.print(f"  Files processed: {self.stats['files_processed']}")
            console.print(f"  Chunks created: {self.stats['chunks_created']}")
            console.print(f"  Chunks uploaded: {self.stats['chunks_uploaded']}")
        else:
            # RDR-006: Structured completion message
            console.print(
                f"\n[INFO] Complete: {self.stats['projects_indexed']} projects, "
                f"{self.stats['files_processed']} files, "
                f"{self.stats['chunks_uploaded']} chunks"
            )

        return self.stats

    def _index_project(
        self,
        project_root: str,
        git_metadata: GitMetadata,
        identifier: str,
        collection_name: str,
        verbose: bool = False,
        project_num: int = 1,
        total_projects: int = 1
    ):
        """Index a single project.

        Args:
            project_root: Path to git repository
            git_metadata: Git metadata
            identifier: Composite identifier (project#branch)
            collection_name: Qdrant collection name
            verbose: If True, show detailed progress
            project_num: Current project number (for progress display)
            total_projects: Total number of projects
        """
        # Track initial stats for this project
        initial_files = self.stats["files_processed"]
        initial_chunks = self.stats["chunks_created"]

        # Get tracked files
        files = self.git_discovery.get_tracked_files(project_root, self.extensions)

        if not files:
            logger.info(f"No files to index in {identifier}")
            return

        total_files = len(files)

        # Process files with progress updates
        all_chunks = []

        for file_idx, file_path in enumerate(files, 1):
            try:
                filename = os.path.basename(file_path)
                chunks_before_file = len(all_chunks)

                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                # Chunk with AST
                chunks = self.chunker.chunk_code(file_path, code)

                if not chunks:
                    continue

                # Create metadata for each chunk
                filename = os.path.basename(file_path)
                file_ext = Path(file_path).suffix
                language = self.chunker.detect_language(file_path) or "text"

                for idx, chunk in enumerate(chunks):
                    metadata = CodeChunkMetadata(
                        git_project_identifier=identifier,
                        file_path=file_path,
                        filename=filename,
                        file_extension=file_ext,
                        programming_language=language,
                        file_size=len(code),
                        line_count=code.count('\n') + 1,
                        chunk_index=idx,
                        chunk_count=len(chunks),
                        text_extraction_method=chunk.method,
                        git_project_root=git_metadata.project_root,
                        git_project_name=git_metadata.project_name,
                        git_branch=git_metadata.branch,
                        git_commit_hash=git_metadata.commit_hash,
                        git_remote_url=git_metadata.remote_url,
                        ast_chunked=(chunk.method != "line_based"),
                        embedding_model=self.embedding_model_name
                    )

                    code_chunk = CodeChunk(
                        content=chunk.content,
                        metadata=metadata
                    )

                    all_chunks.append(code_chunk)

                self.stats["files_processed"] += 1

                # Calculate chunks for this specific file
                file_chunks = len(all_chunks) - chunks_before_file

                # Show incremental file progress (every file with filename and its chunk count)
                if not verbose:
                    # Use sys.stdout for proper \r overwriting
                    # Show filename and how many chunks it generated
                    print(
                        f"\r[{project_num}/{total_projects}] {identifier}: "
                        f"{file_idx}/{total_files} files ({filename}: {file_chunks})" + " " * 30,
                        end="",
                        flush=True,
                        file=sys.stdout
                    )

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if not all_chunks:
            logger.info(f"No chunks created for {identifier}")
            return

        self.stats["chunks_created"] += len(all_chunks)

        # Generate embeddings in batches to avoid hangs
        # Process 100 chunks at a time to prevent FastEmbed from hanging
        EMBEDDING_BATCH_SIZE = 100
        all_embeddings = []

        total_chunks = len(all_chunks)
        for batch_start in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
            batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_chunks)
            batch_chunks = all_chunks[batch_start:batch_end]

            # Show embedding progress
            if not verbose:
                print(
                    f"\r[{project_num}/{total_projects}] {identifier}: "
                    f"→ embedding {batch_end}/{total_chunks} chunks" + " " * 30,
                    end="",
                    flush=True,
                    file=sys.stdout
                )

            # Embed this batch
            texts = [chunk.content for chunk in batch_chunks]
            batch_embeddings = list(self.embedder.embed(texts))
            all_embeddings.extend(batch_embeddings)

        # Attach embeddings to chunks
        for chunk, embedding in zip(all_chunks, all_embeddings):
            chunk.embedding = embedding.tolist()

        # Show upload progress
        if not verbose:
            print(
                f"\r[{project_num}/{total_projects}] {identifier}: "
                f"→ uploading ({total_files} files, {len(all_chunks)} chunks)" + " " * 20,
                end="",
                flush=True,
                file=sys.stdout
            )

        # Upload to Qdrant
        uploaded = self.qdrant_indexer.upload_chunks(
            collection_name,
            all_chunks,
            vector_name=self.vector_name
        )
        self.stats["chunks_uploaded"] += uploaded
        self.stats["projects_indexed"] += 1

        # Calculate files/chunks for this project
        project_files = self.stats["files_processed"] - initial_files
        project_chunks = self.stats["chunks_created"] - initial_chunks

        # Final status line (overwrites progress)
        if not verbose:
            print(
                f"\r[{project_num}/{total_projects}] {identifier}: "
                f"✓ ({project_files} files, {project_chunks} chunks)" + " " * 30,
                flush=True,
                file=sys.stdout
            )
            # Move to next line for next project
            print()
        else:
            logger.info(
                f"Indexed {identifier}: {project_files} files, "
                f"{project_chunks} chunks, {uploaded} uploaded"
            )

    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
