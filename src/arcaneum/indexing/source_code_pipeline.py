"""
Main orchestration pipeline for git-aware source code indexing.

This module integrates all components to provide end-to-end source code
indexing with metadata-based sync (RDR-005 with RDR-006 progress enhancements).
"""

import logging
import os
import time
from typing import List, Optional, Set
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .git_operations import GitProjectDiscovery
from ..embeddings.client import EmbeddingClient
from .git_metadata_sync import GitMetadataSync
from .ast_chunker import ASTCodeChunker
from .qdrant_indexer import QdrantIndexer
from .types import CodeChunk, CodeChunkMetadata, GitMetadata
from ..monitoring.cpu_stats import create_monitor
from ..monitoring.pipeline_profiler import PipelineProfiler

logger = logging.getLogger(__name__)
console = Console()


class EmbeddingTimingCollector:
    """Collects timing metrics during embedding batch processing."""

    def __init__(self):
        self.batch_times = []
        self.start_time = None
        self.total_batches = 0

    def start(self, total_batches: int):
        """Start timing collection."""
        self.start_time = time.time()
        self.total_batches = total_batches
        self.batch_times = []

    def record_batch(self, batch_idx: int, total_batches: int):
        """Record completion of a batch."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.batch_times.append(elapsed)

    def get_summary(self) -> dict:
        """Get timing summary."""
        if not self.batch_times:
            return {"total_time": 0, "avg_per_batch": 0, "num_batches": 0}

        total_time = self.batch_times[-1] if self.batch_times else 0
        num_batches = len(self.batch_times)
        avg_per_batch = total_time / num_batches if num_batches > 0 else 0

        return {
            "total_time": total_time,
            "avg_per_batch": avg_per_batch,
            "num_batches": num_batches
        }


def _process_file_worker(
    file_path: str,
    identifier: str,
    git_metadata: GitMetadata,
    embedding_model_id: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[CodeChunk]:
    """Process a single file: read, chunk, create metadata.

    This is a module-level function to support pickling for ProcessPoolExecutor.

    Args:
        file_path: Path to source file
        identifier: Git project identifier (project#branch)
        git_metadata: Git metadata for this project
        embedding_model_id: Embedding model identifier
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of CodeChunk objects with metadata (no embeddings yet)
    """
    # Set low priority UNLESS disabled by --not-nice flag (arcaneum-mql4)
    if os.environ.get('ARCANEUM_DISABLE_WORKER_NICE') != '1':
        try:
            if hasattr(os, 'nice'):
                os.nice(10)  # Background priority for code processing workers
        except Exception:
            pass  # Ignore if we can't set priority

    try:
        # Create chunker (thread-safe, lightweight)
        chunker = ASTCodeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Chunk with AST
        chunks = chunker.chunk_code(file_path, code)

        if not chunks:
            return []

        # Create metadata for each chunk
        filename = os.path.basename(file_path)
        file_ext = Path(file_path).suffix
        language = chunker.detect_language(file_path) or "text"

        code_chunks = []
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
                embedding_model=embedding_model_id
            )

            code_chunk = CodeChunk(
                content=chunk.content,
                metadata=metadata
            )

            code_chunks.append(code_chunk)

        return code_chunks

    except Exception as e:
        # Log error and return empty list
        logger.error(f"Error processing {file_path}: {e}")
        return []


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
        embedding_client: EmbeddingClient,
        embedding_model_id: str,
        chunk_size: int = 400,
        chunk_overlap: int = 20,
        extensions: Optional[List[str]] = None,
        vector_name: Optional[str] = None,
        parallel_workers: Optional[int] = None,
        embedding_workers: int = 4,
        embedding_batch_size: int = 512
    ):
        """Initialize source code indexer.

        Args:
            qdrant_indexer: Configured QdrantIndexer
            embedding_client: EmbeddingClient instance (with GPU support if enabled)
            embedding_model_id: Model identifier for EmbeddingClient (e.g., "jina-code", "stella")
            chunk_size: Target chunk size in tokens (400 for 8K, 2K-4K for 32K models)
            chunk_overlap: Overlap between chunks in tokens (default: 20)
            extensions: File extensions to index (None = default list)
            vector_name: Name of vector if using named vectors (e.g., "stella")
            parallel_workers: Number of parallel workers for file processing (None = cpu_count // 2)
            embedding_workers: Number of parallel workers for embedding generation (default: 4)
            embedding_batch_size: Batch size for embedding generation (default: 512, GPU-optimal per arcaneum-i7oa, arcaneum-2m1i)
        """
        self.qdrant_indexer = qdrant_indexer
        self.git_discovery = GitProjectDiscovery()
        self.chunker = ASTCodeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.sync = GitMetadataSync(qdrant_indexer.client)

        # Use provided embedding client
        self.embedding_client = embedding_client
        self.embedding_model_id = embedding_model_id
        self.vector_name = vector_name
        self.chunk_size = chunk_size

        # Configure parallel workers (default: cpu_count // 2 for responsive laptop)
        if parallel_workers is None:
            self.parallel_workers = max(1, cpu_count() // 2)
        else:
            self.parallel_workers = max(1, parallel_workers)

        # Configure embedding parallelism
        self.embedding_workers = max(1, embedding_workers)
        self.embedding_batch_size = max(1, embedding_batch_size)

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
            "projects_empty": 0,
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
        verbose: bool = False,
        file_list: Optional[List] = None,
        profile: bool = False
    ) -> dict:
        """Index all git repositories in directory with metadata-based sync.

        Args:
            input_path: Directory containing git repositories (or base directory for file_list)
            collection_name: Qdrant collection name
            depth: Maximum depth to search for repos (None = unlimited)
            force: If True, bypass incremental sync and reindex all
            show_progress: If True, show progress with rich
            verbose: If True, show detailed output
            file_list: Optional list of source files to index (Note: Limited support for code indexing.
                      Code indexing is git-centric and works best with git repositories.)
            profile: If True, show pipeline performance profiling (stage breakdown, throughput)

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
        # Start CPU monitoring (RDR-013 Phase 1)
        cpu_monitor = create_monitor()
        if cpu_monitor:
            cpu_monitor.start()

        # Initialize pipeline profiler if requested
        profiler = PipelineProfiler() if profile else None
        if profiler:
            profiler.start()

        # Warn if file_list is provided (limited support for code indexing)
        if file_list is not None:
            console.print(
                "[yellow]⚠️  Warning: file_list is not fully supported for code indexing. "
                "Code indexing is git-centric and works best with git repositories.[/yellow]\n"
            )
            # For now, ignore file_list and proceed with git discovery
            # Future enhancement: could support standalone file indexing
            file_list = None

        # Show configuration at start
        console.print(f"\n[bold blue]Source Code Indexing Configuration[/bold blue]")
        console.print(f"  Collection: {collection_name} (type: code)")
        console.print(f"  Embedding: {self.embedding_model_id}")
        if self.vector_name:
            console.print(f"  Vector: {self.vector_name}")

        # Show device info (GPU enabled by default)
        device_info = self.embedding_client.get_device_info()
        if not device_info['gpu_enabled']:
            console.print(f"  Device: CPU (GPU acceleration disabled)")
        elif device_info['gpu_available']:
            console.print(f"  [green]Device: {device_info['device'].upper()} (GPU acceleration enabled)[/green]")
        else:
            console.print(f"  Device: CPU (GPU not available)")

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
                total_projects=total_projects,
                profiler=profiler
            )

            # Stats are now shown in _index_project final line

        # Stop profiler
        if profiler:
            profiler.stop()

        # Report final statistics
        if verbose:
            console.print(f"\n[bold green]✓ Indexing complete![/bold green]")
            console.print(f"\nStatistics:")
            console.print(f"  Projects discovered: {self.stats['projects_discovered']}")
            console.print(f"  Projects indexed: {self.stats['projects_indexed']}")
            console.print(f"  Projects skipped: {self.stats['projects_skipped']} (already up to date)")
            if self.stats['projects_empty'] > 0:
                console.print(f"  Projects empty: {self.stats['projects_empty']} (no indexable files)")
            console.print(f"  Files processed: {self.stats['files_processed']}")
            console.print(f"  Chunks created: {self.stats['chunks_created']}")
            console.print(f"  Chunks uploaded: {self.stats['chunks_uploaded']}")

            # Show CPU statistics (RDR-013 Phase 1)
            if cpu_monitor:
                console.print(f"\nPerformance:")
                stats = cpu_monitor.get_stats()
                console.print(f"  CPU usage: {stats['cpu_percent']:.1f}% total ({stats['cpu_percent_per_core']:.1f}% per core)")
                console.print(f"  Threads: {stats['num_threads']} | Cores: {stats['num_cores']}")
                console.print(f"  Elapsed: {stats['elapsed_time']:.1f}s")

            # Show pipeline profiling report if enabled
            if profiler:
                console.print(f"\n{profiler.report()}")
        else:
            # RDR-006: Structured completion message
            msg = f"\n[INFO] Complete: {self.stats['projects_indexed']} projects, "
            msg += f"{self.stats['files_processed']} files, "
            msg += f"{self.stats['chunks_uploaded']} chunks"

            # Add context if projects were processed but empty
            if self.stats['projects_empty'] > 0:
                msg += f" ({self.stats['projects_empty']} projects had no indexable files)"

            console.print(msg)

            # Show CPU stats in compact mode (RDR-013 Phase 1)
            if cpu_monitor:
                console.print(f"[INFO] {cpu_monitor.get_summary()}")

            # Show pipeline profiling in compact mode if enabled
            if profiler:
                console.print(f"[INFO] {profiler.get_compact_summary()}")

        return self.stats

    def _index_project(
        self,
        project_root: str,
        git_metadata: GitMetadata,
        identifier: str,
        collection_name: str,
        verbose: bool = False,
        project_num: int = 1,
        total_projects: int = 1,
        profiler: Optional[PipelineProfiler] = None
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
            profiler: Optional pipeline profiler for stage timing
        """
        # Track initial stats for this project
        initial_files = self.stats["files_processed"]
        initial_chunks = self.stats["chunks_created"]

        # Get tracked files
        files = self.git_discovery.get_tracked_files(project_root, self.extensions)

        if not files:
            ext_list = ", ".join(self.extensions[:5]) + ("..." if len(self.extensions) > 5 else "")
            logger.info(f"No indexable files in {identifier} (looking for: {ext_list})")
            self.stats["projects_empty"] += 1
            if not verbose:
                print(
                    f"\r[{project_num}/{total_projects}] {identifier}: "
                    f"⊘ (no indexable files)" + " " * 30,
                    flush=True,
                    file=sys.stdout
                )
                print()
            return

        total_files = len(files)

        # Show project header in verbose mode
        if verbose:
            console.print(f"\n[{project_num}/{total_projects}] {identifier}")
            console.print(f"  → processing files ({total_files} files found)")

        # Process files in parallel using ProcessPoolExecutor (RDR-013 Phase 2)
        # Default: cpu_count // 2 for responsive laptop
        all_chunks = []
        files_processed = 0

        # Show initial progress
        if not verbose:
            print(
                f"\r[{project_num}/{total_projects}] {identifier}: "
                f"→ processing {total_files} files (parallel, {self.parallel_workers} workers)" + " " * 20,
                end="",
                flush=True,
                file=sys.stdout
            )

        # Track file processing time for profiler
        file_processing_start = time.time()

        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all file processing jobs
            future_to_file = {}
            for file_path in files:
                future = executor.submit(
                    _process_file_worker,
                    file_path,
                    identifier,
                    git_metadata,
                    self.embedding_model_id,
                    self.chunk_size,
                    self.chunker.chunk_overlap  # Pass chunk_overlap from initialized chunker
                )
                future_to_file[future] = file_path

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                filename = os.path.basename(file_path)

                try:
                    file_chunks = future.result()
                    if file_chunks:
                        all_chunks.extend(file_chunks)
                        files_processed += 1

                        # Show incremental progress
                        if not verbose:
                            print(
                                f"\r[{project_num}/{total_projects}] {identifier}: "
                                f"{files_processed}/{total_files} files ({filename}: {len(file_chunks)})    ",
                                end="",
                                flush=True,
                                file=sys.stdout
                            )
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

        # Memory cleanup: Clear futures dictionary after ProcessPoolExecutor (arcaneum-b8lg)
        # Future objects hold references to worker results that prevent GC
        del future_to_file

        # Record file processing stage timing
        if profiler:
            profiler.record_stage("file_processing", time.time() - file_processing_start, files_processed)

        self.stats["files_processed"] += files_processed

        # Show chunking status in verbose mode
        if verbose:
            console.print(f"  → chunking ({len(all_chunks)} chunks created)")

        if not all_chunks:
            logger.info(f"No chunks created for {identifier}")
            return

        self.stats["chunks_created"] += len(all_chunks)

        # Generate embeddings in parallel batches (Phase 2 RDR-013)
        total_chunks = len(all_chunks)

        # Show embedding progress
        if verbose:
            console.print(f"  → embedding ({total_chunks} chunks)")
        else:
            print(
                f"\r[{project_num}/{total_projects}] {identifier}: "
                f"→ embedding {total_chunks} chunks (parallel)" + " " * 30,
                end="",
                flush=True,
                file=sys.stdout
            )

        # Create timing collector for verbose mode
        timing_collector = None
        if verbose:
            timing_collector = EmbeddingTimingCollector()
            total_batches = (total_chunks + self.embedding_batch_size - 1) // self.embedding_batch_size
            timing_collector.start(total_batches)

        # Track embedding time for profiler
        embedding_start = time.time()

        # Embed all texts in parallel
        texts = [chunk.content for chunk in all_chunks]
        all_embeddings = self.embedding_client.embed_parallel(
            texts,
            self.embedding_model_id,
            max_workers=self.embedding_workers,
            batch_size=self.embedding_batch_size,
            progress_callback=timing_collector.record_batch if timing_collector else None
        )

        # Record embedding stage timing
        if profiler:
            profiler.record_stage("embedding", time.time() - embedding_start, total_chunks)

        # Show timing summary in verbose mode
        if verbose and timing_collector:
            timing = timing_collector.get_summary()
            console.print(
                f"     embedded {total_chunks} chunks in {timing['total_time']:.2f}s "
                f"({timing['num_batches']} batches, {timing['avg_per_batch']:.2f}s/batch)"
            )

        # Attach embeddings to chunks
        for chunk, embedding in zip(all_chunks, all_embeddings):
            chunk.embedding = embedding  # Already a list from EmbeddingClient

        # Show upload progress
        if verbose:
            console.print(f"  → uploading ({total_files} files, {len(all_chunks)} chunks)")
        else:
            print(
                f"\r[{project_num}/{total_projects}] {identifier}: "
                f"→ uploading ({total_files} files, {len(all_chunks)} chunks)" + " " * 20,
                end="",
                flush=True,
                file=sys.stdout
            )

        # Track upload time for profiler
        upload_start = time.time()

        # Upload to Qdrant with bulk mode for 1.3-1.5x speedup (RDR-013 Phase 1)
        # Bulk mode disables HNSW indexing during upload, rebuilds after completion
        uploaded = self.qdrant_indexer.upload_chunks(
            collection_name,
            all_chunks,
            vector_name=self.vector_name,
            bulk_mode=True
        )

        # Record upload stage timing
        if profiler:
            profiler.record_stage("upload", time.time() - upload_start, uploaded)

        self.stats["chunks_uploaded"] += uploaded
        self.stats["projects_indexed"] += 1

        # Memory cleanup: Release large data structures after upload (arcaneum-b8lg)
        # These can hold 100MB+ for large projects, preventing GC between projects
        del all_chunks
        del texts
        del all_embeddings
        import gc
        gc.collect()

        # Calculate files/chunks for this project
        project_files = self.stats["files_processed"] - initial_files
        project_chunks = self.stats["chunks_created"] - initial_chunks

        # Final status line
        if verbose:
            console.print(f"  ✓ complete ({project_files} files, {project_chunks} chunks)")
        else:
            print(
                f"\r[{project_num}/{total_projects}] {identifier}: "
                f"✓ ({project_files} files, {project_chunks} chunks)" + " " * 30,
                flush=True,
                file=sys.stdout
            )
            print()

    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0
