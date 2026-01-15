"""MeiliSearch client wrapper for Arcaneum (RDR-008)."""

import meilisearch
from typing import Dict, List, Optional, Any


class FullTextClient:
    """Manages MeiliSearch client with explicit configuration."""

    def __init__(self, url: str, api_key: Optional[str] = None):
        """
        Initialize MeiliSearch client.

        Args:
            url: MeiliSearch server URL (e.g., http://localhost:7700)
            api_key: Master key for authentication (required in production)
        """
        self.url = url
        self.client = meilisearch.Client(url, api_key)

    def create_index(
        self,
        name: str,
        primary_key: str = "id",
        settings: Optional[Dict[str, Any]] = None
    ) -> meilisearch.index.Index:
        """
        Create a new index with optional settings.

        Args:
            name: Index name (e.g., 'source-code', 'pdf-docs')
            primary_key: Primary key field name
            settings: Index settings (searchable/filterable attributes, etc.)

        Returns:
            Created index object
        """
        # Create index - returns TaskInfo object (not dict)
        task = self.client.create_index(name, {'primaryKey': primary_key})
        self.client.wait_for_task(task.task_uid)

        index = self.client.index(name)

        # Apply settings if provided
        if settings:
            task = index.update_settings(settings)
            self.client.wait_for_task(task.task_uid)

        return index

    def get_index(self, name: str) -> meilisearch.index.Index:
        """Get existing index by name."""
        return self.client.index(name)

    def index_exists(self, name: str) -> bool:
        """Check if an index exists."""
        try:
            self.client.get_index(name)
            return True
        except meilisearch.errors.MeilisearchApiError as e:
            if e.code == "index_not_found":
                return False
            raise

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all indexes with their configurations."""
        result = self.client.get_indexes()

        # Extract the list of indexes from the response
        if isinstance(result, dict):
            index_list = result.get('results', [])
        else:
            index_list = result

        # Convert Index objects to dicts
        indexes = []
        for idx in index_list:
            if hasattr(idx, 'uid'):
                # It's an Index object
                indexes.append({
                    'uid': idx.uid,
                    'primaryKey': getattr(idx, 'primary_key', None),
                    'createdAt': getattr(idx, 'created_at', None),
                    'updatedAt': getattr(idx, 'updated_at', None),
                })
            elif isinstance(idx, dict):
                indexes.append(idx)
        return indexes

    def delete_index(self, name: str) -> None:
        """Delete an index."""
        task = self.client.delete_index(name)
        self.client.wait_for_task(task.task_uid)

    def add_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        primary_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add documents to an index.

        Args:
            index_name: Target index name
            documents: List of document dictionaries
            primary_key: Optional primary key field

        Returns:
            Task information
        """
        index = self.get_index(index_name)
        task = index.add_documents(documents, primary_key)
        return {"task_uid": task.task_uid, "status": "enqueued"}

    def add_documents_sync(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        primary_key: Optional[str] = None,
        timeout_ms: int = 60000
    ) -> Dict[str, Any]:
        """
        Add documents to an index and wait for completion.

        Args:
            index_name: Target index name
            documents: List of document dictionaries
            primary_key: Optional primary key field
            timeout_ms: Timeout in milliseconds for waiting on task completion (default: 60s)

        Returns:
            Task result

        Raises:
            RuntimeError: If the task fails
        """
        index = self.get_index(index_name)
        task = index.add_documents(documents, primary_key)
        result = self.client.wait_for_task(task.task_uid, timeout_in_ms=timeout_ms)

        # Check if task failed
        status = getattr(result, 'status', None) or (result.get('status') if isinstance(result, dict) else None)
        if status == 'failed':
            error = getattr(result, 'error', None) or (result.get('error') if isinstance(result, dict) else None)
            error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
            raise RuntimeError(f"Document addition failed: {error_msg}")

        return result

    def add_documents_batch_parallel(
        self,
        index_name: str,
        document_batches: List[List[Dict[str, Any]]],
        primary_key: Optional[str] = None,
        timeout_ms: int = 120000
    ) -> Dict[str, Any]:
        """
        Add multiple document batches in parallel (enqueue all, then wait).

        This is faster than add_documents_sync for multiple batches because
        MeiliSearch can process tasks concurrently.

        Args:
            index_name: Target index name
            document_batches: List of document batches to upload
            primary_key: Optional primary key field
            timeout_ms: Timeout in milliseconds for waiting on all tasks (default: 120s)

        Returns:
            Dict with 'total_documents' and 'task_count'

        Raises:
            RuntimeError: If any task fails
        """
        if not document_batches:
            return {'total_documents': 0, 'task_count': 0}

        index = self.get_index(index_name)
        task_uids = []
        total_docs = 0

        # Enqueue all batches without waiting
        for batch in document_batches:
            if batch:
                task = index.add_documents(batch, primary_key)
                task_uids.append(task.task_uid)
                total_docs += len(batch)

        # Wait for all tasks to complete
        failed_tasks = []
        for task_uid in task_uids:
            result = self.client.wait_for_task(task_uid, timeout_in_ms=timeout_ms)
            status = getattr(result, 'status', None) or (result.get('status') if isinstance(result, dict) else None)
            if status == 'failed':
                error = getattr(result, 'error', None) or (result.get('error') if isinstance(result, dict) else None)
                error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
                failed_tasks.append(f"Task {task_uid}: {error_msg}")

        if failed_tasks:
            raise RuntimeError(f"Document addition failed: {'; '.join(failed_tasks)}")

        return {'total_documents': total_docs, 'task_count': len(task_uids)}

    def search(
        self,
        index_name: str,
        query: str,
        filter: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        attributes_to_highlight: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search an index.

        Args:
            index_name: Index to search
            query: Search query (use quotes for exact phrases)
            filter: Filter expression (e.g., 'language = python')
            limit: Maximum results
            offset: Number of results to skip (for pagination)
            attributes_to_highlight: Fields to highlight in results

        Returns:
            Search results with hits, processing time, etc.
        """
        index = self.get_index(index_name)

        search_params: Dict[str, Any] = {
            'limit': limit,
            'offset': offset,
        }
        if filter:
            search_params['filter'] = filter
        if attributes_to_highlight:
            search_params['attributesToHighlight'] = attributes_to_highlight

        return index.search(query, search_params)

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index."""
        index = self.get_index(index_name)
        stats = index.get_stats()
        # Convert Pydantic model to dict if needed
        if hasattr(stats, 'model_dump'):
            return stats.model_dump(by_alias=True)
        elif hasattr(stats, 'dict'):
            return stats.dict(by_alias=True)
        return stats

    def get_index_settings(self, index_name: str) -> Dict[str, Any]:
        """Get settings for an index."""
        index = self.get_index(index_name)
        settings = index.get_settings()
        # Convert Pydantic model to dict if needed
        if hasattr(settings, 'model_dump'):
            return settings.model_dump(by_alias=True)
        elif hasattr(settings, 'dict'):
            return settings.dict(by_alias=True)
        return settings

    def update_index_settings(
        self,
        index_name: str,
        settings: Dict[str, Any]
    ) -> None:
        """Update settings for an index."""
        index = self.get_index(index_name)
        task = index.update_settings(settings)
        self.client.wait_for_task(task.task_uid)

    def health_check(self) -> bool:
        """Check if MeiliSearch server is healthy."""
        try:
            health = self.client.health()
            return health.get('status') == 'available'
        except Exception:
            return False

    def get_version(self) -> Dict[str, str]:
        """Get MeiliSearch server version information."""
        return self.client.get_version()

    def get_stats(self) -> Dict[str, Any]:
        """Get global statistics."""
        return self.client.get_all_stats()

    def get_all_file_paths(self, index_name: str) -> set:
        """Get all unique file_path values from an index.

        Uses faceted search to efficiently get unique file paths.

        Args:
            index_name: Index to query

        Returns:
            Set of unique file_path strings
        """
        index = self.get_index(index_name)
        file_paths = set()

        # Use documents endpoint to get all unique file paths
        # MeiliSearch doesn't have aggregation, so we paginate through documents
        offset = 0
        limit = 1000

        while True:
            result = index.get_documents({
                'offset': offset,
                'limit': limit,
                'fields': ['file_path']
            })

            # Handle both dict and object results
            if hasattr(result, 'results'):
                docs = result.results
            else:
                docs = result.get('results', [])

            if not docs:
                break

            for doc in docs:
                if hasattr(doc, 'file_path'):
                    file_paths.add(doc.file_path)
                elif isinstance(doc, dict) and 'file_path' in doc:
                    file_paths.add(doc['file_path'])

            offset += limit

            # Check if we've reached the end
            if len(docs) < limit:
                break

        return file_paths
