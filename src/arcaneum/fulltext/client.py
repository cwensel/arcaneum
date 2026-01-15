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
        return result.get('results', [])

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
        primary_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add documents to an index and wait for completion.

        Args:
            index_name: Target index name
            documents: List of document dictionaries
            primary_key: Optional primary key field

        Returns:
            Task result
        """
        index = self.get_index(index_name)
        task = index.add_documents(documents, primary_key)
        result = self.client.wait_for_task(task.task_uid)
        return result

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
        return index.get_stats()

    def get_index_settings(self, index_name: str) -> Dict[str, Any]:
        """Get settings for an index."""
        index = self.get_index(index_name)
        return index.get_settings()

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
