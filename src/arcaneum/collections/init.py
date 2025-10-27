"""Collection initialization module for Qdrant (RDR-002)."""

from qdrant_client import QdrantClient, models
from typing import Dict, List

# Collection configurations with named vectors architecture
COLLECTION_CONFIGS = {
    "source-code": {
        "vectors": {
            "stella": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "modernbert": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "bge": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "jina": models.VectorParams(size=768, distance=models.Distance.COSINE),
        },
        "hnsw_config": models.HnswConfigDiff(m=16, ef_construct=100),
        "on_disk_payload": True,
        "indexes": ["programming_language", "git_project_root", "file_extension"],
    },
    "pdf-docs": {
        "vectors": {
            "stella": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "modernbert": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "bge": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
        "hnsw_config": models.HnswConfigDiff(m=16, ef_construct=100),
        "on_disk_payload": True,
        "indexes": ["filename", "file_path"],
    },
}


def init_collections(url: str = "http://localhost:6333"):
    """Initialize all Arcaneum collections with named vectors.

    Args:
        url: Qdrant server URL

    Returns:
        List of created collection names
    """
    client = QdrantClient(url=url)
    created = []

    for name, config in COLLECTION_CONFIGS.items():
        print(f"Creating collection: {name}")

        client.create_collection(
            collection_name=name,
            vectors_config=config["vectors"],
            hnsw_config=config["hnsw_config"],
            on_disk_payload=config["on_disk_payload"],
        )

        # Create payload indexes for efficient filtering
        for field_name in config["indexes"]:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        print(f"✅ Created {name} with {len(config['vectors'])} vector types")
        created.append(name)

    return created


if __name__ == "__main__":
    init_collections()
