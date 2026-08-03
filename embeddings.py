# embedding.py
from pathlib import Path
import uuid
from qdrant_edge import (
    Point,
    UpdateOperation,
    SearchRequest,
    Query,
)



def add_text(
    edge_shard,
    text_model,
    vector_name: str,
    text: str,
    point_id: str | None = None,
):
    """Embed and store a text document."""

    embedding = next(text_model.embed([text]))

    point = Point(
        id=point_id or str(uuid.uuid4()),
        vector={vector_name: embedding.tolist()},
        payload={
            "type": "text",
            "text": text,
        },
    )

    edge_shard.update(UpdateOperation.upsert_points([point]))
    return point.id


def add_image(
    edge_shard,
    vision_model,
    vector_name: str,
    path: str | Path,
    point_id: str | None = None,
):
    """Embed and store an image."""

    path = Path(path)
    embedding = list(vision_model.embed([path]))[0]

    point = Point(
        id=point_id or str(uuid.uuid4()),
        vector={vector_name: embedding.tolist()},
        payload={
            "type": "image",
            "path": str(path),
        },
    )

    edge_shard.update(UpdateOperation.upsert_points([point]))
    return point.id

def search_text(
    edge_shard,
    text_model,
    vector_name: str,
    query: str,
    limit: int = 5,
):
    embedding = next(text_model.embed([query]))

    search_query = Query.Nearest(
        embedding.tolist(),
        using=vector_name,
    )

    request = SearchRequest(
        query=search_query,
        limit=limit,
        with_payload=True,
        with_vector=False,
    )

    return edge_shard.search(request)


def search_image(
    edge_shard,
    vision_model,
    vector_name: str,
    path,
    limit: int = 5,
):
    embedding = list(vision_model.embed([Path(path)]))[0]

    search_query = Query.Nearest(
        embedding.tolist(),
        using=vector_name,
    )

    request = SearchRequest(
        query=search_query,
        limit=limit,
        with_payload=True,
        with_vector=False,
    )

    return edge_shard.search(request)