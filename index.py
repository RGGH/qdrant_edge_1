# index.py
from pathlib import Path

from main import (
    edge_shard,
    text_model,
    vision_model,
    VECTOR_NAME,
)

from embeddings import add_text, add_image


add_text(
    edge_shard=edge_shard,
    text_model=text_model,
    vector_name=VECTOR_NAME,
    text="hello world",
    point_id=1,
)

add_image(
    edge_shard=edge_shard,
    vision_model=vision_model,
    vector_name=VECTOR_NAME,
    path=Path("images") / "temp.jpg",
    point_id="26744dc7-f342-4497-9863-dcbb1b46551d",
)

print("Indexed")