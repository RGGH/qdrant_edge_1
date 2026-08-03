# main.py
from pathlib import Path

from fastembed import ImageEmbedding, TextEmbedding
from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    # SearchRequest
)

from embeddings import (
    add_text,
    add_image,
    search_text,
    search_image,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"
VISION_MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"

MODELS_DIR = Path("./models")
SHARD_DIR = Path("./data/shard")

VECTOR_NAME = "my-vector"
VECTOR_DIMENSION = 512

# ---------------------------------------------------------------------
# Load embedding models
# ---------------------------------------------------------------------

text_model = TextEmbedding(
    model_name=TEXT_MODEL_NAME,
    cache_dir=str(MODELS_DIR),
    local_files_only=True,
)

vision_model = ImageEmbedding(
    model_name=VISION_MODEL_NAME,
    cache_dir=str(MODELS_DIR),
    local_files_only=True,
)

# ---------------------------------------------------------------------
# Create/Open shard
# ---------------------------------------------------------------------

config = EdgeConfig(
    vectors={
        VECTOR_NAME: EdgeVectorParams(
            size=VECTOR_DIMENSION,
            distance=Distance.Cosine,
        )
    }
)

SHARD_DIR.mkdir(parents=True, exist_ok=True)

if (SHARD_DIR / "edge_config.json").exists():
    edge_shard = EdgeShard.load(str(SHARD_DIR))
else:
    edge_shard = EdgeShard.create(str(SHARD_DIR), config)

# ---------------------------------------------------------------------
# Insert data
# ---------------------------------------------------------------------

text_id = add_text(
    edge_shard=edge_shard,
    text_model=text_model,
    vector_name=VECTOR_NAME,
    text="hello world",

)

image_id = add_image(
    edge_shard=edge_shard,
    vision_model=vision_model,
    vector_name=VECTOR_NAME,
    path=Path("images") / "temp.jpg",
)

image_id = add_image(
    edge_shard=edge_shard,
    vision_model=vision_model,
    vector_name=VECTOR_NAME,
    path=Path("images") / "temp2.jpg",
)

# ---------------------------------------------------------------------
# Search using text
# ---------------------------------------------------------------------

print("\n=== Text search ===")

results = search_text(
    edge_shard=edge_shard,
    text_model=text_model,
    vector_name=VECTOR_NAME,
    query="hello world",
)

for hit in results:
    print(f"score={hit.score:.4f} payload={hit.payload}")

# ---------------------------------------------------------------------
# Search using image
# ---------------------------------------------------------------------

print("\n=== Image search ===")

results = search_image(
    edge_shard=edge_shard,
    vision_model=vision_model,
    vector_name=VECTOR_NAME,
    path="images/temp.jpg",
)

for hit in results:
    print(f"score={hit.score:.4f} payload={hit.payload}")