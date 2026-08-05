# setup.py
from pathlib import Path

from fastembed import ImageEmbedding, TextEmbedding
from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    # ScalarQuantizationConfig,
    # ScalarType,
    TurboQuantQuantizationConfig,
    TurboQuantBitSize,
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
    },
    quantization_config=TurboQuantQuantizationConfig(
        bits=TurboQuantBitSize.Bits2, 
        always_ram=True,
    ),
)

SHARD_DIR.mkdir(parents=True, exist_ok=True)

if (SHARD_DIR / "edge_config.json").exists():
    edge_shard = EdgeShard.load(str(SHARD_DIR))
else:
    edge_shard = EdgeShard.create(str(SHARD_DIR), config)