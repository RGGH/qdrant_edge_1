# index.py
from pathlib import Path
import uuid

from setup import edge_shard, text_model, vision_model, VECTOR_NAME
from embeddings import add_text, add_image

IMAGES_DIR = Path("images")


def stable_id(path: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))


def main():
    add_text(
        edge_shard=edge_shard,
        text_model=text_model,
        vector_name=VECTOR_NAME,
        text="hello world",
        point_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "hello world")),
    )

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))

    if not image_paths:
        print(f"No .jpg files found in {IMAGES_DIR}/")
        return

    for path in image_paths:
        add_image(
            edge_shard=edge_shard,
            vision_model=vision_model,
            vector_name=VECTOR_NAME,
            path=path,
            point_id=stable_id(path),
        )
        print(f"Indexed {path}")

    print(f"\nIndexed {len(image_paths)} image(s) from {IMAGES_DIR}/")


if __name__ == "__main__":
    main()