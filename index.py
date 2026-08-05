# index.py
from pathlib import Path
import time
import uuid

from setup import edge_shard, text_model, vision_model, VECTOR_NAME
from embeddings import add_text, add_image

IMAGES_DIR = Path("images")


def stable_id(path: Path) -> str:
    return str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        str(path.resolve())
    ))


def main():
    start_time = time.perf_counter()

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
        img_start = time.perf_counter()
        add_image(
            edge_shard=edge_shard,
            vision_model=vision_model,
            vector_name=VECTOR_NAME,
            path=path,
            point_id=stable_id(path),
        )
        img_elapsed = time.perf_counter() - img_start
        print(f"Indexed {path} ({img_elapsed:.2f}s)")

    total_elapsed = time.perf_counter() - start_time
    print(f"\nIndexed {len(image_paths)} image(s) from {IMAGES_DIR}/ in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()