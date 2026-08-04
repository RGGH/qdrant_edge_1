# main.py
from setup import edge_shard, vision_model, text_model, VECTOR_NAME
from embeddings import search_text, search_image


def main():
    print("\n=== Text search ===")
    results = search_text(
        edge_shard=edge_shard,
        text_model=text_model,
        vector_name=VECTOR_NAME,
        query="hello world",
    )
    for hit in results:
        print(f"score={hit.score:.4f} payload={hit.payload}")

    print("\n=== Image search ===")
    results = search_image(
        edge_shard=edge_shard,
        vision_model=vision_model,
        vector_name=VECTOR_NAME,
        path="images/temp.jpg",
    )
    for hit in results:
        print(f"score={hit.score:.4f} payload={hit.payload}")


if __name__ == "__main__":
    main()