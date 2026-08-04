# Local Image & Text Search with Qdrant Edge

A local, offline semantic search engine over your own images (and text) — no cloud,
no API keys. Embeddings are generated on-device with [FastEmbed](https://github.com/qdrant/fastembed)
using a CLIP model, and stored/queried with [Qdrant Edge](https://qdrant.tech/edge/), an
embedded, in-process vector engine (think "SQLite, but for vector search").

Because images and text share one CLIP embedding space, you can search images with a
text query, or find visually similar images with an image query — same index, either
direction.

## How it works

```
Image / Text → FastEmbed (CLIP) → 512-d vector → Qdrant Edge shard → Ranked results
```

Two entry points share one setup:

- **`index.py`** — embeds every `.jpg` in `images/` (plus an example text doc) and
  writes them into the local shard. Safe to re-run.
- **`main.py`** — embeds a query (text or image) and asks the shard for its nearest
  neighbors.

## Project structure

```
setup.py        # loads the CLIP models + opens/creates the Qdrant Edge shard
                 # (shared state — imported by both scripts below, never run directly)
embeddings.py    # add_text / add_image / search_text / search_image helpers
index.py         # writes: embeds images/*.jpg + sample text into the shard
main.py          # reads: embeds a query and prints ranked search results
models/          # local CLIP model cache (downloaded once)
data/shard/      # the Qdrant Edge shard's on-disk storage
images/          # source images to index (*.jpg)
```

`setup.py` is the shared foundation — both `index.py` and `main.py` import
`edge_shard`, `text_model`, `vision_model`, and `VECTOR_NAME` from it. Neither script
imports from the other, so running one never has side effects on the other.

## Requirements

- Python (managed here with [uv](https://docs.astral.sh/uv/))
- [`fastembed`](https://pypi.org/project/fastembed/)
- [`qdrant-edge`](https://qdrant.tech/documentation/edge/) Python bindings

CLIP models are downloaded once into `./models` on first run; after that, both scripts
load them with `local_files_only=True`, so no network access is needed to run the app.

## Usage

**1. Add your images**

Drop any number of `.jpg` files into `images/`. No manual registration needed —
`index.py` picks up everything in the directory automatically.

**2. Build the index**

```bash
uv run index.py
```

This embeds every image in `images/` (and a sample text doc, `"hello world"`) and
upserts them into the shard at `data/shard/`. Re-run this any time you add or change
images — see [Deterministic IDs](#deterministic-ids-safe-to-re-run) below for why this
is safe to do repeatedly.

**3. Search**

```bash
uv run main.py
```

Runs both a text search (`"hello world"`) and an image search (`images/temp.jpg`) and
prints ranked results:

```
=== Text search ===
score=1.0000 payload={'type': 'text', 'text': 'hello world'}
score=0.2287 payload={'type': 'image', 'path': 'images/16.jpg'}

=== Image search ===
score=1.0000 payload={'type': 'image', 'path': 'images/temp.jpg'}
score=0.9847 payload={'type': 'image', 'path': 'images/15.jpg'}
score=0.9729 payload={'type': 'image', 'path': 'images/16.jpg'}
score=0.9496 payload={'type': 'image', 'path': 'images/20.jpg'}
score=0.9438 payload={'type': 'image', 'path': 'images/22.jpg'}
```

## Deterministic IDs (safe to re-run)

Every point stored in the shard needs a `point_id`. Left unset, `add_image`/`add_text`
fall back to `str(uuid.uuid4())` — a fresh random ID every call, even for the exact
same file. Re-running the indexer would then insert duplicate copies of every image on
every run, flooding search results with near-identical self-matches.

`index.py` avoids this by hashing each file's path into a **stable, deterministic**
UUID:

```python
def stable_id(path: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))
```

`uuid.uuid5(namespace, name)` always produces the same UUID for the same
`(namespace, name)` pair. So the same image path always maps to the same `point_id`,
and re-running `index.py` **upserts** (overwrites) the existing point instead of
creating a new one — indexing is idempotent.

## Adding new images

Just add `.jpg` files to `images/` and re-run `uv run index.py`. Indexing uses:

```python
image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
```

which picks up every `.jpg` directly inside `images/`. (Swap for `**/*.jpg` if you
want to index subfolders too, or broaden the glob if your files use `.jpeg`/`.png`.)

## Notes

- All models and data stay on disk — nothing is sent over the network after the
  initial model download.
- Text and image embeddings share the same 512-dimensional CLIP space
  (`Qdrant/clip-ViT-B-32-text` / `Qdrant/clip-ViT-B-32-vision`), so `search_text` and
  `search_image` both return results ranked by cosine similarity against the same
  index.
- Search results print raw payload dicts (`path`, `type`, etc.) — the underlying
  image paths are just filesystem paths, so they can be opened directly (some
  terminals, like Kitty, iTerm2, or WezTerm, even support making these clickable via
  OSC 8 terminal hyperlinks).