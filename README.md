# qdrant-edge-1

A minimal example project demonstrating [Qdrant Edge](https://qdrant.tech/) — an embedded, in-process vector search engine — combined with [FastEmbed](https://github.com/qdrant/fastembed) CLIP models for text-to-image and image-to-image semantic search.

## What Is Qdrant Edge?

Qdrant Edge is a lightweight, embedded vector search engine for in-process retrieval with a minimal memory footprint and no background services. It's designed for applications requiring low-latency vector search in environments with limited or intermittent connectivity, such as robots, kiosks, home assistants, and mobile phones.

Unlike Qdrant Server, which uses a client-server architecture, Qdrant Edge runs inside the application process — think of it as SQLite, but for vector search. Data is stored and queried locally, ensuring low-latency access and enhanced privacy since data never has to leave the device. Qdrant Edge also provides APIs to synchronize data with a Qdrant server, so you can offload heavy computations like indexing to more powerful instances, back up and restore data, and centrally aggregate data from multiple edge devices.

## How This Example Works

This project embeds text and images into a shared **512-dimensional CLIP vector space** (`Qdrant/clip-ViT-B-32-text` / `Qdrant/clip-ViT-B-32-vision`), stores them in a local Qdrant Edge shard, and lets you search across both modalities — e.g. search images using a text query, or find similar images using an image query.

### Project structure

```
.
├── data/shard/            # Qdrant Edge shard storage (created at runtime)
├── images/                # Sample images used for indexing/search
│   ├── temp.jpg
│   └── temp2.jpg
├── models/                 # Local FastEmbed model cache (must be pre-downloaded)
├── embeddings.py            # add_text / add_image / search_text / search_image helpers
├── main.py                  # Loads models, creates/opens the shard, runs a demo insert + search
├── index.py                 # Standalone script to add more points to an existing shard
├── pyproject.toml
└── uv.lock
```

### Key files

- **`main.py`** — loads the text and vision embedding models, creates (or opens) the Qdrant Edge shard at `data/shard`, inserts a sample text point and two sample image points, then runs a text search and an image search.
- **`embeddings.py`** — reusable helper functions:
  - `add_text()` / `add_image()` — embed content and upsert it as a `Point` into the shard.
  - `search_text()` / `search_image()` — embed a query and run a nearest-neighbor `SearchRequest` against the shard.
- **`index.py`** — imports the already-initialized `edge_shard` and models from `main.py` to add extra points (including a fixed integer ID and a fixed UUID) without re-running the full demo.

## Prerequisites

- Python **3.12** or newer
- [`uv`](https://docs.astral.sh/uv/) for dependency management (recommended — a `uv.lock` is included)
- ~500 MB free disk space for the CLIP text/vision model weights

## Installation

1. **Clone / open the project directory**, then install dependencies with `uv`:

   ```bash
   uv sync
   ```

   This creates a virtual environment and installs `qdrant-edge-py` and its dependencies (including `fastembed`) as pinned in `uv.lock`.

   > No `uv`? You can use plain `pip` instead:
   > ```bash
   > python -m venv .venv
   > source .venv/bin/activate   # Windows: .venv\Scripts\activate
   > pip install qdrant-edge-py
   > ```

2. **Download the embedding models.**

   `main.py` loads models with `local_files_only=True`, which means the CLIP text and vision models must already be cached in `./models` before you run anything — FastEmbed won't fetch them on demand in this configuration. Download them once with:

   ```bash
   uv run python - <<'EOF'
   from fastembed import TextEmbedding, ImageEmbedding

   TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text", cache_dir="./models")
   ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision", cache_dir="./models")
   EOF
   ```

   This populates `models/models--Qdrant--clip-ViT-B-32-text` and `models/models--Qdrant--clip-ViT-B-32-vision`. You only need to do this once (or whenever you switch models).

3. **Add sample images.**

   Place at least two `.jpg` images in the `images/` folder named `temp.jpg` and `temp2.jpg` (or edit the paths in `main.py`).

## Running the Demo

With models cached and images in place, run:

```bash
uv run python main.py
```

On first run this will:

1. Create a new Qdrant Edge shard at `data/shard/` (subsequent runs will detect `edge_config.json` and reopen the existing shard instead of recreating it).
2. Embed and upsert a text point (`"hello world"`) and two image points (`temp.jpg`, `temp2.jpg`).
3. Run a **text search** for `"hello world"` and print the results.
4. Run an **image search** using `temp.jpg` and print the results.

Example output shape:

```
=== Text search ===
score=0.9999 payload={'type': 'text', 'text': 'hello world'}

=== Image search ===
score=1.0000 payload={'type': 'image', 'path': 'images/temp.jpg'}
score=0.8421 payload={'type': 'image', 'path': 'images/temp2.jpg'}
```

## Indexing Additional Points

`index.py` reuses the shard and models already initialized by `main.py` to add more points — including points with explicit IDs (an integer and a UUID) rather than auto-generated ones:

```bash
uv run python index.py
```

This will print `Indexed` once the additional text and image points have been upserted.

> **Note:** because `index.py` imports `edge_shard`, `text_model`, and `vision_model` directly from `main.py`, running it will also re-execute all of `main.py`'s top-level code (including the demo inserts and searches) before adding the new points.

## Resetting the Shard

The shard is just local files under `data/shard/`. To start fresh (e.g. after changing the vector size or distance metric), simply delete the directory:

```bash
rm -rf data/shard
```

The next run of `main.py` will detect that `edge_config.json` is missing and create a new shard.

## Configuration Reference

| Setting | Value | Where |
|---|---|---|
| Vector name | `my-vector` | `main.py` |
| Vector dimension | `512` | `main.py` |
| Distance metric | Cosine | `main.py` |
| Text model | `Qdrant/clip-ViT-B-32-text` | `main.py` |
| Vision model | `Qdrant/clip-ViT-B-32-vision` | `main.py` |
| Shard storage path | `./data/shard` | `main.py` |
| Model cache path | `./models` | `main.py` |

## Troubleshooting

- **`local_files_only` errors / model not found** — you skipped the model download step above, or `cache_dir` doesn't match where the models were downloaded. Re-run the download snippet in step 2.
- **Shard fails to load** — check that `data/shard/edge_config.json` exists and wasn't partially written (e.g. from an interrupted run). Delete the `data/shard` directory and let it recreate.
- **Image search returns no results / low scores** — confirm the image paths in `images/` match what was indexed (`add_image` stores the path in the payload, so mismatched relative paths between indexing and querying will still search correctly, but double-check the files exist).

A few things worth naming clearly here, since "Qdrant Edge" is genuinely new (beta, launched mid‑2025) and doesn't have published benchmark numbers of its own yet the way Qdrant Server does. So this is a mix of confirmed facts about the architecture and reasonable inference from how embedded vector engines behave in general — I'll flag which is which.

## Expected performance

Because Qdrant Edge runs **in-process** rather than over gRPC/HTTP to a separate server, you get:

- **No network round-trip.** Every query in our `main.py` demo is a local function call, so latency is dominated by the embedding step (CLIP inference), not the vector search itself. For small collections (hundreds to low thousands of points), search itself will likely be sub-millisecond; CLIP embedding on CPU will be the actual bottleneck, probably tens to low-hundreds of milliseconds per image/text depending on hardware.
- **No indexing overhead at small scale.** With a handful of points like our demo, Qdrant Edge is almost certainly doing something close to brute-force/flat search rather than building an HNSW graph — which is fine and fast at this size, but won't be representative of larger-scale performance.
- **Unknown large-scale ceiling.** For context, benchmarks show Qdrant Server's performance degrading beyond 10 million vectors, achieving only about 41 QPS at 50 million vectors and 99% recall — but that's the server engine, tuned for throughput under concurrent load, which isn't the design target for Edge. Edge is optimized for footprint and single-process latency on constrained devices, not high-QPS concurrent serving, so that number isn't directly transferable either way.

**Practical takeaway:** for a demo with 2–3 points, you won't observe anything meaningful about Edge's scaling behavior — you're really benchmarking CLIP inference time and Python overhead.

## Trade-offs (Edge vs. Server)

| | Qdrant Edge | Qdrant Server |
|---|---|---|
| Deployment | In-process, no daemon | Client-server, needs a running instance |
| Network dependency | None | Required (even localhost adds a hop) |
| Concurrency | Single-process; not built for many simultaneous clients | Built for concurrent multi-client load |
| Scale ceiling | Best for small-to-medium local datasets | Scales to distributed, billions of vectors |
| Privacy | Data never leaves the device | Data transits to server |
| Ops burden | ~Zero (like SQLite) | Requires running/monitoring a service |
| Sync | Optional push/pull to a server | N/A (it *is* the server) |

The core trade-off is the classic embedded-vs-server one: you give up horizontal scale and concurrent multi-client throughput in exchange for zero ops, zero network latency, and offline/privacy-preserving operation.

## Benefits for this specific setup

- **Multi-modal in one space.** Using the same CLIP text/vision models means text and image queries land in a shared 512-dim space — you can search images with a text prompt and vice versa, which is the interesting part of our demo.
- **Offline-first.** `local_files_only=True` plus a local shard means this can run with zero internet access after initial setup — genuinely useful for kiosks/robots/edge hardware as the product targets.
- **Persistence without a database server.** `EdgeShard.load()`/`.create()` gives you durable storage on disk with no separate process to manage, which is a real quality-of-life win over ad hoc in-memory FAISS-style setups.
- **Future sync path.** If this outgrows a single device, Edge's stated sync APIs mean you could later push this shard's data up to a Qdrant Server instance for centralized indexing/aggregation without re-architecting.

## Next 5 steps

1. **Benchmark realistically.** Load a few thousand real images (not 2 sample files) and time indexing throughput and query latency separately from CLIP inference time, so you know where time is actually going.
2. **Add filtering/payload indexes.** Right now you're only storing `type`/`text`/`path` in payload with no filtering. Try combining a `Query.Nearest` with a payload filter (e.g. restrict to `type == "image"`) to test Edge's filtered-search path, which is a common real-world need.
3. **Test the sync API.** Since this is meant to be an "edge" node, try pushing/pulling this shard to a Qdrant Server instance (even a local Docker one) to validate the promised backup/centralization workflow before you depend on it.
4. **Handle model download failures gracefully.** `local_files_only=True` will hard-fail if the cache is missing or the model naming changes between fastembed versions — worth wrapping in a clear error message or a fallback download path for anyone else running this.
5. **Decide on ID strategy up front.** We're mixing UUIDs, auto-generated IDs, and one hardcoded integer ID (`point_id=1` in `index.py`) — pin down whether IDs will be deterministic (e.g. hash of file path) or random before this grows, since that decision affects idempotent re-indexing (re-running `index.py` today will create duplicate points for the auto-ID text upsert).