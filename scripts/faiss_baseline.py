import time
import faiss
import numpy as np

EMBEDDING_FILE = "outputs/embeddings/baseline_val_ms_embeddings.npy"

# -------------------------------------------------
# 1. Load embeddings
# -------------------------------------------------

embeddings = np.load(EMBEDDING_FILE).astype("float32")

print("Loaded embeddings")
print("Shape:", embeddings.shape)

assert embeddings.shape == (4500, 2048)

# -------------------------------------------------
# 2. Build FAISS index
# -------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

start_build = time.perf_counter()

index.add(embeddings)

build_time = (
    time.perf_counter() - start_build
) * 1000

print()
print("FAISS index created")
print("Vectors:", index.ntotal)
print("Dimension:", dimension)
print(f"Index build time: {build_time:.3f} ms")

assert index.ntotal == 4500

# -------------------------------------------------
# 3. Test query
# -------------------------------------------------

query = embeddings[0:1]

start = time.perf_counter()

distances, indices = index.search(
    query,
    k=10
)

query_time = (
    time.perf_counter() - start
) * 1000

print()
print("Query results")
print("Top-10 indices:", indices[0])
print("Distances:", distances[0])
print(f"Query time: {query_time:.3f} ms")

print()
print("FAISS baseline retrieval test passed!")