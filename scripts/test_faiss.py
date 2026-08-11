import faiss
import numpy as np
import time


d = 512
n = 1000

np.random.seed(42)

gallery = np.random.randn(
    n, d
).astype("float32")


index = faiss.IndexFlatL2(d)

index.add(gallery)

print(
    f"Index built: {index.ntotal} vectors"
)


query = np.random.randn(
    1, d
).astype("float32")


start = time.perf_counter()

distances, indices = index.search(
    query,
    k=10
)

elapsed_ms = (
    time.perf_counter() - start
) * 1000


print(
    f"Top-10 indices: {indices[0]}"
)

print(
    f"Query time: {elapsed_ms:.3f} ms"
)


assert elapsed_ms < 100

print("FAISS test passed OK")