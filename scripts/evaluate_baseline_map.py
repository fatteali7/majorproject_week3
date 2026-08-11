import time
import faiss
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

EMBEDDING_FILE = (
    "outputs/embeddings/baseline_val_ms_embeddings.npy"
)

CSV_FILE = (
    "outputs/metadata/val_split.csv"
)

TOP_K = 10


# ============================================================
# 1. Parse BigEarthNet labels
# ============================================================

def parse_labels(label_string):
    """
    Convert BigEarthNet label string into a Python set.

    Example input:
    "['Agro-forestry areas' 'Arable land' 'Urban fabric']"

    Output:
    {
        'Agro-forestry areas',
        'Arable land',
        'Urban fabric'
    }
    """

    if pd.isna(label_string):
        return set()

    text = str(label_string).strip()

    # Remove [ and ]
    text = text.strip("[]")

    # NumPy-style array strings separate labels by spaces,
    # but labels themselves can contain spaces.
    #
    # We use the single quotes to identify each label.
    import re

    labels = re.findall(r"'([^']*)'", text)

    return set(labels)


# ============================================================
# 2. Load embeddings
# ============================================================

print("Loading embeddings...")

embeddings = np.load(
    EMBEDDING_FILE
).astype("float32")

print("Embedding shape:", embeddings.shape)

assert embeddings.shape == (4500, 2048)


# ============================================================
# 3. Load metadata
# ============================================================

print("Loading validation metadata...")

df = pd.read_csv(CSV_FILE)

print("Validation rows:", len(df))

assert len(df) == 4500

assert "patch_id" in df.columns
assert "labels" in df.columns


# ============================================================
# 4. Parse labels
# ============================================================

print("Parsing labels...")

label_sets = [
    parse_labels(x)
    for x in df["labels"]
]

print("Labels parsed successfully.")

print()
print("Example labels:")
print(label_sets[0])


# ============================================================
# 5. Build FAISS index
# ============================================================

print()
print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

start = time.perf_counter()

index.add(embeddings)

build_time = (
    time.perf_counter() - start
) * 1000

print("FAISS vectors:", index.ntotal)
print("Embedding dimension:", dimension)
print(f"Index build time: {build_time:.3f} ms")


# ============================================================
# 6. Retrieval + Average Precision
# ============================================================

def average_precision(relevant):

    num_relevant = sum(relevant)

    if num_relevant == 0:
        return 0.0

    score = 0.0
    hits = 0

    for rank, is_relevant in enumerate(
        relevant,
        start=1
    ):

        if is_relevant:
            hits += 1
            precision_at_rank = hits / rank
            score += precision_at_rank

    return score / num_relevant


# ============================================================
# 7. Evaluate all 4500 queries
# ============================================================

print()
print("Running baseline retrieval evaluation...")

start_eval = time.perf_counter()

average_precisions = []

for query_idx in range(len(embeddings)):

    query = embeddings[
        query_idx:query_idx + 1
    ]

    distances, indices = index.search(
        query,
        k=TOP_K + 1
    )

    query_labels = label_sets[query_idx]

    retrieved_indices = indices[0]

    relevant = []

    for retrieved_idx in retrieved_indices:

        # Skip the query itself
        if retrieved_idx == query_idx:
            continue

        retrieved_labels = label_sets[
            retrieved_idx
        ]

        # Relevant if at least one land-cover
        # label is shared.
        is_relevant = bool(
            query_labels.intersection(
                retrieved_labels
            )
        )

        relevant.append(
            is_relevant
        )

        if len(relevant) == TOP_K:
            break

    ap = average_precision(
        relevant
    )

    average_precisions.append(ap)


# ============================================================
# 8. Calculate mAP
# ============================================================

evaluation_time = (
    time.perf_counter()
    - start_eval
)

mAP = float(
    np.mean(average_precisions)
)


# ============================================================
# 9. Display results
# ============================================================

print()
print("=" * 60)
print("BASELINE RETRIEVAL RESULTS")
print("=" * 60)

print("Number of queries:", len(embeddings))
print("Gallery size:", index.ntotal)
print("Embedding dimension:", dimension)
print("Top-K:", TOP_K)

print()
print(
    f"Baseline mAP@{TOP_K}: {mAP:.6f}"
)

print(
    f"Evaluation time: {evaluation_time:.2f} seconds"
)

print("=" * 60)


# ============================================================
# 10. Save result
# ============================================================

result_file = (
    "outputs/embeddings/baseline_map_result.txt"
)

with open(result_file, "w") as f:

    f.write(
        "Baseline S2 Retrieval Evaluation\n"
    )

    f.write(
        f"Queries: {len(embeddings)}\n"
    )

    f.write(
        f"Gallery: {index.ntotal}\n"
    )

    f.write(
        f"Embedding dimension: {dimension}\n"
    )

    f.write(
        f"Top-K: {TOP_K}\n"
    )

    f.write(
        f"mAP@{TOP_K}: {mAP:.6f}\n"
    )

    f.write(
        f"Evaluation time: "
        f"{evaluation_time:.2f} seconds\n"
    )

print()
print(
    f"Result saved to: {result_file}"
)