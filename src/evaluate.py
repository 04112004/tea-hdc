import lancedb
import numpy as np


DB_PATH = "data/tea_hypervectors.lance"


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ---------------------------------------------------------
# Load database
# ---------------------------------------------------------

def load_database():
    print("Connecting to LanceDB...")

    db = lancedb.connect(DB_PATH)
    table = db.open_table("teas")

    df = table.to_pandas()

    print(f"Loaded {len(df)} teas")

    return df


# ---------------------------------------------------------
# Calculate all pairwise similarities
# ---------------------------------------------------------

def calculate_similarities(df):

    results = []

    for i in range(len(df)):

        query = df.iloc[i]

        query_vector = np.asarray(
            query["hypervector"],
            dtype=np.float32
        )

        for j in range(i + 1, len(df)):

            candidate = df.iloc[j]

            candidate_vector = np.asarray(
                candidate["hypervector"],
                dtype=np.float32
            )

            similarity = cosine_similarity(
                query_vector,
                candidate_vector
            )

            same_class = (
                query["class"] == candidate["class"]
            )

            results.append({
                "tea1": query["title"],
                "tea2": candidate["title"],
                "class1": query["class"],
                "class2": candidate["class"],
                "similarity": similarity,
                "same_class": same_class,
            })

    return results


# ---------------------------------------------------------
# Evaluate class separation
# ---------------------------------------------------------

def evaluate_class_separation(results):

    same_class = [
        r["similarity"]
        for r in results
        if r["same_class"]
    ]

    different_class = [
        r["similarity"]
        for r in results
        if not r["same_class"]
    ]

    print()
    print("=" * 70)
    print("CLASS SEPARATION")
    print("=" * 70)

    print(
        f"Same-class pairs: "
        f"{len(same_class)}"
    )

    print(
        f"Different-class pairs: "
        f"{len(different_class)}"
    )

    if same_class:
        print(
            f"Average same-class similarity: "
            f"{np.mean(same_class):.4f}"
        )

    if different_class:
        print(
            f"Average different-class similarity: "
            f"{np.mean(different_class):.4f}"
        )

    if same_class and different_class:

        separation = (
            np.mean(same_class)
            - np.mean(different_class)
        )

        print(
            f"Class separation: "
            f"{separation:.4f}"
        )


# ---------------------------------------------------------
# Show highest similarity pairs
# ---------------------------------------------------------

def show_top_pairs(results, limit=10):

    sorted_results = sorted(
        results,
        key=lambda x: x["similarity"],
        reverse=True
    )

    print()
    print("=" * 70)
    print("TOP SIMILAR TEA PAIRS")
    print("=" * 70)

    for i, result in enumerate(
        sorted_results[:limit],
        start=1
    ):

        print(
            f"{i}. "
            f"{result['tea1']} ↔ "
            f"{result['tea2']}"
        )

        print(
            f"   Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"   Classes: "
            f"{result['class1']} / "
            f"{result['class2']}"
        )


# ---------------------------------------------------------
# Show lowest similarity pairs
# ---------------------------------------------------------

def show_lowest_pairs(results, limit=10):

    sorted_results = sorted(
        results,
        key=lambda x: x["similarity"]
    )

    print()
    print("=" * 70)
    print("LOWEST SIMILAR TEA PAIRS")
    print("=" * 70)

    for i, result in enumerate(
        sorted_results[:limit],
        start=1
    ):

        print(
            f"{i}. "
            f"{result['tea1']} ↔ "
            f"{result['tea2']}"
        )

        print(
            f"   Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"   Classes: "
            f"{result['class1']} / "
            f"{result['class2']}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("TEA HDC EVALUATION")
    print("=" * 70)

    df = load_database()

    print()
    print("Calculating pairwise similarities...")

    results = calculate_similarities(df)

    print(
        f"Calculated {len(results)} tea pairs"
    )

    evaluate_class_separation(results)

    show_top_pairs(
        results,
        limit=10
    )

    show_lowest_pairs(
        results,
        limit=10
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()