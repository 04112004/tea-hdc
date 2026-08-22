import argparse

import lancedb
import numpy as np


DB_PATH = "data/tea_hypervectors.lance"


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    """

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
# Search similar teas
# ---------------------------------------------------------

def search_similar_teas(
    tea_id,
    limit=3
):
    """
    Find the most similar teas to the given tea.
    """

    # Connect to LanceDB
    db = lancedb.connect(DB_PATH)

    table = db.open_table("teas")

    # Load all records
    df = table.to_pandas()

    # Find query tea
    matches = df[df["id"].astype(str) == str(tea_id)]

    if matches.empty:
        raise ValueError(
            f"Tea with id {tea_id} not found."
        )

    query = matches.iloc[0]

    query_vector = np.asarray(
        query["hypervector"],
        dtype=np.float32
    )

    print()
    print(
        f"Seed: {query['title']} "
        f"(id={query['id']})"
    )

    # Calculate similarity
    results = []

    for _, row in df.iterrows():

        # Don't compare tea with itself
        if str(row["id"]) == str(tea_id):
            continue

        vector = np.asarray(
            row["hypervector"],
            dtype=np.float32
        )

        similarity = cosine_similarity(
            query_vector,
            vector
        )

        results.append({
            "similarity": similarity,
            "id": row["id"],
            "title": row["title"],
            "class": row["class"],
            "country": row["country"],
            "region": row["region"],
        })

    # Highest similarity first
    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:limit]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Find similar teas using HDC"
    )

    parser.add_argument(
        "--tea-id",
        required=True,
        help="ID of the tea to search for"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of similar teas"
    )

    args = parser.parse_args()

    results = search_similar_teas(
        args.tea_id,
        args.limit
    )

    print()
    print(
        f"{'similarity':<12}"
        f"{'class':<10}"
        f"{'title':<45}"
    )

    print("-" * 67)

    for result in results:

        print(
            f"{result['similarity']:<12.4f}"
            f"{result['class']:<10}"
            f"{result['title']:<45}"
        )


if __name__ == "__main__":
    main()