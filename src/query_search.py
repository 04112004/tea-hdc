import argparse
import re

import lancedb
import numpy as np

from embeddings import text_to_hypervector
from query_parser import parse_query


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DB_PATH = "data/tea_hypervectors.lance"

# HDC remains the primary signal.
HDC_WEIGHT = 0.80
METADATA_WEIGHT = 0.20


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
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


SENSORY_GROUPS = {
    "floral": ["floral", "flower", "lavender", "jasmine", "orchid", "rose"],
    "fruity": ["fruity", "fruit", "peach", "apple", "berry", "citrus", "citrusy"],
    "sweet": ["sweet", "honey", "caramel", "sugar", "candied"],
    "earthy": ["earthy", "soil", "moss", "woodsy", "woody"],
    "smoky": ["smoky", "smoke", "charcoal", "roasted"],
    "herbal": ["herbal", "tea leaf", "grass", "mint", "lemongrass"],
    "nutty": ["nutty", "nut", "nuts", "almond", "peanut", "toasted peanut", "hazelnut"],
    "woody": ["woody", "wood", "oak", "exotic wood"],
    "buttery": ["butter", "buttery", "melted butter", "freshly churned butter"],
    "citrusy": ["citrus", "citrus peel", "lemon", "lemon zest", "orange", "mandarine", "mandarin"],
}


def calculate_sensory_score(
    query_sensory,
    aroma,
    taste,
):
    """
    Calculate how well a tea's aroma/taste matches
    the sensory terms detected in the query.
    Returns a value between 0.0 and 1.0.
    """
    if not query_sensory:
        return 0.0

    sensory_text = ""
    if aroma is not None:
        sensory_text += " " + str(aroma)
    if taste is not None:
        sensory_text += " " + str(taste)
    sensory_text = sensory_text.lower()

    matched = 0
    for query_term in query_sensory:
        keywords = SENSORY_GROUPS.get(
            query_term,
            [query_term]
        )
        found = False
        for keyword in keywords:
            if keyword.lower() in sensory_text:
                found = True
                break
        if found:
            matched += 1

    return matched / len(query_sensory)


# ---------------------------------------------------------
# Query metadata extraction
# ---------------------------------------------------------

def extract_query_metadata(query):
    """
    Detect explicit metadata information from the
    natural-language query.

    Examples:

        "black tea"
            -> class = black

        "Taiwanese oolong"
            -> country = Taiwan
               class = oolong

        "roasted oolong"
            -> roast = roasted

        "high elevation tea"
            -> elevation = high
    """

    query_lower = query.lower()

    metadata = {
        "class": None,
        "country": None,
        "roast": None,
        "elevation": None,
    }

    # -----------------------------------------------------
    # Tea classes
    # -----------------------------------------------------

    tea_classes = [
        "green",
        "black",
        "oolong",
        "white",
        "yellow",
    ]

    for tea_class in tea_classes:

        if re.search(
            rf"\b{tea_class}\b",
            query_lower
        ):
            metadata["class"] = tea_class
            break

    # -----------------------------------------------------
    # Countries
    # -----------------------------------------------------

    countries = {
        "taiwan": "Taiwan",
        "taiwanese": "Taiwan",

        "china": "China",
        "chinese": "China",

        "japan": "Japan",
        "japanese": "Japan",

        "india": "India",
        "indian": "India",

        "nepal": "Nepal",
        "nepalese": "Nepal",

        "sri lanka": "Sri Lanka",
        "ceylon": "Sri Lanka",
    }

    for keyword, country in countries.items():

        if keyword in query_lower:
            metadata["country"] = country
            break

    # -----------------------------------------------------
    # Roast
    # -----------------------------------------------------

    roast_words = [
        "roasted",
        "roast",
        "roasting",
    ]

    for word in roast_words:

        if re.search(
            rf"\b{word}\b",
            query_lower
        ):
            metadata["roast"] = "roasted"
            break

    # -----------------------------------------------------
    # Elevation
    # -----------------------------------------------------

    if (
        "high elevation" in query_lower
        or "high altitude" in query_lower
        or "mountain tea" in query_lower
    ):
        metadata["elevation"] = "high"

    elif (
        "low elevation" in query_lower
        or "low altitude" in query_lower
    ):
        metadata["elevation"] = "low"

    return metadata


# ---------------------------------------------------------
# Metadata matching score
# ---------------------------------------------------------

def metadata_score(
    query_metadata,
    row
):
    """
    Calculate a normalized metadata score.

    Each explicitly requested metadata property
    contributes equally.
    """

    requested = 0
    matched = 0

    # -----------------------------------------------------
    # Class
    # -----------------------------------------------------

    requested_class = query_metadata["class"]

    if requested_class is not None:

        requested += 1

        row_class = str(
            row["class"]
        ).lower()

        if row_class == requested_class:
            matched += 1

    # -----------------------------------------------------
    # Country
    # -----------------------------------------------------

    requested_country = query_metadata["country"]

    if requested_country is not None:

        requested += 1

        row_country = str(
            row["country"]
        ).lower()

        if row_country == requested_country.lower():
            matched += 1

    # -----------------------------------------------------
    # Roast
    # -----------------------------------------------------

    requested_roast = query_metadata["roast"]

    if requested_roast is not None:

        requested += 1

        row_roast = str(
            row["roast"]
        ).lower()

        if (
            row_roast != "none"
            and row_roast != ""
            and row_roast != "nan"
        ):
            matched += 1

    # -----------------------------------------------------
    # Elevation
    # -----------------------------------------------------

    requested_elevation = query_metadata["elevation"]

    if requested_elevation is not None:

        requested += 1

        try:
            elevation = float(
                row["elevation_meters"]
            )
        except (
            TypeError,
            ValueError
        ):
            elevation = 0.0

        if requested_elevation == "high":

            # Dataset contains elevations from
            # low hundreds to above 2000m.
            if elevation >= 1000:
                matched += 1

        elif requested_elevation == "low":

            if elevation < 1000:
                matched += 1

    # -----------------------------------------------------
    # No explicit metadata
    # -----------------------------------------------------

    if requested == 0:
        return 0.0

    return matched / requested


# ---------------------------------------------------------
# Natural language query search
# ---------------------------------------------------------

def search_by_query(
    query,
    limit=5
):
    """
    Search teas using HDC + metadata-aware ranking.
    """

    print()
    print("Generating query hypervector...")

    query_vector = text_to_hypervector(
        query
    )
    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )

    print(
        f"Query hypervector shape: "
        f"{query_vector.shape}"
    )

    metadata = parse_query(query)

    print()
    print("Detected query metadata:")

    for key, value in metadata.items():
        if value:
            print(f"  {key}: {value}")

    db = lancedb.connect(DB_PATH)
    table = db.open_table("teas")
    df = table.to_pandas()

    if df.empty:
        raise ValueError(
            "Tea database is empty."
        )

    results = []

    for _, row in df.iterrows():
        tea_vector = np.asarray(
            row["hypervector"],
            dtype=np.float32
        )

        hdc_similarity = cosine_similarity(
            query_vector,
            tea_vector
        )

        # Metadata score from class/country/roast/oxidation.
        requested = 0
        matched = 0

        for field in ["class", "country", "roast", "oxidation"]:
            query_value = metadata.get(field)

            if query_value is None:
                continue

            requested += 1
            row_value = str(row.get(field, "")).lower()
            query_value = str(query_value).lower()

            if row_value == query_value:
                matched += 1

        metadata_score = (
            matched / requested
            if requested > 0
            else 0.0
        )

        # -------------------------------------------------
        # Sensory score
        # -------------------------------------------------
        aroma = row.get("aroma", [])
        taste = row.get("taste", [])
        sensory_score = calculate_sensory_score(
            metadata.get("sensory", []),
            aroma,
            taste,
        )

        final_score = (
            0.30 * hdc_similarity
            + 0.50 * metadata_score
            + 0.20 * sensory_score
        )

        results.append({
            "similarity": final_score,
            "hdc_similarity": hdc_similarity,
            "metadata_score": metadata_score,
            "sensory_score": sensory_score,
            "score": final_score,
            "id": row["id"],
            "title": row["title"],
            "class": row["class"],
            "country": row["country"],
            "region": row["region"],
            "oxidation": row["oxidation"],
            "roast": row["roast"],
            "aroma": row.get("aroma"),
            "taste": row.get("taste"),
            "elevation_meters": row["elevation_meters"],
        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:limit]


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

def display_results(
    query,
    results
):
    """
    Display search results.
    """

    print()
    print("=" * 90)
    print("NATURAL LANGUAGE TEA SEARCH")
    print("=" * 90)

    print()
    print(
        f"Query: {query}"
    )

    print()
    print(
        f"{'score':<10}"
        f"{'HDC':<10}"
        f"{'metadata':<10}"
        f"{'sensory':<10}"
        f"{'class':<10}"
        f"{'title':<40}"
        f"{'country':<15}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['similarity']:<10.4f}"
            f"{result['hdc_similarity']:<10.4f}"
            f"{result['metadata_score']:<10.2f}"
            f"{result['sensory_score']:<10.2f}"
            f"{str(result['class']):<10}"
            f"{str(result['title'])[:38]:<40}"
            f"{str(result['country'])[:13]:<15}"
        )

    print("-" * 90)

    print()
    print("Detailed results:")

    for index, result in enumerate(
        results,
        start=1
    ):
        print()
        print(
            f"{index}. {result['title']}"
        )

        print(
            f"   Final score: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"   HDC similarity: "
            f"{result['hdc_similarity']:.4f}"
        )

        print(
            f"   Metadata score: "
            f"{result['metadata_score']:.2f}"
        )

        print(
            f"   Sensory score: "
            f"{result['sensory_score']:.2f}"
        )

        print(
            f"   Class: "
            f"{result['class']}"
        )

        print(
            f"   Country: "
            f"{result['country']}"
        )

        print(
            f"   Region: "
            f"{result['region']}"
        )

        print(
            f"   Oxidation: "
            f"{result['oxidation']}"
        )

        print(
            f"   Roast: "
            f"{result['roast']}"
        )

        print(
            f"   Elevation: "
            f"{result['elevation_meters']} m"
        )

    print()
    print("=" * 90)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Search teas using "
            "HDC + metadata-aware ranking."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language tea query"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results"
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError(
            "Limit must be greater than 0."
        )

    results = search_by_query(
        args.query,
        args.limit
    )

    display_results(
        args.query,
        results
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()