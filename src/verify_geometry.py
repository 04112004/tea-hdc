from data_loader import load_tea_dataset
from encoder import encode_tea
from hypervectors import cosine_similarity


print("=" * 70)
print("HDC GEOMETRY VERIFICATION")
print("=" * 70)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

dataset = load_tea_dataset()
teas = dataset["train"]

print("\nNumber of teas:", len(teas))


# ---------------------------------------------------------
# Encode all teas
# ---------------------------------------------------------

print("\nEncoding teas...")

encoded = {}

for index, tea in enumerate(teas):

    tea_id = str(tea["id"])

    encoded[tea_id] = encode_tea(tea)

    print(
        f"\rEncoded {index + 1}/{len(teas)}",
        end="",
    )

print("\n")


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def find_tea(title):
    for tea in teas:
        if tea["title"].lower() == title.lower():
            return tea

    raise ValueError(
        f"Tea not found: {title}"
    )


def similarity_between(title1, title2):

    tea1 = find_tea(title1)
    tea2 = find_tea(title2)

    vector1 = encoded[str(tea1["id"])]
    vector2 = encoded[str(tea2["id"])]

    return cosine_similarity(
        vector1,
        vector2,
    )


# =========================================================
# TEST 1 — Same class
# =========================================================

print("=" * 70)
print("TEST 1: SAME CLASS")
print("=" * 70)

tea1 = find_tea("Ali Shan")
tea2 = find_tea("Jin Xuan")

similarity = similarity_between(
    "Ali Shan",
    "Jin Xuan",
)

print(
    f"\nAli Shan class: {tea1['class']}"
)

print(
    f"Jin Xuan class: {tea2['class']}"
)

print(
    f"Similarity: {similarity:.4f}"
)


# =========================================================
# TEST 2 — Different class
# =========================================================

print("\n" + "=" * 70)
print("TEST 2: DIFFERENT CLASS")
print("=" * 70)

tea3 = find_tea("Ali Shan")
tea4 = find_tea("Yunnan Dian Hong")

similarity = similarity_between(
    "Ali Shan",
    "Yunnan Dian Hong",
)

print(
    f"\nAli Shan class: {tea3['class']}"
)

print(
    f"Yunnan Dian Hong class: {tea4['class']}"
)

print(
    f"Similarity: {similarity:.4f}"
)


# =========================================================
# TEST 3 — Same tea
# =========================================================

print("\n" + "=" * 70)
print("TEST 3: SAME TEA")
print("=" * 70)

similarity = similarity_between(
    "Ali Shan",
    "Ali Shan",
)

print(
    f"\nAli Shan ↔ Ali Shan: "
    f"{similarity:.4f}"
)


# =========================================================
# TEST 4 — Elevation
# =========================================================

print("\n" + "=" * 70)
print("TEST 4: ELEVATION")
print("=" * 70)

print(
    "\nComparing teas with different elevations."
)

pairs = [
    (
        "Ali Shan",
        "Jin Xuan",
    ),
    (
        "Ali Shan",
        "Da Yu Ling",
    ),
]

for title1, title2 in pairs:

    tea1 = find_tea(title1)
    tea2 = find_tea(title2)

    similarity = similarity_between(
        title1,
        title2,
    )

    print(
        f"\n{title1}: "
        f"{tea1['elevation_meters']} m"
    )

    print(
        f"{title2}: "
        f"{tea2['elevation_meters']} m"
    )

    print(
        f"Similarity: {similarity:.4f}"
    )


# =========================================================
# TEST 5 — Similar sensory teas
# =========================================================

print("\n" + "=" * 70)
print("TEST 5: SENSORY SIMILARITY")
print("=" * 70)

similarity = similarity_between(
    "Yunnan Dian Hong",
    "Assam Doomni",
)

print(
    "\nYunnan Dian Hong ↔ Assam Doomni"
)

print(
    f"Similarity: {similarity:.4f}"
)


# =========================================================
# Finish
# =========================================================

print("\n" + "=" * 70)
print("GEOMETRY VERIFICATION COMPLETE")
print("=" * 70)