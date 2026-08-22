import numpy as np

from embeddings import (
    EMBEDDING_DIMENSION,
    HYPERVECTOR_DIMENSION,
    get_embedding,
    embedding_to_hypervector,
    phrases_to_embedding,
    text_to_hypervector,
    phrases_to_hypervector,
)


print("=" * 60)
print("EMBEDDING → HYPERVECTOR TEST")
print("=" * 60)


# ---------------------------------------------------------
# 1. Ollama embedding
# ---------------------------------------------------------

text = "lavender"

embedding = get_embedding(text)

print("\nOllama embedding")
print("Shape:", embedding.shape)
print("First 10:", embedding[:10])

assert embedding.shape == (
    EMBEDDING_DIMENSION,
)

print("✓ Ollama embedding works")


# ---------------------------------------------------------
# 2. Projection
# ---------------------------------------------------------

hv = embedding_to_hypervector(
    embedding
)

print("\nProjected hypervector")
print("Shape:", hv.shape)
print("First 10:", hv[:10])

assert hv.shape == (
    HYPERVECTOR_DIMENSION,
)

assert np.all(
    np.isin(hv, [-1.0, 1.0])
)

print("✓ 768D → 10,000D bipolar conversion works")


# ---------------------------------------------------------
# 3. Direct text conversion
# ---------------------------------------------------------

hv2 = text_to_hypervector(
    "lavender"
)

print("\nDirect text → hypervector")
print("Shape:", hv2.shape)

assert hv2.shape == (
    HYPERVECTOR_DIMENSION,
)

print("✓ Direct conversion works")


# ---------------------------------------------------------
# 4. Multiple phrases
# ---------------------------------------------------------

phrases = [
    "LEMONGRASS",
    "WHITE PEACH",
    "LAVENDER",
]

hv3 = phrases_to_hypervector(
    phrases
)

print("\nMultiple phrases → hypervector")
print("Shape:", hv3.shape)
print("First 10:", hv3[:10])

assert hv3.shape == (
    HYPERVECTOR_DIMENSION,
)

assert np.all(
    np.isin(hv3, [-1.0, 1.0])
)

print("✓ Multiple phrase conversion works")


print("\n" + "=" * 60)
print("ALL EMBEDDING TESTS PASSED")
print("=" * 60)

