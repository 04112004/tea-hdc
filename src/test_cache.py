from embeddings import get_embedding


print("=" * 60)
print("EMBEDDING CACHE TEST")
print("=" * 60)


text = "floral lavender"


print("\nFirst request...")
embedding1 = get_embedding(text)

print("Dimensions:", embedding1.shape)


print("\nSecond request...")
embedding2 = get_embedding(text)

print("Dimensions:", embedding2.shape)


print("\nChecking equality...")

if (embedding1 == embedding2).all():
    print("✓ Cached embedding matches original embedding")
else:
    print("✗ Embeddings do not match")


print("\nCache test complete.")