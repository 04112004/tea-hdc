import numpy as np


# HDC dimensionality
DIMENSION = 10_000

# Reproducible random generator
RNG = np.random.default_rng(42)


def random_bipolar_vector():
    """
    Create a random bipolar hypervector.

    Each coordinate is either -1 or +1.
    """
    return RNG.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=DIMENSION,
    )


def bind(a, b):
    """
    Bind two hypervectors using element-wise multiplication.
    """
    return a * b


def bundle(vectors, weights=None):
    """
    Bundle multiple hypervectors using weighted addition.
    """

    if not vectors:
        raise ValueError("Cannot bundle an empty list of vectors.")

    if weights is None:
        weights = [1.0] * len(vectors)

    if len(vectors) != len(weights):
        raise ValueError("Number of vectors and weights must match.")

    result = np.zeros(DIMENSION, dtype=np.float32)

    for vector, weight in zip(vectors, weights):
        result += weight * vector

    return result


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    """

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)
def level_hypervector(index, total_levels, base_vector=None):
    """
    Create an ordered level hypervector.

    Nearby levels are intentionally more similar than
    distant levels.
    """

    if total_levels <= 0:
        raise ValueError("total_levels must be positive.")

    if not 0 <= index < total_levels:
        raise ValueError(
            f"index must be between 0 and {total_levels - 1}"
        )

    if base_vector is None:
        base_vector = random_bipolar_vector()

    if total_levels == 1:
        return base_vector.copy()

    vector = base_vector.copy()

    # Number of coordinates that change at each level.
    coordinates_per_step = DIMENSION // (total_levels - 1)

    start = 0

    for level in range(index):
        end = start + coordinates_per_step

        if level == total_levels - 2:
            end = DIMENSION

        vector[start:end] *= -1

        start = end

    return vector