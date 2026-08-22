import json
from pathlib import Path

import numpy as np
import requests


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"

EMBEDDING_DIMENSION = 768
HYPERVECTOR_DIMENSION = 10000

CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "embeddings.json"

# Projection matrix configuration
PROJECTION_FILE = CACHE_DIR / "projection_matrix.npy"
RANDOM_SEED = 42


# ---------------------------------------------------------
# Cache handling
# ---------------------------------------------------------

def load_cache():
    """Load cached embeddings from disk."""

    if not CACHE_FILE.exists():
        return {}

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache):
    """Save embeddings to disk."""

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


# ---------------------------------------------------------
# Projection matrix
# ---------------------------------------------------------

def get_projection_matrix():
    """
    Load or create the fixed projection matrix.

    Shape:
        768 × 10,000

    The same matrix must be used for every tea.
    """

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load existing matrix
    if PROJECTION_FILE.exists():

        matrix = np.load(
            PROJECTION_FILE
        )

        expected_shape = (
            EMBEDDING_DIMENSION,
            HYPERVECTOR_DIMENSION
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                f"Projection matrix has shape "
                f"{matrix.shape}, expected "
                f"{expected_shape}"
            )

        return matrix.astype(
            np.float32
        )

    # Create deterministic matrix
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    matrix = rng.choice(
        [-1.0, 1.0],
        size=(
            EMBEDDING_DIMENSION,
            HYPERVECTOR_DIMENSION
        )
    ).astype(
        np.float32
    )

    # Save it so every future run uses
    # exactly the same projection.
    np.save(
        PROJECTION_FILE,
        matrix
    )

    return matrix


# ---------------------------------------------------------
# Single text → 768-D embedding
# ---------------------------------------------------------

def get_embedding(text):
    """
    Get a 768-D embedding from cache or Ollama.
    """

    text = str(text).strip()

    if not text:
        return np.zeros(
            EMBEDDING_DIMENSION,
            dtype=np.float32
        )

    cache = load_cache()

    # Check cache first
    if text in cache:

        return np.array(
            cache[text],
            dtype=np.float32
        )

    # Not cached → call Ollama
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    embedding = response.json()["embedding"]

    # Save embedding
    cache[text] = embedding

    save_cache(cache)

    return np.array(
        embedding,
        dtype=np.float32
    )


# ---------------------------------------------------------
# Multiple phrases → averaged 768-D embedding
# ---------------------------------------------------------

def phrases_to_embedding(phrases):
    """
    Convert multiple phrases into one averaged
    768-D embedding.
    """

    if not phrases:
        return np.zeros(
            EMBEDDING_DIMENSION,
            dtype=np.float32
        )

    embeddings = []

    # Normalize, deduplicate and sort
    cleaned = set()

    for phrase in phrases:

        phrase = str(
            phrase
        ).strip().lower()

        if phrase:
            cleaned.add(
                phrase
            )

    cleaned = sorted(
        cleaned
    )

    # Generate/get cached embeddings
    for phrase in cleaned:

        embedding = get_embedding(
            phrase
        )

        embeddings.append(
            embedding
        )

    if not embeddings:
        return np.zeros(
            EMBEDDING_DIMENSION,
            dtype=np.float32
        )

    # Average
    mean_embedding = np.mean(
        embeddings,
        axis=0
    )

    return mean_embedding.astype(
        np.float32
    )


# ---------------------------------------------------------
# 768-D embedding → 10,000-D bipolar hypervector
# ---------------------------------------------------------

def embedding_to_hypervector(
    embedding
):
    """
    Convert a 768-D embedding into a
    10,000-D bipolar hypervector.
    """

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    if embedding.shape != (
        EMBEDDING_DIMENSION,
    ):
        raise ValueError(
            f"Expected embedding shape "
            f"({EMBEDDING_DIMENSION},), "
            f"got {embedding.shape}"
        )

    # Get the fixed projection matrix
    projection_matrix = (
        get_projection_matrix()
    )

    # 768-D → 10,000-D
    projected = (
        embedding @ projection_matrix
    )

    # Convert to bipolar
    hypervector = np.sign(
        projected
    )

    # Replace zero with +1
    hypervector[
        hypervector == 0
    ] = 1

    return hypervector.astype(
        np.float32
    )


# ---------------------------------------------------------
# Text → 10,000-D bipolar hypervector
# ---------------------------------------------------------

def text_to_hypervector(
    text
):
    """
    Convert a single text string directly
    into a 10,000-D bipolar hypervector.
    """

    embedding = get_embedding(
        text
    )

    return embedding_to_hypervector(
        embedding
    )


# ---------------------------------------------------------
# Multiple phrases → 10,000-D bipolar hypervector
# ---------------------------------------------------------

def phrases_to_hypervector(
    phrases
):
    """
    Convert multiple phrases into a
    10,000-D bipolar hypervector.
    """

    embedding = phrases_to_embedding(
        phrases
    )

    return embedding_to_hypervector(
        embedding
    )