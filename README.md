# 🍵 Tea HDC — Hyperdimensional Computing for Natural Language Tea Search

A hands-on implementation of **Hyperdimensional Computing (HDC)** for semantic tea search.

This project demonstrates how natural-language tea queries can be converted into high-dimensional hypervectors and combined with structured tea metadata and sensory information to retrieve relevant teas.

**Stack:**
- Python
- Hyperdimensional Computing (HDC)
- 10,000-dimensional hypervectors
- Text embeddings
- Binding and bundling
- LanceDB
- NumPy
- `uv` for Python environment and dependency management

---

## 📌 Project Overview

The goal of this project is to build a tea search system where a user can enter queries such as:

```
green tea
Taiwanese oolong
floral tea
fruity tea
floral fruity Taiwanese oolong
```

and retrieve relevant teas from a dataset.

The search system combines three main signals:

1. **HDC similarity**
2. **Metadata matching**
3. **Sensory matching**

The final ranking is calculated using these signals.

---

## 🧠 What is Hyperdimensional Computing?

Hyperdimensional Computing represents information using very large vectors called **hypervectors**.

In this project, every hypervector has **10,000 dimensions**, for example:

```
[1, -1, 1, 1, -1, ...]   (10,000 values)
```

Instead of representing information using a small number of features, HDC represents information in a very high-dimensional space. This provides useful properties such as:

- Distributed representation
- Robustness
- Similarity-based computation
- Symbolic operations using vector algebra (binding, bundling)

---

## 🧩 Core HDC Operations

### 1. Hypervector

A hypervector is the basic representation used by the HDC system:

```
Tea → Hypervector (10,000 dimensions)
```

A tea can have many attributes, each represented as an individual hypervector:

| Attribute | Example value |
|---|---|
| class | oolong |
| country | Taiwan |
| oxidation | medium |
| roast | none |

### 2. 🔗 Binding

Binding combines two pieces of information into a new representation.

```
COUNTRY ⊗ TAIWAN  →  "country = Taiwan"
```

The binding operation allows relationships between values to be represented:

```
CLASS   ⊗ OOLONG
COUNTRY ⊗ TAIWAN
ROAST   ⊗ LIGHT
```

Each produces a different hypervector.

### 3. 📦 Bundling

Bundling combines multiple hypervectors into one representation:

```
CLASS ⊗ OOLONG
  +
COUNTRY ⊗ TAIWAN
  +
ROAST ⊗ LIGHT
  =
Tea hypervector
```

Conceptually:

```
Tea
 ├── class
 ├── country
 ├── oxidation
 ├── roast
 ├── elevation
 └── sensory information
```

The result is one high-dimensional tea hypervector.

---

## 🧠 Tea Hypervector vs. Individual Hypervector

**Individual hypervectors** represent single concepts, e.g.:
`OOLONG`, `TAIWAN`, `GREEN`, `FLORAL`, `FRUITY`, `LIGHT_ROAST`

**A tea hypervector** represents the combination of information belonging to one tea. For example, *Li Shan* may encode:

- class = oolong
- country = Taiwan
- region = Li Shan
- oxidation = medium
- roast = none
- elevation = 2000m
- sensory characteristics

These individual pieces are combined to create the stored representation for that tea.

---

## 🗄️ LanceDB

The project uses **LanceDB** as the local vector database. The database stores tea records together with their generated hypervectors and metadata.

```
Tea Dataset → Generate hypervectors → Tea hypervector → LanceDB
```

A record can contain fields such as:

- `id`, `title`, `class`, `country`, `region`
- `oxidation`, `roast`, `aroma`, `taste`, `elevation`
- `hypervector`

> **Note:** The project stores **both** the tea metadata *and* the tea hypervector — not metadata alone. The `hypervector` column holds the full binding/bundling result for each tea.

---

## 🔄 Overall Data Flow

**Indexing pipeline:**

```
Tea Dataset
  → Data Loading
  → Text / Attribute Encoding
  → Individual Hypervectors
  → Binding
  → Bundling
  → Tea Hypervector
  → Store (metadata + hypervector) in LanceDB
```

**Query pipeline:**

```
User Query
  → Query Parser ──┬── Metadata
                    └── Sensory terms
  → Text → Hypervector
  → Compare with stored tea hypervectors
  → HDC similarity + Metadata score + Sensory score
  → Final ranking
  → Top K teas
```

---

## 📁 Project Structure

```
tea-hdc/
│
├── data/
│   └── tea_hypervectors.lance
│
├── src/
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── hypervectors.py
│   ├── encoder.py
│   ├── build_hypervectors.py
│   ├── verify_geometry.py
│   ├── verify_invertibility.py
│   ├── search.py
│   ├── associative_memory.py
│   ├── query_parser.py
│   ├── query_search.py
│   └── inspect_sensory.py
│
├── tests/
│
├── pyproject.toml
│
└── README.md
```

---

## 📂 Important Files

| File | Purpose |
|---|---|
| `data_loader.py` | Loads the tea dataset (id, source_url, image, class, title, description, country, region, oxidation, roast, aroma, taste, elevation_meters, elevation_confidence). |
| `embeddings.py` | Converts text into vector representations — e.g. turns `"floral fruity Taiwanese oolong"` into a 10,000-dimensional query hypervector. |
| `hypervectors.py` | Core HDC operations — the mathematical representation of hypervectors, binding, and bundling. |
| `encoder.py` | Encodes tea attributes into hypervectors: attributes → individual hypervectors → binding/bundling → tea hypervector. |
| `build_hypervectors.py` | Builds the hypervector representation for the full dataset (read tea → encode attributes → create tea hypervector → store result). |
| `query_parser.py` | Extracts structured information from natural-language queries (see example below). |
| `query_search.py` | Main natural-language search program; combines HDC similarity, metadata matching, and sensory matching into a final ranking. |

**Example — `embeddings.py` output:**
```
Generating query hypervector...
Query hypervector shape: (10000,)
```

**Example — `query_parser.py` on `"floral fruity Taiwanese oolong"`:**
```
class:   oolong
country: Taiwan
sensory: [floral, fruity]
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd tea-hdc
```

### 2. Python environment (`uv`)

This project uses [`uv`](https://github.com/astral-sh/uv), a fast Python package and project management tool. It manages Python environments, dependencies, package installation, and running commands — replacing the usual `python -m venv .venv` + `pip install ...` workflow.

### 3. Install dependencies

```bash
uv sync
```

This reads the project configuration and installs the required dependencies.

### 4. Run the project

```bash
uv run python src/query_search.py --query "green tea" --limit 5
```

This runs the program inside the project's managed environment.

---

## 🔎 Example Queries

| Query | Parsed as |
|---|---|
| `"green tea"` | `class: green` |
| `"oolong tea"` | `class: oolong` |
| `"Taiwanese oolong"` | `class: oolong`, `country: Taiwan` |
| `"floral tea"` | `sensory: [floral]` |
| `"fruity tea"` | `sensory: [fruity]` |
| `"floral fruity Taiwanese oolong"` | `class: oolong`, `country: Taiwan`, `sensory: [floral, fruity]` |

```bash
uv run python src/query_search.py --query "green tea" --limit 5
uv run python src/query_search.py --query "oolong tea" --limit 5
uv run python src/query_search.py --query "Taiwanese oolong" --limit 5
uv run python src/query_search.py --query "floral tea" --limit 5
uv run python src/query_search.py --query "fruity tea" --limit 5
uv run python src/query_search.py \
    --query "floral fruity Taiwanese oolong" \
    --limit 5
```

The complex query combines **semantic information + structured metadata + sensory characteristics**.

---

## 📊 Understanding the Search Score

The search produces three component scores.

### 1. HDC Similarity

```
HDC similarity: 0.0118
```

Measures how similar the query hypervector is to a tea hypervector, using **cosine similarity**. Higher similarity means the vectors are more aligned.

### 2. Metadata Score

```
Metadata score: 1.00
```

For `"Taiwanese oolong"`, the system expects `class = oolong` and `country = Taiwan`. If a tea satisfies both, metadata score = 1.00.

### 3. Sensory Score

```
Sensory score: 1.00
```

For `"floral fruity"`, the system checks the tea's `aroma` and `taste` fields for matching sensory terms, grouped by concept:

- **Floral-related:** floral, flower, lavender, jasmine, orchid, rose
- **Fruity-related:** fruity, fruit, peach, apple, berry, citrus

If both requested sensory groups are found, sensory score = 1.00.

### 🧮 Final Score

```
Final Score = 0.30 × HDC similarity
            + 0.50 × Metadata score
            + 0.20 × Sensory score
```

The result is used to rank the teas.

---

## 🧪 Example Output

For the query `"floral fruity Taiwanese oolong"`:

```
Detected query metadata:
  class: oolong
  country: Taiwan
  sensory: ['floral', 'fruity']

score     HDC       metadata   sensory   class     title
----------------------------------------------------------------
0.7035    0.0118    1.00       1.00      oolong    Gui Fei
0.7027    0.0091    1.00       1.00      oolong    Ali Shan - Roasted
0.7019    0.0062    1.00       1.00      oolong    Dong Ding - Charcoal roasted
```

For the top result:
```
0.30 × 0.0118 + 0.50 × 1.00 + 0.20 × 1.00 ≈ 0.7035
```

### 🔬 Why HDC Similarity Can Be Small

A small value like `HDC similarity = 0.0118` is **not a failure** — the final ranking isn't based on HDC similarity alone. It's combined with metadata and sensory scores:

- For structured queries like `"Taiwanese oolong"`, **metadata** contributes strongly.
- For sensory queries like `"floral tea"`, **sensory matching** becomes important.

---

## 🧠 Complete Search Pipeline (Worked Example)

For the query `"floral fruity Taiwanese oolong"`:

1. **Query** — `"floral fruity Taiwanese oolong"`
2. **Generate query hypervector** — text → embedding → 10,000-dimensional hypervector
3. **Parse query** — `class = oolong`, `country = Taiwan`, `sensory = [floral, fruity]`
4. **Load LanceDB** — load tea records and stored hypervectors
5. **Calculate HDC similarity** — cosine similarity between query hypervector and each tea hypervector
6. **Calculate metadata score** — check class, country, roast, oxidation as specified by the query
7. **Calculate sensory score** — search aroma/taste for the requested sensory characteristics
8. **Calculate final score** — `0.30×HDC + 0.50×Metadata + 0.20×Sensory`
9. **Sort** — highest score to lowest
10. **Return Top K** — e.g. `--limit 5` returns the five highest-ranked teas

---

## 🧪 Dataset

The project currently works with a tea dataset containing approximately **166 teas**.

**Tea classes:** black, green, oolong, white, yellow

**Fields include:** country, region, oxidation, roast, aroma, taste, elevation

---

## 🔬 Sensory Search

Sensory information comes primarily from the `aroma` and `taste` fields. For example, a tea's aroma might include:

```
apple blossom
honeysuckle
fresh apricot
wild berries
lemon zest
```

A query like `"fruity tea"` can match terms such as: fruit, peach, apple, berry, citrus. The project groups related sensory words together into concept clusters.

---

## 🎯 Current Project Goal

The primary purpose of this project is **educational and experimental** — demonstrating how:

```
Natural Language
   → Hypervector Representation
   → HDC Similarity + Structured Metadata + Sensory Matching
   → Ranked Search Results
```

can be implemented using Hyperdimensional Computing.

### 🚫 Out of Scope (for now)

- Web UI
- Mobile application
- Authentication
- Cloud deployment
- Production API
- Large-scale distributed databases
- Advanced ranking models

The primary objective is understanding and implementing the underlying HDC concepts, not building a production application.

---

## 📚 Concepts Demonstrated

- Hyperdimensional Computing & hypervectors
- High-dimensional representations
- Vector / cosine similarity
- Text embeddings
- Binding & bundling
- Symbolic representation
- Metadata matching
- Sensory matching
- Natural-language query parsing
- Vector databases (LanceDB)
- Retrieval and ranking
- Python environment management with `uv`

---

## 🚀 Future Improvements

```
Better sensory embeddings
   → Better semantic similarity
   → Improved ranking
   → More natural-language queries
   → Larger tea dataset
   → HDC associative memory
   → More advanced retrieval
```

Other possible experiments:

- Compare HDC search with traditional vector search
- Experiment with different hypervector dimensions
- Test different embedding models
- Improve sensory concept encoding
- Evaluate retrieval accuracy
- Add query evaluation metrics

---

## 🧑‍💻 Learning Objective

The main learning objective is to understand the complete flow:

```
DATA → ENCODING → HYPERVECTORS → BINDING → BUNDLING
     → TEA REPRESENTATION → LANCEDB
     → QUERY → QUERY HYPERVECTOR
     → SIMILARITY → METADATA MATCHING → SENSORY MATCHING
     → FINAL RANKING
```

This project is intended to make the concepts of Hyperdimensional Computing concrete through a working implementation, rather than only theoretical examples.

---

## 📜 License

This project is currently unlicensed.

## 👤 Author

**Harini P** — a hands-on learning project exploring Hyperdimensional Computing, vector representations, and semantic search.
