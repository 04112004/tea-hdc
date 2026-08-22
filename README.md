# 🍵 Tea HDC — Hyperdimensional Computing for Natural Language Tea Search

A hands-on implementation of **Hyperdimensional Computing (HDC)** for semantic tea search.

This project demonstrates how natural-language tea queries can be converted into high-dimensional hypervectors and combined with structured tea metadata and sensory information to retrieve relevant teas.

The project uses:

- Python
- Hyperdimensional Computing (HDC)
- 10,000-dimensional hypervectors
- Text embeddings
- Binding and bundling
- LanceDB
- NumPy
- `uv` for Python environment and dependency management

---

# 📌 Project Overview

The goal of this project is to build a tea search system where a user can enter queries such as:

```text
green tea
Taiwanese oolong
floral tea
fruity tea
floral fruity Taiwanese oolong

and retrieve relevant teas from a dataset.

The search system combines three main signals:

HDC similarity
Metadata matching
Sensory matching

The final ranking is calculated using these signals.

🧠 What is Hyperdimensional Computing?

Hyperdimensional Computing represents information using very large vectors called hypervectors.

In this project, every hypervector has:

10,000 dimensions

For example:

[1, -1, 1, 1, -1, ...]

with 10,000 values.

Instead of representing information using a small number of features, HDC represents information in a very high-dimensional space.

This provides useful properties such as:

distributed representation
robustness
similarity-based computation
symbolic operations using vector algebra
binding
bundling
🧩 Core HDC Operations
1. Hypervector

A hypervector is the basic representation used by the HDC system.

For example:

Tea
 ↓
Hypervector
 ↓
10,000 dimensions

A tea can have many attributes:

class = oolong
country = Taiwan
oxidation = medium
roast = none

These attributes can be represented using individual hypervectors.

🔗 Binding

Binding combines two pieces of information into a new representation.

Conceptually:

COUNTRY ⊗ TAIWAN

produces a new hypervector representing:

country = Taiwan

The binding operation allows relationships between values to be represented.

For example:

CLASS ⊗ OOLONG
COUNTRY ⊗ TAIWAN
ROAST ⊗ LIGHT

Each produces a different hypervector.

📦 Bundling

Bundling combines multiple hypervectors into one representation.

For example:

CLASS ⊗ OOLONG
+
COUNTRY ⊗ TAIWAN
+
ROAST ⊗ LIGHT

can be bundled into a single tea representation.

Conceptually:

Tea
 |
 +-- class
 +-- country
 +-- oxidation
 +-- roast
 +-- elevation
 +-- sensory information

The result is one high-dimensional tea hypervector.

🧠 Tea Hypervector vs Individual Hypervector

An important distinction in this project is:

Individual hypervectors

These represent individual concepts.

Examples:

OOLONG
TAIWAN
GREEN
FLORAL
FRUITY
LIGHT_ROAST
Tea hypervector

A tea hypervector represents the combination of information belonging to one tea.

For example:

Li Shan

may contain information related to:

class = oolong
country = Taiwan
region = Li Shan
oxidation = medium
roast = none
elevation = 2000m
sensory characteristics

These individual pieces are combined to create the representation stored for that tea.

🗄️ LanceDB

The project uses LanceDB as the local vector database.

The database stores the tea records together with their generated hypervectors and metadata.

Conceptually:

Tea Dataset
     |
     ↓
Generate hypervectors
     |
     ↓
Tea hypervector
     |
     ↓
LanceDB

A record can contain information such as:

id
title
class
country
region
oxidation
roast
aroma
taste
elevation
hypervector

The database therefore stores both:

tea information
its numerical hypervector representation
🔄 Overall Data Flow

The complete pipeline is approximately:

Tea Dataset
     |
     ↓
Data Loading
     |
     ↓
Text / Attribute Encoding
     |
     ↓
Individual Hypervectors
     |
     ↓
Binding
     |
     ↓
Bundling
     |
     ↓
Tea Hypervector
     |
     ↓
LanceDB

For a search query:

User Query
     |
     ↓
Query Parser
     |
     ├── Metadata
     |
     └── Sensory terms
     |
     ↓
Text → Hypervector
     |
     ↓
Compare with stored tea hypervectors
     |
     ↓
HDC similarity
     +
Metadata score
     +
Sensory score
     |
     ↓
Final ranking
     |
     ↓
Top K teas
📁 Project Structure
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
📂 Important Files
data_loader.py

Responsible for loading the tea dataset.

The dataset contains information such as:

id
source_url
image
class
title
description
country
region
oxidation
roast
aroma
taste
elevation_meters
elevation_confidence
embeddings.py

Responsible for converting text into vector representations.

The query:

floral fruity Taiwanese oolong

is converted into a 10,000-dimensional query hypervector.

Example output:

Generating query hypervector...
Query hypervector shape: (10000,)
hypervectors.py

Contains the fundamental HDC hypervector operations.

This is where the mathematical representation of hypervectors and operations such as binding/bundling are handled.

encoder.py

Responsible for encoding tea information into hypervectors.

Conceptually:

Tea attributes
      |
      ↓
Encode attributes
      |
      ↓
Individual hypervectors
      |
      ↓
Binding / Bundling
      |
      ↓
Tea hypervector
build_hypervectors.py

Builds the hypervector representation for the dataset.

The general process is:

Dataset
   ↓
Read tea
   ↓
Encode attributes
   ↓
Create tea hypervector
   ↓
Store result
query_parser.py

Extracts structured information from natural-language queries.

For example:

floral fruity Taiwanese oolong

becomes:

class:
    oolong

country:
    Taiwan

sensory:
    floral
    fruity
query_search.py

This is the main natural-language search program.

It combines:

HDC similarity
Metadata matching
Sensory matching

and produces the final ranking.

⚙️ Installation
1. Clone the repository
git clone <your-repository-url>

Move into the project:

cd tea-hdc
🐍 Python Environment

This project uses uv.

What is uv?

uv is a fast Python package and project management tool.

It can manage:

Python environments
dependencies
package installation
running Python commands

Instead of manually doing:

python -m venv .venv

and then:

pip install ...

the project can use:

uv

to manage the environment and dependencies.

📦 Install Dependencies

After entering the project:

uv sync

This reads the project configuration and installs the required dependencies.

▶️ Running the Project

The project uses:

uv run python

For example:

uv run python src/query_search.py --query "green tea" --limit 5

This runs the Python program inside the project's managed environment.

🔎 Example Queries
Green tea
uv run python src/query_search.py --query "green tea" --limit 5

Example result:

Detected query metadata:
  class: green

The system detects:

green

as the tea class.

Oolong Tea
uv run python src/query_search.py --query "oolong tea" --limit 5

The parser detects:

class: oolong
Taiwanese Oolong
uv run python src/query_search.py --query "Taiwanese oolong" --limit 5

The query is interpreted as:

class: oolong
country: Taiwan

This allows metadata matching to improve the ranking.

Floral Tea
uv run python src/query_search.py --query "floral tea" --limit 5

The query parser detects:

sensory:
    floral

The system then examines the tea's aroma and taste information.

Fruity Tea
uv run python src/query_search.py --query "fruity tea" --limit 5

The parser detects:

sensory:
    fruity
Complex Query
uv run python src/query_search.py \
    --query "floral fruity Taiwanese oolong" \
    --limit 5

The system detects:

class: oolong
country: Taiwan
sensory:
    floral
    fruity

This is an example of combining:

semantic information
+
structured metadata
+
sensory characteristics
📊 Understanding the Search Score

The search produces three important scores.

1. HDC Similarity

Example:

HDC similarity: 0.0118

This measures how similar the query hypervector is to the tea hypervector.

The project uses cosine similarity.

Conceptually:

Query Hypervector
        |
        | cosine similarity
        ↓
Tea Hypervector

Higher similarity means the vectors are more aligned.

2. Metadata Score

Example:

Metadata score: 1.00

For:

Taiwanese oolong

the system expects:

class = oolong
country = Taiwan

If a tea satisfies both:

oolong ✓
Taiwan ✓

then:

Metadata score = 1.00
3. Sensory Score

Example:

Sensory score: 1.00

For:

floral fruity

the system checks the tea's:

aroma
taste

and searches for matching sensory terms.

For example, floral-related terms can include:

floral
flower
lavender
jasmine
orchid
rose

Fruity-related terms can include:

fruity
fruit
peach
apple
berry
citrus

If both requested sensory groups are found:

floral ✓
fruity ✓

then:

Sensory score = 1.00
🧮 Final Score

The current implementation calculates:

Final Score =
    0.30 × HDC similarity
  + 0.50 × Metadata score
  + 0.20 × Sensory score

Therefore:

Final Score =
    HDC contribution
    +
    Metadata contribution
    +
    Sensory contribution

The result is then used to rank the teas.

🧪 Example Output

For:

floral fruity Taiwanese oolong

the system can produce:

Detected query metadata:
  class: oolong
  country: Taiwan
  sensory: ['floral', 'fruity']

Example:

score     HDC       metadata   sensory   class     title
----------------------------------------------------------------
0.7035    0.0118    1.00       1.00      oolong    Gui Fei
0.7027    0.0091    1.00       1.00      oolong    Ali Shan - Roasted
0.7019    0.0062    1.00       1.00      oolong    Dong Ding - Charcoal roasted

The first result has:

HDC similarity = 0.0118
Metadata score = 1.00
Sensory score = 1.00

Therefore:

0.30 × 0.0118
+
0.50 × 1.00
+
0.20 × 1.00

which produces approximately:

0.7035
🔬 Why HDC Similarity Can Be Small

It is important not to interpret:

HDC similarity = 0.0118

as a failure.

The final ranking is not based only on HDC similarity.

The system combines:

HDC
+
metadata
+
sensory information

For structured queries such as:

Taiwanese oolong

metadata can contribute strongly to the final ranking.

For sensory queries such as:

floral tea

sensory matching becomes important.

🧠 Complete Search Pipeline

Suppose the user enters:

floral fruity Taiwanese oolong
Step 1 — Query
floral fruity Taiwanese oolong
Step 2 — Generate query hypervector
text
 ↓
embedding
 ↓
10,000-dimensional hypervector
Step 3 — Parse query

The parser identifies:

class = oolong
country = Taiwan
sensory = floral, fruity
Step 4 — Load LanceDB

The system loads the tea records and their stored hypervectors.

Step 5 — Calculate HDC similarity

For every tea:

query hypervector
        ↓
cosine similarity
        ↓
tea hypervector
Step 6 — Calculate metadata score

Check:

class
country
roast
oxidation

depending on what the query specifies.

Step 7 — Calculate sensory score

Search:

aroma
taste

for the requested sensory characteristics.

Step 8 — Calculate final score
Final =
0.30 × HDC
+
0.50 × Metadata
+
0.20 × Sensory
Step 9 — Sort

Results are sorted from highest score to lowest score.

Step 10 — Return Top K

If:

--limit 5

the system returns the five highest-ranked teas.

🧪 Dataset

The project currently works with a tea dataset containing approximately:

166 teas

Tea classes include:

black
green
oolong
white
yellow

The dataset also contains information such as:

country
region
oxidation
roast
aroma
taste
elevation
🔬 Sensory Search

Sensory information comes primarily from:

aroma
taste

For example, a tea may contain:

Aroma:
    apple blossom
    honeysuckle
    fresh apricot
    wild berries
    lemon zest

The query:

fruity tea

can therefore match terms such as:

fruit
peach
apple
berry
citrus

The project groups related sensory words together.

🎯 Current Project Goal

The primary purpose of this project is educational and experimental.

It demonstrates how:

Natural Language
       ↓
Hypervector Representation
       ↓
HDC Similarity
       +
Structured Metadata
       +
Sensory Matching
       ↓
Ranked Search Results

can be implemented using Hyperdimensional Computing.

🚫 What This Project Does Not Currently Focus On

The current implementation intentionally focuses on understanding the HDC pipeline rather than building a production application.

It does not currently focus on:

Web UI
Mobile application
Authentication
Cloud deployment
Production API
Large-scale distributed databases
Advanced ranking models

The primary objective is to understand and implement the underlying HDC concepts.

📚 Concepts Demonstrated

This project provides hands-on implementation of:

Hyperdimensional Computing
Hypervectors
High-dimensional representations
Vector similarity
Cosine similarity
Text embeddings
Binding
Bundling
Symbolic representation
Metadata matching
Sensory matching
Natural-language query parsing
Vector databases
LanceDB
Retrieval and ranking
Python environment management with uv
🚀 Future Improvements

Possible future improvements include:

Better sensory embeddings
        ↓
Better semantic similarity
        ↓
Improved ranking
        ↓
More natural-language queries
        ↓
Larger tea dataset
        ↓
HDC associative memory
        ↓
More advanced retrieval

Other possible experiments:

Compare HDC search with traditional vector search
Experiment with different hypervector dimensions
Test different embedding models
Experiment with HDC similarity
Improve sensory concept encoding
Evaluate retrieval accuracy
Add query evaluation metrics
🧑‍💻 Learning Objective

The main learning objective is to understand the complete flow:

DATA
 ↓
ENCODING
 ↓
HYPERVECTORS
 ↓
BINDING
 ↓
BUNDLING
 ↓
TEA REPRESENTATION
 ↓
LANCEDB
 ↓
QUERY
 ↓
QUERY HYPERVECTOR
 ↓
SIMILARITY
 ↓
METADATA MATCHING
 ↓
SENSORY MATCHING
 ↓
FINAL RANKING

This project is intended to make the concepts of Hyperdimensional Computing concrete through a working implementation rather than only theoretical examples.

📜 License

Add your preferred license here.

For example:

MIT License
👤 Author

Harini

A hands-on learning project exploring:

Hyperdimensional Computing
+
Vector Representations
+
Semantic Search

### One important correction to what we discussed earlier

Your current project **does store the hypervector in LanceDB**. It isn't storing only the tea metadata. Your output/code shows a `hypervector` column, so the conceptual flow is:

```text
Individual hypervectors
        ↓
Binding + Bundling
        ↓
Tea hypervector
        ↓
Store tea metadata + tea hypervector
        ↓
LanceDB

Then at search time:

User query
    ↓
Query hypervector
    ↓
Compare against stored tea hypervectors
    ↓
HDC similarity
