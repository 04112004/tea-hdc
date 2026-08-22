import lancedb

DB_PATH = "data/tea_hypervectors.lance"

db = lancedb.connect(DB_PATH)
table = db.open_table("teas")

df = table.to_pandas()

print("=" * 80)
print("SENSORY DATA INSPECTION")
print("=" * 80)

print()

print("Columns:")
print(df.columns.tolist())

print()

for _, row in df.head(10).iterrows():

    print("-" * 80)

    print("Title:", row["title"])

    print("Aroma:", row.get("aroma"))

    print("Taste:", row.get("taste"))