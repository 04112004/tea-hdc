import lancedb


DB_PATH = "data/tea_hypervectors.lance"


print("=" * 60)
print("LANCEDB DATABASE TEST")
print("=" * 60)


# Connect to database
db = lancedb.connect(DB_PATH)

print("\nDatabase connected successfully!")


# List tables
tables = db.list_tables()

print("\nTables:")
print(tables)


# Open teas table
table = db.open_table("teas")

print("\nTable opened successfully!")


# Number of records
count = table.count_rows()

print("\nNumber of records:", count)


# Get first 5 records
df = table.to_pandas().head(5)

print("\nFirst 5 teas:")
print(df[[
    "id",
    "title",
    "class",
    "country"
]])


# Check hypervector
first_vector = df.iloc[0]["hypervector"]

print("\nFirst hypervector:")
print("Shape:", first_vector.shape)
print("First 10 values:", first_vector[:10])


print("\n" + "=" * 60)
print("DATABASE TEST COMPLETE")
print("=" * 60)