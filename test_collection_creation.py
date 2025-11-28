import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from dotenv import load_dotenv

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = "user-data-collection"

print("Testing safe collection creation...\n")
print("Using QDRANT_URL:", qdrant_url)
print("Using QDRANT_API_KEY present:", bool(qdrant_api_key))

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)

# List collections before
try:
    collections_before = client.get_collections()
    print(f"Collections before: {len(collections_before.collections)}")
except Exception as e:
    print("Error listing collections:", repr(e))
    collections_before = None

# Try to get collection (should fail if it doesn't exist)
try:
    info = client.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception as e_get:
    # Create the collection
    print(f"Collection '{collection_name}' does not exist. Creating it...")
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE)
        )
        print(f"✓ Collection '{collection_name}' created successfully!")
    except Exception as e_create:
        print(f"✗ Failed to create collection: {repr(e_create)}")

# List collections after
try:
    collections_after = client.get_collections()
    print(f"Collections after: {len(collections_after.collections)}")
    for coll in collections_after.collections:
        print(f"  - {coll.name}")
except Exception as e:
    print("Error listing collections:", repr(e))
