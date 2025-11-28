import os
from qdrant_client import QdrantClient, http
from dotenv import load_dotenv

load_dotenv()  # only if you use .env locally

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = "user-data-collection"

print("Using QDRANT_URL:", qdrant_url)
print("Using QDRANT_API_KEY present:", bool(qdrant_api_key))

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Try to list collections
try:
    collections = client.get_collections()
    print("Collections list:", collections)
except Exception as e:
    print("Error listing collections:", repr(e))

# Try to get specific collection info
try:
    info = client.get_collection(collection_name=collection_name)
    print("Collection info:", info)
except Exception as e:
    print("Error getting collection:", repr(e))

# Optionally check approximate points (if collection exists)
try:
    count = client.count(collection_name=collection_name, exact=True).count
    print("Point count:", count)
except Exception as e:
    print("Error counting points:", repr(e))