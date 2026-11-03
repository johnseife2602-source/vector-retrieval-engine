from IndexFlat_engine import VectorStore
from IndexMMap_engine import MMapVectorStore

print("=== 1. Building & Saving Database with IndexFlat ===")
flat_store = VectorStore(dim=3)
flat_store.insert("doc_dog",    [0.9, 0.1, 0.0], {"text": "Golden retriever puppy playing in the park"})
flat_store.insert("doc_cat",    [0.8, 0.2, 0.1], {"text": "Kitten sleeping on the sofa"})
flat_store.insert("doc_server", [0.1, 0.9, 0.0], {"text": "Linux cloud server with 64GB RAM"})
flat_store.insert("doc_laptop", [0.0, 0.8, 0.1], {"text": "MacBook Pro with M3 processor"})
flat_store.insert("doc_pizza",  [0.0, 0.1, 0.9], {"text": "Pepperoni pizza with extra cheese"})
flat_store.insert("doc_burger", [0.1, 0.0, 0.8], {"text": "Cheeseburger with french fries"})

flat_store.save()
print("Saved to disk successfully!")

print("\n=== 2. Searching with MMapVectorStore (Zero Floats in RAM!) ===")
query = [0.0, 0.95, 0.05]  # Tech / Hardware query

# Use context manager: opens mmap on enter, closes on exit!
with MMapVectorStore() as mmap_store:
    print(f"MMap Store opened! Total docs on disk: {mmap_store.count}, Dim: {mmap_store.dim}")
    
    results = mmap_store.search(query, top_k=2, metric="cosine")
    print("\nTop 2 Tech Matches (via MMap Out-of-Core):")
    for r in results:
        print(f"  [{r['id']}] Score: {r['score']:.4f} | Text: {r['metadata']['text']}")

print("\nMMap successfully closed cleanly!")