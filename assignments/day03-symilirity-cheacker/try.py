import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
collection.add(
    ids=["id1", "id2"],
    documents=[
        "This is a document about pineapples",
        "This is a document about oranges"
    ]
)
results = collection.query(
    query_texts=["This is a document about hawaii"],
    n_results=2
)
print(results)