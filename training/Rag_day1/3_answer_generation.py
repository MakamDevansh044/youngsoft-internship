import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()
hf_api_key = os.getenv("HF_API_KEY")


persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

query = "How much did Microsoft pay to acquire GitHub?"

# retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
    }
)


relevant_docs = retriever.invoke(query)

# print(f"User Query: {query}")
# Display results
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")



##################################################
####################LLM###########################
##################################################


# Combine the query and the relevant document contents
def build_rag_prompt(query, relevant_docs):
    combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents.If you can't find the answer in the documents, say:"I don't have enough information to answer that question based on the provided documents."
"""
    return combined_input


# Model ID
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Create HF client
client = InferenceClient(
    model=model_id,
    token=hf_api_key
)

print("HF LLaMA-3 8B (API) loaded")

# Build RAG prompt
prompt = build_rag_prompt(query, relevant_docs)

# System prompt (important for LLaMA-3)
system_prompt = "You are a helpful assistant that answers questions strictly using the provided documents."

# Invoke model
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ],
    max_tokens=512,
    temperature=0.2,
)

# Output
print("\n--- User Query ---")
print(query)
print("\n--- Generated Response ---")
print(response.choices[0].message.content)
