import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings


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


# Set up AI model

# Model ID
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# Create HF client
client = InferenceClient(
    model=model_id,
    token=hf_api_key
)

print("HF LLaMA-3 8B (API) loaded")

# Store our conversation as messages
chat_history = []

# Combine the query and the relevant document contents
def build_rag_prompt(query, relevant_docs):
    combined_input = f"""Based on the following documents, please answer this question: {query}

    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents.If you can't find the answer in the documents, say:'I don't have enough information to answer that question based on the provided documents.'
    """
    return combined_input

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages=[
            {"role": "system", "content": "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."},
             *chat_history,
            {"role": "user", "content":f"New question: {user_question}"}
        ]
        
        response = client.chat.completions.create(
            messages= messages,
            max_tokens=512,
            temperature=0.2,
        )

        search_question = response.choices[0].message.content
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(search_question)
    
    # print(f"Found {len(relevant_docs)} relevant documents:")
    # for i, doc in enumerate(relevant_docs, 1):
    #     # Show first 2 lines of each document
    #     lines = doc.page_content.split('\n')[:2]
    #     preview = '\n'.join(lines)
    #     print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt
    combined_input = build_rag_prompt(search_question, relevant_docs)
    
    # Step 4: Get the answer
    messages=[
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents and conversation history."},
        {"role": "user", "content":combined_input}
    ]

    response = client.chat.completions.create(
        messages= messages,
        max_tokens=512,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    
    # Step 5: Remember this conversation
    chat_history.append({"role": "user","content": user_question})
    chat_history.append({"role": "assistant","content": answer})
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()