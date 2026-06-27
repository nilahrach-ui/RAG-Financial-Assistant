from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# نفس embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# تحميل DB لي درتي
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# test
query = "Quel est le bénéfice net 2024 ?"

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Doc {i+1} ---")
    print(doc.page_content[:300])