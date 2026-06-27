from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = PyPDFLoader("data/rapport.pdf")

documents = loader.load()

print("Nombre de pages :", len(documents))

print("\nPremier extrait :")
print(documents[0].page_content[:1000])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("\nNombre de chunks :", len(chunks))

print("\nPremier chunk :")
print(chunks[0].page_content[:500])