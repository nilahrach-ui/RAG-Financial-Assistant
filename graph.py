from typing import TypedDict
from langgraph.graph import StateGraph, END

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ---------- STATE ----------
class State(TypedDict):
    question: str
    context: str
    answer: str


# ---------- MODEL ----------
llm = ChatOllama(model="llama3")


# ---------- DB ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# ---------- NODE 1: RETRIEVE ----------
def retrieve(state):
    docs = retriever.invoke(state["question"])

    context = "\n\n".join([d.page_content for d in docs])

    return {"context": context}


# ---------- NODE 2: ANSWER ----------
def answer(state):
    prompt = f"""
    Answer the question using ONLY this context:

    {state['context']}

    Question: {state['question']}
    """

    res = llm.invoke(prompt)

    return {"answer": res.content}


# ---------- GRAPH ----------
graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

app = graph.compile()