from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()

# -----------------------------
# LOAD PDF
# -----------------------------
loader = PyPDFLoader("data/ml_notes.pdf")
documents = loader.load()

# -----------------------------
# SPLIT
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)

# -----------------------------
# EMBEDDINGS + VECTOR DB
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# GROQ LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# -----------------------------
# ASK QUESTION LOOP
# -----------------------------
while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    if not docs:
        print("❌ No relevant documents found")
        continue

    context = "\n\n".join([doc.page_content for doc in docs])

    response = llm.invoke(f"""
You are an AI assistant.

Answer using the context below.
If not enough info, give a general answer.

---------------------
Context:
{context}
---------------------

Question:
{query}

Answer:
""")

    print("\n🤖 AI Answer:\n")
    print(response.content)