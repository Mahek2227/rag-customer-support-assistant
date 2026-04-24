# RAG Customer Support Assistant

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that answers user queries based on a PDF knowledge base using semantic search and a Large Language Model (LLM).

---

## Tech Stack
- Python  
- LangChain  
- ChromaDB (Vector Database)  
- HuggingFace Embeddings  
- Groq LLM (llama-3.1-8b-instant)

---

## Features
- PDF document ingestion  
- Text chunking and embedding  
- Vector-based semantic retrieval  
- Context-aware answer generation  
- Fast inference using Groq  

---

## Project Structure
```
rag_project/
 ├── src/
 │    ├── rag_pipeline.py
 │    ├── graph_flow.py
 │    └── __init__.py
 ├── data/
 │    └── ml_notes.pdf
 ├── main.py
 ├── requirements.txt
 ├── .gitignore
 └── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add API key
Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_api_key_here
```

### 3. Run the project
```bash
python main.py
```

---

## Usage
Enter a question related to the PDF content.  
The system retrieves relevant context and generates an answer.

---

## Example
```
Ask a question: What is machine learning?

AI Answer:
Machine learning is a branch of artificial intelligence that enables systems to learn from data...
```

---

## Notes
- Do not upload `.env` file to GitHub  
- Remove `db/` or `chroma_db/` before pushing  
- Ensure internet connection for Groq API  

---

## Future Improvements
- Add LangGraph workflow (routing + HITL)  
- Build UI (Streamlit/Web app)  
- Support multiple documents  
- Improve retrieval accuracy  

---

## Author
Mahek Sultana  
B.Tech Final Year Student
