# RAG Customer Support Assistant

This project implements a simple Retrieval-Augmented Generation (RAG) system using a PDF knowledge base.

## Tech Stack
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq LLM

## Features
- Load and process PDF documents  
- Store embeddings in vector database  
- Retrieve relevant context  
- Generate answers using Groq  

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Create .env file:

GROQ_API_KEY=your_api_key_here

3. Run:

python main.py