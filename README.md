# RAG project

# 🤖 Chat With Your PDF – AI-Powered Q&A App
> Upload any PDF. Ask anything. Get intelligent answers using RAG (Retrieval-Augmented Generation) powered by LangChain, Hugging Face, and FAISS.

## 🚀 Features

- 📄 Upload any textbook, research paper, or course notes (PDF)
- 💬 Ask questions about the uploaded document
- 🔍 Contextual answers using vector-based retrieval (FAISS)
- 🧠 Embeddings via `intfloat/e5-small-v2`
- 🧾 Parsing with `pdfplumber`
- 🔁 Multi-turn chat history using Streamlit session state
- ⚙️ Built with LangChain and Hugging Face LLM (`Llama-3.1-Nemotron-70B-Instruct`)

## 🛠️ Tech Stack

| Tool/Library           | Purpose                                 |
|------------------------|-----------------------------------------|
| `Streamlit`            | Frontend UI                             |
| `LangChain`            | RAG pipeline and chaining logic         |
| `HuggingFace Transformers` | LLM access and embeddings           |
| `FAISS`                | Vector store for semantic search        |
| `pdfplumber`           | Extract text from PDFs                  |
| `intfloat/e5-small-v2` | Lightweight sentence embedding model    |
| `nvidia/Llama-3.1-Nemotron-70B-Instruct` | Open-source instruction-following LLM |
