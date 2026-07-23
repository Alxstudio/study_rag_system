# Study RAG System

A custom-built Retrieval-Augmented Generation (RAG) pipeline in Python, designed to turn certification study notes into targeted practice tests built from the ground up rather than wrapped around a pre-built framework.

## Why this project

Most "AI-powered" study tools just forward your notes to a chatbot and hope for a good answer. This project implements the actual RAG mechanics by hand: text chunking, embedding generation, a custom vector similarity search engine, and a retrieval layer that grounds every generated question in the user's own notes with an LLM call used only for the final quiz-writing step, not for deciding what's relevant.

Built while studying for Microsoft's AI-901 (Azure AI Fundamentals) certification, but designed to be certification-agnostic: any exam's domains and study notes can be loaded in without touching the core pipeline.

## How it works

1. **Chunking** —> study notes are split into paragraph-aware fragments with overlap, preserving semantic continuity across chunk boundaries
2. **Embeddings** —> each chunk is converted into a 384-dimension vector using a multilingual sentence-transformer model, running locally with no external API calls
3. **Vector store** —> a from-scratch cosine-similarity search engine (NumPy, vectorized matrix operations) retrieves the most relevant chunks for a given query
4. **Retrieval** —> queries are embedded and matched against stored chunks, filtered by exam domain when needed
5. **Generation** —> only this final step calls an LLM API (Groq), which writes quiz questions strictly from the retrieved context

## Tech stack

- **Python** —> core RAG pipeline (chunking, embeddings, vector search, retrieval)
- **sentence-transformers** —> local, multilingual embedding generation
- **NumPy** —> vectorized similarity search
- **SQLite** —> persistent storage for content, chunks, and generated questions
- **Streamlit** —> interface
- **Groq API** —> LLM-based quiz generation, grounded in retrieved context
