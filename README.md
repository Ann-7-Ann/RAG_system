# 🧗 Rock Climbing RAG Assistant

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline using **Streamlit** for the interface, combined with:  

- **LangChain**  
- **Ollama (LLaMA 3)**  
- **HuggingFace embeddings**  
- **Wikipedia** as a knowledge source  
- **In-memory vector storage**  

The system loads content from the Rock Climbing Wikipedia page, chunks and embeds it, retrieves relevant context, and uses a local LLM to answer user questions through an interactive Streamlit chat interface.
