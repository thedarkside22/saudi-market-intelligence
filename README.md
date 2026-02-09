# 📈 Saudi Market Intelligence (RAG)

A specialized Retrieval-Augmented Generation (RAG) application designed to analyze Saudi Stock Market (Tadawul) data. This tool ingests complex financial documents (Annual Reports in PDF and Market Summaries in HTML) to provide accurate, source-cited financial insights in both Arabic and English.

## 🧠 Key Features

* **Hybrid Search Strategy:** Combines **Semantic Search** (for narrative/strategy understanding) with **Keyword/HTML Search** (for precise price lookups).
* **Two-Stage Retrieval:** Utilizes **NVIDIA's Reranker (Llama 3.2)** to refine retrieval results, ensuring high precision for specific queries (e.g., stock prices).
* **Multilingual Support:** Fully capable of processing and responding in both Arabic and English.
* **Source Awareness:** Intelligently distinguishes between "Market Data" (HTML) and "Strategic Reports" (PDF) to answer context-aware questions.

## 🛠️ Tech Stack

* **LLM:** Llama 3.1 70B (via NVIDIA NIM)
* **Vector Database:** Milvus (Zilliz Cloud)
* **Embeddings:** NVIDIA NeMo Retriever (`nv-embedqa-e5-v5`)
* **Reranker:** NVIDIA Llama 3.2 Reranker
* **Framework:** LangChain & Streamlit
* **Language:** Python 3.10+

## 🏗️ System Architecture

The system uses a "Double-Dip" retrieval method to ensure no data is lost between file formats:

```mermaid
graph TD
    User[User Query] --> Retriever
    Retriever -->|Fetch Top 10| PDF[PDF Reports (Strategy)]
    Retriever -->|Fetch Top 30| HTML[HTML Reports (Prices)]
    PDF & HTML --> Combined[Candidate Pool (40 docs)]
    Combined --> Reranker[NVIDIA Llama 3.2 Reranker]
    Reranker -->|Top 5 Contexts| LLM[Llama 3.1 70B]
    LLM --> Answer[Final Answer]