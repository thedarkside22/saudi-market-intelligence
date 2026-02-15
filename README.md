# 📈 Saudi Market Intelligence Unit (SMIU)

A Hybrid RAG (Retrieval-Augmented Generation) system designed for the Saudi Stock Market (Tadawul). unlike standard RAG chatbots, SMIU features a **Dual-Path Architecture** that intelligently switches between **Real-Time API Data** (for prices) and **Vector Search** (for strategy & analysis), ensuring accurate financial insights without hallucinations.

## 🚀 Key Features (v2.0)

* **🧠 Intelligent Intent Classification:** Automatically detects if a user wants *Live Data* ("What is Aramco's price?") or *Deep Analysis* ("What are STC's risks?").
* **⚡ Hybrid Architecture:**
    * **Path A (Live):** Connects to a custom Tadawul API for sub-second stock prices.
    * **Path B (Research):** Uses semantic search on Annual Reports (PDFs) and Historical Data (HTML).
* **🚫 Hallucination Prevention:** Implements **Dynamic Prompt Injection** to force the LLM to prioritize API data over outdated document context when answering price questions.
* **📊 Strategy Impact Analysis:** Correlates qualitative strategic initiatives (found in PDFs) with quantitative market performance (found via API).
* **🏢 Entity Resolution Registry:** A semantic vector index that resolves company names (e.g., "The oil giant", "2222", "Aramco") to their correct stock symbols.

## 🛠️ Tech Stack

* **LLM:** Llama 3.1 70B (via NVIDIA NIM)
* **Vector Database:** Milvus (Zilliz Cloud)
* **Embeddings:** NVIDIA NeMo Retriever (`nv-embedqa-e5-v5`)
* **Reranker:** NVIDIA Llama 3.2 Reranker
* **Backend:** Python 3.10+, LangChain
* **Frontend:** Streamlit

## 🏗️ System Architecture (v2.0)

The system uses a **Router-Based** workflow to ensure the right tool is used for the right job:

```mermaid
graph TD
    User[User Query] --> Intent{Intent Classifier}

    %% Path A: Live Data & Entity Resolution
    Intent -->|Live Price / Comparison| Registry[Entity Registry & Resolver]
    Registry --> API[Tadawul Live API]
    
    %% Path B: Document Retrieval
    Intent -->|Analysis / History / Comparison| Retriever[Milvus Vector Store]
    Retriever -->|Strategy| PDF[Annual Reports PDF]
    Retriever -->|History| HTML[Market Summaries HTML]
    
    %% Synthesis Layer
    PDF & HTML --> Reranker[NVIDIA Reranker]
    Reranker --> Context[Context Window]
    API --> Context
    
    %% Generation
    Context -->|Dynamic Instructions| LLM[Llama 3.1 70B]
    LLM --> Answer[Final Answer]