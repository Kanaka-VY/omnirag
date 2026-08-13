# OmniRAG — Multimodal Enterprise RAG Platform

OmniRAG is a multimodal enterprise Retrieval-Augmented Generation (RAG)
platform designed for intelligent document search and question answering.

It supports document ingestion, semantic retrieval, lexical retrieval,
hybrid retrieval, cross-encoder reranking, LLM-based generation,
citations, evaluation, monitoring, and experiment tracking.

---

## 🚀 Key Features

- Multimodal document ingestion
- PDF and document processing using Unstructured
- Text and structured-content extraction
- Semantic vector search using Qdrant
- BM25 lexical retrieval
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Context-aware LLM generation
- Source citations
- FastAPI backend
- Streamlit user interface
- RAGAS evaluation
- Arize Phoenix tracing
- MLflow experiment tracking
- Automated evaluation pipeline
- Retrieval and generation latency tracking

---

## 🏗️ Architecture

                         DOCUMENTS
                            │
              ┌─────────────┴─────────────┐
              │                           │
           PDFs                      Images / Tables
              │                           │
              └─────────────┬─────────────┘
                            ▼
                    Unstructured.io
                            │
                            ▼
                   Extraction + Parsing
                            │
                            ▼
                     Chunking + Metadata
                            │
                            ▼
                       Embeddings
                            │
                            ▼
                         Qdrant
                            │
                            │
                            ▼
                       User Query
                            │
                            ▼
                    Query Processing
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       Dense Retrieval               BM25 Retrieval
              │                           │
              └─────────────┬─────────────┘
                            ▼
                     RRF Fusion
                            │
                            ▼
                  Cross-Encoder Reranker
                            │
                            ▼
                  Context Construction
                            │
                            ▼
                        Groq LLM
                            │
                    ┌───────┴───────┐
                    ▼               ▼
                  Answer        Citations


          ┌────────────────────────────────┐
          │         Observability          │
          │                                │
          │ Phoenix → Tracing              │
          │ MLflow  → Experiment Tracking  │
          │ RAGAS   → Evaluation           │
          └────────────────────────────────┘

##Project Structure

OmniRAG/
│
├── app/
│   ├── api/
│   │   ├── main.py
│   │   └── routes/
│   │       └── chat.py
│   │
│   └── ui/
│       └── streamlit_app.py
│
├── src/
│   │
│   ├── ingestion/
│   │   └── ...
│   │
│   ├── multimodal/
│   │   └── ...
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── vectorstore/
│   │   ├── client.py
│   │   └── repository.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── lexical.py
│   │   ├── hybrid.py
│   │   ├── reranker.py
│   │   ├── config.py
│   │   └── search_pipeline.py
│   │
│   ├── generation/
│   │   ├── generator.py
│   │   ├── citations.py
│   │   └── providers/
│   │
│   ├── evaluation/
│   │   ├── rag_runner.py
│   │   ├── ragas_evaluator.py
│   │   ├── evaluation_runner.py
│   │   └── ragas_dashboard.py
│   │
│   ├── monitoring/
│   │   ├── phoenix.py
│   │   └── mlflow_tracking.py
│   │
│   └── config/
│       └── settings.py
│
├── scripts/
│   ├── evaluate/
│   │   └── generate_evaluation_records.py
│   └── ...
│
├── data/
│   └── evaluation/
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md