# 🚀 OmniRAG — Multimodal Enterprise RAG Platform

OmniRAG is a production-oriented multimodal Retrieval-Augmented Generation
(RAG) platform designed for intelligent document understanding, semantic
search, grounded question answering, and enterprise knowledge retrieval.

The platform processes documents using Unstructured.io, stores embeddings
in Qdrant, combines dense semantic retrieval with BM25 lexical search,
applies Reciprocal Rank Fusion (RRF) and cross-encoder reranking, and
generates grounded answers using Groq-hosted LLMs.

The system also integrates LLMOps and MLOps practices including:

- 🔎 RAGAS evaluation
- 📈 MLflow experiment tracking
- 🔭 Arize Phoenix observability
- 🧪 PyTest testing
- ⚙️ GitHub Actions CI/CD
- 🐳 Docker-based deployment

## 🚀 Key Features

| Feature | Description |
|---|---|
| 📄 Multimodal Ingestion | Processes PDF content including text, tables, and images |
| 🧩 Intelligent Chunking | Creates retrieval-ready chunks with metadata |
| 🔢 Embeddings | Converts document content into vector representations |
| 🗄️ Vector Search | Qdrant-powered semantic retrieval |
| 🔤 BM25 Retrieval | Keyword and exact-term retrieval |
| 🔀 RRF Fusion | Combines dense and lexical rankings |
| 🎯 Cross-Encoder | Reranks retrieved candidates |
| 🤖 LLM Generation | Generates grounded answers using Groq |
| 📚 Citations | Returns source document and page metadata |
| 📊 RAGAS | Evaluates RAG quality |
| 🔭 Phoenix | Traces RAG execution |
| 📈 MLflow | Tracks experiments and runtime metrics |
| 🧪 PyTest | Automated testing |
| 🔄 GitHub Actions | Continuous integration |
| 🐳 Docker | Containerized deployment |

## 🏗️ System Architecture

## 🔄 RAG Pipeline

1. User uploads a document
2. Unstructured.io extracts document elements
3. Content is chunked with metadata
4. Embeddings are generated
5. Chunks are stored in Qdrant
6. User submits a natural-language query
7. Dense semantic retrieval is performed
8. BM25 lexical retrieval is performed
9. Reciprocal Rank Fusion combines both rankings
10. Cross-encoder reranks candidate chunks
11. Relevant context is constructed
12. Groq LLM generates a grounded answer
13. Source citations are returned
14. Phoenix traces the request
15. MLflow records runtime information
16. RAGAS evaluates retrieval and generation quality

## 📊 RAGAS Evaluation

The RAG pipeline is evaluated using four core RAGAS metrics:

| Metric | Score |
|---|---:|
| Faithfulness | 0.4333 |
| Context Precision | 0.3333 |
| Context Recall | 0.8889 |
| Answer Relevancy | 0.5573 |

### Evaluation Interpretation

- High Context Recall indicates that the required information is generally
  retrieved successfully.
- Faithfulness and Context Precision indicate opportunities for further
  improving context filtering and answer grounding.
- Answer Relevancy measures how closely generated responses align with the
  user's question.

## 🔭 Observability & LLMOps

### Arize Phoenix

Phoenix provides tracing across the RAG pipeline, including:

- Query execution
- BM25 refresh
- Retrieval
- Context construction
- LLM generation
- Citation generation
- Latency
- Errors

### MLflow

MLflow tracks:

- Model
- Query
- Retrieval configuration
- Retrieval latency
- Generation latency
- Total latency
- Number of retrieved chunks
- Citations
- RAGAS evaluation metrics

## 🐳 Docker Deployment

OmniRAG uses Docker for reproducible deployment.

Core services include:

- FastAPI application
- Qdrant vector database
- Arize Phoenix observability
- MLflow tracking

Build the API image:

```bash
docker build -t omnirag-api .

docker compose up -d

docker compose ps

curl http://localhost:8000/health


---

# Screenshots


1. **Streamlit OmniRAG UI**
2. **Document upload**
3. **Chat answer**
4. **Citation/source information**
5. **Phoenix trace**

6. **MLflow experiment**
7. **RAGAS dashboard**
8. **GitHub Actions CI passing**



```markdown
## 📸 Screenshots
