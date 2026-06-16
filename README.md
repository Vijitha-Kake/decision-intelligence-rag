# Decision Intelligence Platform — Manufacturing Defect Analysis (RAG)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

A manufacturing plant generates thousands of pages of unstructured
documents — production logs, maintenance reports, supplier quality
records, defect tickets. When a defect occurs on the line,
engineers waste hours manually searching through these documents
for root causes.

This project builds a **Retrieval-Augmented Generation (RAG)
system** that lets engineers ask plain-English questions and
instantly get context-grounded answers with source attribution —
reducing root cause investigation time significantly.

> This repository demonstrates the GenAI/RAG methodology applied
> professionally in manufacturing analytics. Sample documents used
> here are synthetic — no proprietary operational data is included.

---

## 🚀 Live Demo

**[Try the app here →](https://manufacturing-rag-app-6habrndrkxoos39atcuv6d.streamlit.app/)**

Sample questions you can ask:
- *"What are the most common defect causes in the welding process?"*
- *"Which supplier has the highest rejection rate?"*
- *"What maintenance actions were taken on the welding robot?"*
- *"What are the quality standards for body panel tolerances?"*

---

## The Core Problem

Traditional keyword search fails for manufacturing documents:
Engineer searches: "welding defect cause"Keyword search → finds documents containing those exact words

→ misses context, relationships, root causes
→ returns irrelevant resultsRAG system     → understands semantic meaning of the question
→ retrieves most relevant passages across all docs
→ generates precise answer with source citation
→ engineer gets root cause in seconds, not hours

---

## System Architecture

Unstructured Documents
(PDFs — Logs, Maintenance, Quality, Supplier)
↓
Document Ingestion (LangChain Document Loaders)
↓
Text Chunking & Preprocessing
(RecursiveCharacterTextSplitter)
↓
OpenAI Embeddings (text-embedding-ada-002)
↓
FAISS Vector Store (Semantic Index)
↓
User Query → Semantic Search (Top-K Retrieval)
↓
Re-ranking Layer (Contextual Accuracy)
↓
LLM Response Generation (GPT-3.5-turbo)
↓
Answer + Source Attribution
↓
Streamlit Interactive UI


---

## What I Built

**1. Multi-source document ingestion pipeline.**
Built a document ingestion system using LangChain loaders —
handling PDFs across heterogeneous manufacturing data sources
including defect logs, maintenance SOPs, production overviews,
supplier BOMs, and quality reports. Documents are chunked with
overlap to preserve context across boundaries.

**2. Semantic search with FAISS vector store.**
Documents are embedded using OpenAI embeddings and indexed
in a FAISS vector store. Similarity search retrieves the
Top-K most semantically relevant passages for any query —
far more effective than keyword matching for technical
manufacturing language.

**3. Two-stage retrieval pipeline.**
Implemented semantic search for candidate generation followed
by a re-ranking layer to improve contextual accuracy. This
ensures the most relevant context reaches the LLM, reducing
hallucinations on domain-specific manufacturing queries.

**4. Context-grounded responses with source attribution.**
Every LLM response cites the specific source documents used —
enabling engineers to verify answers and trace root causes
back to original maintenance records or quality reports.
Critical for ISO 9001 quality audit trails.

**5. Configurable Streamlit application.**
Interactive app with guided sample questions, configurable
Top-K retrieval, temperature control, and free-form querying
— designed for non-technical manufacturing engineers, not
data scientists.

---

## Key Design Decisions

**Why RAG over fine-tuning?**
Manufacturing documents change frequently — new defect reports,
updated maintenance SOPs, revised supplier quality records.
RAG retrieves from the latest documents without retraining.
Fine-tuning would require expensive retraining every update cycle.

**Why FAISS over hosted vector database?**
For internal manufacturing deployments, FAISS runs locally
with no external API calls — keeping sensitive operational
data on-premise. Architecture can swap to Pinecone or Chroma
with minimal code changes for cloud deployments.

**Why source attribution matters in manufacturing?**
When an engineer acts on a root cause finding, they need to
know *which document* said it — for quality audit trails,
regulatory compliance (ISO 9001), and accountability.
A RAG system without citations is a liability in manufacturing.


---

## RAG Pipeline Details

### Stage 1 — Document Ingestion & Chunking
```python
# Multi-format document loading
loader = PyPDFDirectoryLoader("Data/")
documents = loader.load()

# Chunk with overlap to preserve context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
```

### Stage 2 — Embedding & Vector Store
```python
# OpenAI embeddings → FAISS index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vector_store/faiss_openai")
```

### Stage 3 — Retrieval & Generation
```python
# Semantic search → LLM response with citations
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=True
)
```

---

## Features

- 📄 **Multi-format document ingestion** — PDFs across
  logs, maintenance, quality, supplier categories
- 🔍 **Semantic search** — finds relevant context,
  not just keyword matches
- 🔄 **Two-stage retrieval** — semantic search + re-ranking
- 💬 **Free-form querying** — plain English questions
- 📎 **Source attribution** — every answer cites its
  source document
- ⚙️ **Configurable parameters** — Top-K, temperature
- 🎯 **Guided sample questions** — helps engineers
  get started quickly
- 🏭 **Domain-specific** — tuned for manufacturing
  defect root cause analysis

---

## Manufacturing Document Categories

| Category | Documents | Use Case |
|----------|-----------|----------|
| Defect Logs | defect_log_sample.pdf | Defect history lookup |
| Maintenance | welding_robot_maintenance_sop.pdf | Maintenance decision support |
| Production | assembly_process_overview.pdf | Process investigation |
| Quality | welding_defects_and_rca.pdf | Root cause analysis |
| Supplier | body_panel_bom_and_tolerance.pdf | Supplier quality investigation |
| Web/Context | manufacturing_context_news.pdf | Industry benchmarking |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | OpenAI text-embedding-ada-002 |
| Language Model | GPT-3.5-turbo |
| Re-ranking | Custom reranker pipeline |
| App Framework | Streamlit |
| Deployment | AWS EC2 / Streamlit Cloud |
| Language | Python 3.9+ |

---

## Setup & Run Locally

```bash
# Clone the repository
git clone https://github.com/Vijitha-Kake/decision-intelligence-rag.git
cd decision-intelligence-rag

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
# Create .env file and add:
# OPENAI_API_KEY=your-api-key-here

# Step 1 — Ingest and chunk documents
python src/ingest_and_chunk.py

# Step 2 — Build FAISS vector store
python src/embedding_faiss.py

# Step 3 — Run the app
streamlit run app/app.py
```

---

## Use Cases

Three primary manufacturing investigation scenarios:

**1. Defect Root Cause Analysis**
> *"What caused the surface finish defects in
> the welding process?"*

**2. Maintenance Decision Support**
> *"What maintenance actions were previously taken
> when the welding robot showed similar issues?"*

**3. Supplier Quality Investigation**
> *"What are the tolerance specifications for
> body panel supplier components?"*

---

## Deployment

Deployed on **AWS EC2** for internal manufacturing
team use. Public demo hosted on **Streamlit Cloud**.

For production deployment with proprietary documents,
the FAISS index is built and served on-premise — no
sensitive operational data leaves the manufacturing
network.

---

## Learning Resources & References

**Course**
- CampusX — *Generative AI with LangChain* —
  RAG pipeline architecture and LangChain implementation
  patterns that informed this project's approach

**LangChain Documentation**
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
  — Retrieval chain implementation

**Research Papers**
- Lewis et al. (2020), *Retrieval-Augmented Generation
  for Knowledge-Intensive NLP Tasks*, NeurIPS —
  foundational RAG architecture paper
- [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

**OpenAI**
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
  — text-embedding-ada-002 documentation

---

## Author

**Vijitha Kake** — Data Scientist | ML Engineer | Gen AI

vijitha13k@gmail.com |
[LinkedIn](https://linkedin.com/in/vijitha-kake) |
[GitHub](https://github.com/Vijitha-Kake)


