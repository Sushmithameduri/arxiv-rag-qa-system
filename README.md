# ArXiv RAG Question Answering System

**Production-ready RAG pipeline for OpenRAGBench ArXiv dataset.** FastAPI backend with Chroma vector DB, local Ollama LLM, and Streamlit UI. Supports retrieval eval and full Docker deployment.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-blue?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-orange?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-27-blue?logo=docker)](https://docker.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=langchain)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-yellow?logo=ollama)](https://ollama.com)


## 🚀 Key Features

- **End-to-end RAG pipeline**
  - PDF ingestion → chunking → embeddings → vector search → answer generation
- **Grounded answers with citations**
  - Answers are generated strictly from retrieved context
  - No hallucinations. Explicit “I don’t know” when evidence is missing
- **Local LLM inference**
  - Uses **Ollama** with `llama3.2:3b` (no external API dependency)
- **Production-style API**
  - FastAPI service with clear request/response schema
- **Interactive UI**
  - Streamlit interface for querying and inspecting retrieved context
- **Dockerized**
  - Fully containerized API + UI for reproducible setup
- **Config-driven**
  - All runtime configuration via `.env`

### 📽️ Demo Video

This demo shows the Streamlit UI in action for the ArXiv RAG QA system.

- Ask a natural-language research question

- Retrieve Top-K relevant chunks from the arXiv corpus

- Generate a grounded, citation-aware answer using a local LLM (Ollama)

- Display retrieved context alongside the answer for transparency

- The demo highlights semantic retrieval, hallucination-safe generation, and evidence-bounded RAG behavior in a fully local, reproducible setup.

🎥 Demo video: ![Adobe+Express+-+Arxvi+Rag+Demo+(1)](https://github.com/user-attachments/assets/192a2451-1047-4648-844b-983e933ea77d)

---

## 🧠 Architecture Overview
```
User Question
↓
Streamlit UI
↓
FastAPI (/query)
↓
ChromaDB Vector Search
↓
Top-K Relevant Chunks
↓
Ollama LLM (Grounded Prompt)
↓
Answer + Citations

```

---

## 📂 Project Structure

```
rag_arxiv_api/
├── app/                    # Core application (FastAPI, RAG, ingestion)
│   ├── main.py            # API entrypoint - FastAPI application setup
│   ├── rag.py             # RAGService - retrieval + LLM generation
│   ├── ingest.py          # Dataset ingestion → Chroma vector store
│   ├── config.py          # Pydantic Settings - configuration management
│   └── schemas.py         # Pydantic models - request/response schemas
├── ui/                    # Streamlit frontend - user interface
├── data/                  # Raw OpenRAGBench dataset (~1.5GB)
├── db/                    # Chroma persistence - vector database storage
├── eval/                  # Retrieval metrics - evaluation scripts
├── Dockerfile             # Container image definition
├── docker-compose.yml     # Multi-container orchestration
└── requirements.txt       # Python dependencies
```

## 🎯 Quick Start

This section explains how to set up, ingest data, and run the RAG pipeline locally.

### 1️⃣ Prerequisites

Make sure you have the following installed:

* Python 3.10+

* Docker & Docker Compose

* Git

* Ollama (for local LLM inference)

Install Ollama

👉 [Download Ollama](https://ollama.com/download)

Then pull the required model:

```bash
ollama pull llama3.2:3b
```

Verify Ollama is running:
```bash
ollama serve
```

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/Sushmithameduri/arxiv-rag-qa.git
cd arxiv-rag-qa
```

### 3️⃣ Download Dataset (Hugging Face)

This project uses the OpenRAGBench ArXiv corpus, hosted on Hugging Face.

### Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

This installs `huggingface_hub` and all other required libraries.

####  Download OpenRAGBench Dataset (Using Repo Code)

Run the provided downloader script. Downloader implementation (already in this repo):

Download ~1.5GB ArXiv dataset (1000s of papers)

```bash
python app/download_raw-dataset.py
```
This script downloads the *OpenRAGBench arXiv* dataset directly from Hugging Face and stores it locally.


What this does:

* Downloads the dataset from Hugging Face

* Stores it under data/open_ragbench_raw/

* Preserves the directory structure required by the ingestion pipeline

Expected structure:

```
data/
└── open_ragbench_raw/
    └── pdf/
        └── arxiv/
            ├── corpus/
            ├── queries.json
            ├── answers.json
            ├── qrels.json
            └── pdf_urls.json
```

### 4️⃣ Ingest Data into Vector DB 

Quick: first 100 papers (~10k chunks)

Run ingestion (locally or inside Docker):
```bash
python -m app.ingest
```


This will:

* Parse arXiv documents

* Chunk text

* Generate embeddings

* Store vectors in ChromaDB (db/)

Example Output:

```bash
Ingestion complete.
Papers processed: 100
Sections loaded: 1810
Chunks indexed: 15540
Chroma dir: db/chroma_open_ragbench
```

⚙️ Controlling Ingestion Size (Optional)


For faster local runs, ingestion can be limited via .env:

```bash
DEFAULT_DOC_LIMIT=100
```

This allows:

  * Quick experimentation on laptops
  
  * Full-scale ingestion later by increasing the limit.

Full dataset
```bash
python app/ingest.py       # No limit arg = all papers
```


🔁 When to Re-Run Download or Ingestion

Re-run download if:

* You delete the data/ directory

Re-run ingestion if:

```bash
rm -rf db/chroma_open_ragbench
python app/ingest.py
```

* You change chunk size or overlap

* You change the embedding model

* You increase DEFAULT_DOC_LIMIT

* You delete the db/ directory

* Otherwise, the existing vector store is reused.


### 5️⃣ Run the RAG API (Docker)

1. Start Ollama (host machine)
```bash
ollama serve
ollama pull llama3.2:3b
```
2 . Build and run services
```bash
docker compose up --build
```
3. Access services

Starting the Docker  serices launches:

* **FastAPI:** [http://localhost:8000](http://localhost:8000)


### 6️⃣ Verify the API Health


* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

Query the RAG system:
```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{"question":"What problem does the Virtuoso paper address?", "top_k": 4}'
```

### 7️⃣ Open the Streamlit UI

Open your browser: [http://localhost:8501](http://localhost:8501)


You can:

* Ask research questions

* Control Top-K retrieval

* Inspect retrieved context

* Verify citations and evidence

### 🧪 Development Mode (Without Docker)

If you prefer running locally:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start API:

```bash
uvicorn app.main:app --reload
```

Start UI:
```bash
streamlit run ui/streamlit_app.py

```


📈 Why This Project Matters

This project reflects real-world GenAI engineering practices:

* Separation of retrieval and generation

* Controlled prompts with explicit uncertainty handling

* Local LLM deployment

* Production-style API + UI

* Dockerized for portability




