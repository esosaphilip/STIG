# STIG — Sovereign Truth & Intelligence Gateway

> *Nigeria's Digital National Archive, powered by AI*

---

## What is STIG?

STIG is a digital national archive for Nigerian history, built as an AI-powered alternative to Nigeria's neglected physical National Archive. It holds different accounts and sources available on any topic in Nigerian history — from the colonial era to the military regimes and the current republic.

Unlike Google, STIG makes the connections. Rather than returning a list of links, STIG reads multiple historical sources and synthesizes them into one clear, sourced answer in plain language.

---

## The Problem

Nigeria's National Archive is not properly maintained. Historical records, government documents, and political accounts are scattered, inaccessible, or simply lost. Most Nigerians have no single reliable place to research their own history.

STIG exists as a digital alternative — a living archive that any Nigerian can query in natural language and receive a sourced, intelligent answer. As the system grows, it will hold conflicting accounts of the same events side by side, letting citizens read the evidence and form their own judgment.

---

## How It Works

STIG is built on the **Naive RAG (Retrieval-Augmented Generation)** pattern — one of the most important architectural patterns in modern AI engineering.

**Ingestion (runs once):**
1. Documents are ingested from Wikipedia and other sources
2. Each document is split into small chunks of text
3. Each chunk is converted into a vector (a list of numbers representing its meaning) using a local embedding model
4. All vectors are stored in ChromaDB, a local vector database

**Query (runs every time a user asks a question):**
1. The user sends a question to the API
2. `StigRAGChain` — the orchestrator — receives the question
3. The question is converted into a vector using the same embedding model
4. ChromaDB searches for the chunks whose vectors are most similar to the question vector
5. The most relevant chunks are retrieved as plain text
6. The question and the retrieved chunks are passed together to the local LLM (Mistral 7B)
7. Mistral reads the context and generates a sourced answer
8. The answer is returned to the user via the API

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Orchestration | LangChain |
| LLM | Mistral 7B (via Ollama, runs locally) |
| Embeddings | nomic-embed-text (via Ollama, runs locally) |
| Vector Database | ChromaDB |
| API Framework | FastAPI + Pydantic |
| Language | Python 3.11 |
| Data Sources | Wikipedia REST API |
| IDE | VS Code |

> **Fully local** — no OpenAI API key required. STIG runs entirely on your machine using Ollama.

---

## Project Structure

```
STIG/
├── api/
│   └── main.py              # FastAPI endpoints
├── core/
│   ├── documentIngestion.py # Document loading
│   ├── TextSplitter.py      # Chunking
│   ├── embeddingAgents.py   # Embedding model
│   ├── vectordb.py          # ChromaDB interface
│   ├── llm_agent.py         # Mistral LLM agent
│   └── rag_chain.py         # RAG orchestrator
├── ingestion/
│   └── run_ingestion.py     # Ingestion pipeline
├── .env                     # API keys (not committed)
└── requirements.txt
```

---

## How to Run Locally

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed on your machine

### 1. Clone the repository
```bash
git clone https://github.com/esosaphilip/STIG.git
cd STIG
```

### 2. Create and activate a virtual environment
```bash
python -m venv stig-env
source stig-env/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the required Ollama models
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 5. Create your `.env` file
```bash
touch .env
```
Add the following:
```
USER_AGENT=STIG/1.0
```

### 6. Run the ingestion pipeline
```bash
python -m ingestion.run_ingestion
```
This loads Nigerian history content into your local vector database.

### 7. Start the API server
```bash
uvicorn api.main:app --reload
```

### 8. Test the API
Open your browser and go to:
```
http://localhost:8000/docs
```
Use the `/ask` endpoint to query STIG with any Nigerian history question.

---

## Example Query

**Request:**
```json
{
  "question": "Who started the Nigerian Civil War?"
}
```

**Response:**
```json
{
  "answer": "The conflict emerged from political, ethnic, cultural, and religious tensions... [sourced answer with citations]"
}
```

---

## Roadmap

- [ ] Add more Nigerian history sources (Punch archives, government documents, academic papers)
- [ ] Politician intelligence database with EFCC records
- [ ] Account conflict engine — show multiple sources on the same event side by side
- [ ] State intelligence dashboards for all 36 states
- [ ] Frontend chat interface

---

## Built By

**Eseosa Edosomwan** — AI Engineering Portfolio Project

*Demonstrating: RAG architecture, LangChain, local LLM deployment, FastAPI, vector databases, and production-grade Python project structure.*