# STIG — Sovereign Truth & Intelligence Gateway

> **Nigeria's Truth. Declassified.**

![STIG Demo](https://raw.githubusercontent.com/esosaphilip/STIG/master/assets/StigGif%202.gif)

---

## What is STIG?

STIG is an AI-powered Nigerian political intelligence platform — a digital alternative to Nigeria's neglected National Archive. It aggregates historical records, politician profiles, and sourced accounts of key events, making them queryable in plain language.

Unlike a search engine, STIG **makes the connections** — retrieving relevant sources from multiple documents and synthesising them into one clear, cited answer.

---

## Screenshots

| Home | Archive | Dossier |
|------|---------|---------|
| ![Home](https://raw.githubusercontent.com/esosaphilip/STIG/master/assets/stighomescreen.png) | ![Archive](https://raw.githubusercontent.com/esosaphilip/STIG/master/assets/archivescreen.png) | ![Dossier](https://raw.githubusercontent.com/esosaphilip/STIG/master/assets/dossierscreen_1.png) |

![Dossier Detail](https://raw.githubusercontent.com/esosaphilip/STIG/master/assets/dossierscreen_2.png)

---

## Features

**ARCHIVE MODE** — Query Nigerian historical records in natural language
- Ask anything about colonial Nigeria, military rule, the civil war, or the republic
- Answers are sourced, cited, and returned as an intelligence briefing
- Redis caching for instant repeated queries

**DOSSIER MODE** — Retrieve a full intelligence profile on any Nigerian politician
- Powered by a LangGraph multi-node sourcing agent
- Searches Wikipedia and live web sources via Tavily
- Extracts structured fields: offices held, party, scandals, court cases, allies, enemies, sources

---

## Architecture

```
User Query
    ↓
FastAPI REST API  (/ask or /infocard)
    ↓
Redis Cache Check  →  Cache HIT: return instantly
    ↓
Cache MISS
    ↓
┌─────────────────────────────────────┐
│  ARCHIVE: StigRAGChain              │
│  ├── StigEmbeddingAgent             │
│  │   └── nomic-embed-text (Ollama)  │
│  ├── StigVectorDatabase (ChromaDB)  │
│  │   └── similarity_search (k=5)   │
│  └── StigLLMAgent                  │
│      └── Mistral 7B (Ollama)       │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  DOSSIER: LangGraph Sourcing Agent  │
│  ├── Node 1: Wikipedia Search       │
│  ├── Node 2: Tavily Web Search      │
│  └── Node 3: LLM Profile Extraction │
└─────────────────────────────────────┘
    ↓
Store in Redis Cache
    ↓
Return to User
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Orchestration | LangChain + LangGraph |
| LLM | Mistral 7B via Ollama (fully local, no API cost) |
| Embeddings | nomic-embed-text via Ollama (fully local) |
| Vector Database | ChromaDB |
| Caching | Redis |
| Web Search | Tavily API |
| API Framework | FastAPI + Pydantic |
| Language | Python 3.11 |
| Frontend | HTML + CSS + JavaScript (single file) |
| Data Sources | Wikipedia MediaWiki API, Tavily web search |

> **Fully local LLM** — STIG runs Mistral 7B and embeddings on-device via Ollama. No OpenAI API key required.

---

## Project Structure

```
STIG/
├── api/
│   └── main.py                 # FastAPI endpoints: /ask, /infocard, /health
├── core/
│   ├── documentIngestion.py    # Wikipedia MediaWiki API loader
│   ├── TextSplitter.py         # RecursiveCharacterTextSplitter
│   ├── embeddingAgents.py      # OllamaEmbeddings (nomic-embed-text)
│   ├── vectordb.py             # ChromaDB interface
│   ├── llm_agent.py            # ChatOllama (Mistral 7B)
│   ├── rag_chain.py            # RAG orchestrator
│   ├── sourcing_agent.py       # LangGraph politician research agent
│   └── cache.py                # Redis caching layer
├── data/
│   └── database.py             # PostgreSQL (future)
├── ingestion/
│   └── run_ingestion.py        # Wikipedia ingestion pipeline
├── ui/
│   └── stig_ui.html            # Intelligence database frontend
├── assets/                     # Screenshots and demo GIF
└── tests/
    └── test_pipeline.py
```

---

## How to Run Locally

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed
- [Docker](https://docker.com) installed

### 1. Clone the repository
```bash
git clone https://github.com/esosaphilip/STIG.git
cd STIG
```

### 2. Create and activate virtual environment
```bash
python -m venv stig-env
source stig-env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirement.txt
```

### 4. Pull Ollama models
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 5. Start Redis
```bash
docker run -d -p 6379:6379 --name stig-redis redis:alpine
```

### 6. Create `.env` file
```
USER_AGENT=STIG/1.0
TAVILY_API_KEY=your_tavily_key_here
```

### 7. Run ingestion
```bash
python -m ingestion.run_ingestion
```

### 8. Start API server
```bash
uvicorn api.main:app --reload
```

### 9. Serve the UI
```bash
python -m http.server 3000
```

Open `http://localhost:3000/ui/stig_ui.html`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Server health check |
| `POST` | `/ask` | Query Nigerian historical records |
| `POST` | `/infocard` | Generate politician intelligence profile |

**Example — ARCHIVE:**
```json
POST /ask
{"question": "Who was Sani Abacha and what did he do to Nigeria?"}

→ {"answer": "Sani Abacha was a military head of state...", "cached": false}
```

**Example — DOSSIER:**
```json
POST /infocard
{"name": "Olusegun Obasanjo"}

→ {"name": "Olusegun Obasanjo", "party": "PDP", "offices_held": [...], "scandals": [...]}
```

---

## Knowledge Base

Currently ingested topics:

**History:** History of Nigeria, Nigerian Civil War, Slavery in Nigeria, Colonial Nigeria, Independence Day, First & Second Republic

**Politicians:** Olusegun Obasanjo, Sani Abacha, Abdulsalami Abubakar, Atiku Abubakar, Muhammadu Buhari, Goodluck Jonathan, Bola Tinubu, Ngozi Okonjo-Iweala, Wole Soyinka, Ken Saro-Wiwa, Moshood Abiola, Yakubu Gowon

---

## Roadmap

- [ ] Politician connection graph (who appointed who, allies network)
- [ ] Account conflict engine — side-by-side accounts of disputed events
- [ ] EFCC records integration
- [ ] State intelligence dashboards (all 36 states)
- [ ] IngestionTracker to prevent duplicate ingestion
- [ ] React frontend (post-WBS bootcamp)
- [ ] Production deployment

---

## Built By

**Eseosa Edosomwan** — AI Engineering Portfolio Project

*Demonstrating: RAG architecture · LangGraph agents · Local LLM deployment · FastAPI · ChromaDB · Redis caching · Python OOP · Production-grade project structure*

[![GitHub](https://img.shields.io/badge/GitHub-esosaphilip-blue?style=flat&logo=github)](https://github.com/esosaphilip)
