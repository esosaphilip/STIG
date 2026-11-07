# api/main.py
import sys
import os
import logging

# Add project root to path so Python can find core/, data/, ingestion/ modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.rag_chain import StigRAGChain
from data.cache import StigCache
from core.sourcing_agent import create_sourcing_agent
from core.sourcing_agent import PoliticianState  # Ensure PoliticianState is imported
from config.settings import STIGConfig, CONFIG_DIR
from fastapi.middleware.cors import CORSMiddleware

# Basic logging so server-side exceptions (e.g. a failed /ask) actually land in
# the logs even though the client only receives a generic message.
logging.basicConfig(level=logging.INFO)

# create the app
app = FastAPI()


# Add this AFTER app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins including file://
    allow_methods=["*"],
    allow_headers=["*"],
)

# One RAG chain per country, built lazily and cached for the process lifetime.
# Building a chain loads a Chroma vector store from disk, which is expensive and
# must not happen on every request — but we also can't build every country's
# chain at import time, since that would load all stores on startup.
_rag_chains: dict[str, StigRAGChain] = {}


def get_rag_chain(country: str) -> StigRAGChain:
    """Return the cached chain for a country, building it on first use."""
    key = country.lower()
    if key not in _rag_chains:
        _rag_chains[key] = StigRAGChain(country=key)
    return _rag_chains[key]


# create the Cache object
cache = StigCache()

# define what the incoming request looks like
class QuestionRequest(BaseModel):
    question: str
    # Default to Nigeria so existing clients that don't send a country keep working.
    country: str = "nigeria"


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def greet():
    return {"message": "Welcome to STIG"}

@app.get("/countries")
def countries():
    # Expose the country list straight from the config files so the frontend can
    # build its picker without hardcoding which countries we support.
    return {"countries": STIGConfig(CONFIG_DIR).available_countries()}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        cached_answer = cache.get(request.question)  # retrieved already answered Question?
        if cached_answer:
            return {"answer": cached_answer, "cached": True}  # return cached: True to show it stored successfully.
        answer = get_rag_chain(request.country).answer_question(request.question)  # answer the question
        cache.set(request.question, answer)  # store BEFORE returning incase same Question comes.
        return {"answer": answer, "cached": False}  # cached: False?
    except Exception:
        # Log the real error server-side, but return a clean 500 to the client.
        # Callers need a status code they can branch on, and we must not leak raw
        # exception text. Previously this returned 200 with the exception string
        # in the body, which hid failures behind a success status.
        logging.exception("Failed to answer question")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong answering this question.",
        )

    # define what the incoming request looks like
class InfoRequest(BaseModel):
    name: str
    # Optional: a name is usually enough to find the person on Wikipedia, so the
    # country is a disambiguation hint rather than a requirement.
    country: str = ""

# define the endpoint
@app.post("/infocard")
def get_infocard(request: InfoRequest):
    # step 1 - create the sourcing agent
    agent = create_sourcing_agent()
    # step 2 - invoke it with the politician name

    politician_state = {
        "name": request.name,
        "country": request.country,
        "date_of_birth": "",
        "date_of_death": "",
        "state_of_origin": "",
        "party": "",
        "offices_held": [],
        "tenure_dates": [],
        "scandals": [],
        "court_cases": [],
        "associations": [],
        "allies": [],
        "enemies": [],
        "business_partners": [],
        "notable_statements": [],
        "social_media": {},
        "old_politicians_newspaper_appearances": [],
        "any_other_interesting_facts": [],
        "news_mentions": [],
        "raw_wikipedia_text": "",
        "raw_news_text": "",
        "sources": []
    }
    result = agent.invoke(politician_state)

    # step 3 - return the result
    return result
