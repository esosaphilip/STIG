# api/main.py


# api/main.py
import sys
import os

# Add project root to path so Python can find core/, data/, ingestion/ modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from core.rag_chain import StigRAGChain
from data.cache import StigCache
from core.sourcing_agent import create_sourcing_agent
from core.sourcing_agent import PoliticianState  # Ensure PoliticianState is imported


# create the app
app = FastAPI()

# create the RAG chain once when the app starts
rag = StigRAGChain()

# create the Cache object
cache = StigCache()

# define what the incoming request looks like
class QuestionRequest(BaseModel):
    # what field goes here?
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def greet():
    return {"message": "Welcome to STIG"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        cached_answer = cache.get(request.question)  # retrieved already answered Question?
        if cached_answer:
            return {"answer": cached_answer, "cached": True}  # return cached: True to show it stored successfully.
        answer = rag.answer_question(request.question)  # answer the question
        cache.set(request.question, answer)  # store BEFORE returning incase same Question comes.
        return {"answer": answer, "cached": False}  # cached: False?
    except Exception as e:
        return {"error": str(e)}  # catch all exceptions here
    
    # define what the incoming request looks like
class InfoRequest(BaseModel):
    # what field goes here?
    name: str
    
# define the endpoint
@app.post("/infocard")
def get_infocard(request: InfoRequest):
    # step 1 - create the sourcing agent
    agent = create_sourcing_agent()
    # step 2 - invoke it with the politician name

    politician_state = {
        "name": request.name,
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