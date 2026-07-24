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