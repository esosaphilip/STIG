# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from core.rag_chain import StigRAGChain

# create the app
app = FastAPI()

# create the RAG chain once when the app starts
rag = StigRAGChain()

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


# define the endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    # step 1 - call the right method on the right class
    answer = rag.answer_question(request.question)
    # step 2 - return the answer
    return {"answer": answer}