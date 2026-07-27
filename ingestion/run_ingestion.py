# ingestion/run_ingestion.py
from core.documentIngestion import Documentloader
from core.TextSplitter import StigTextSplitter
from core.embeddingAgents import StigEmbeddingAgent
from core.vectordb import StigVectorDatabase
import traceback
import time

# a list of Nigerian history topics to load from Wikipedia
NIGERIAN_HISTORY_TOPICS = [
    "history of Nigeria", 
    "Nigerian Civil War", 
    "Slavery in Nigeria", 
    "colonial Nigeria", 
    "Independence Day (Nigeria)",
    "Olusegun Obasanjo",
    "Sani Abacha",
    "Abdulsalami Abubakar",
    "First Nigerian Republic",
    "Second Nigerian Republic",
    # new additions
    "Omoyele Sowore",
    "Atiku Abubakar",
    "Muhammadu Buhari",
    "Goodluck Jonathan",
    "Bola Tinubu",
    "Ngozi Okonjo-Iweala",
    "Wole Soyinka",
    "Ken Saro-Wiwa",
    "Moshood Abiola",
    "Yakubu Gowon",
]

def run_ingestion():
    splitter = StigTextSplitter(chunk_size=500, chunk_overlap=50)
    embedding_agent = StigEmbeddingAgent()
    vector_db = StigVectorDatabase(embedding_agent, persist_directory="db")

    for topic in NIGERIAN_HISTORY_TOPICS:
        try:
            print(f"Starting ingestion for topic: {topic}")
            loader = Documentloader("wikipedia", topic)
            documents = loader.load_documents()
            chunks = splitter.split(documents)
            vector_db.create_vectorstore(chunks)
            print(f"Finished ingesting topic: {topic}")

            # wait 3 seconds between requests to avoid rate limiting
            time.sleep(10)

        except Exception as e:
            print(f"Error ingesting topic {topic}: {e}")
            # wait longer if rate limited
            time.sleep(30)
            continue

# this tells Python to run the function when you execute this file directly
if __name__ == "__main__":
    run_ingestion()