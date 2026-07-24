# ingestion/run_ingestion.py
from core.documentIngestion import Documentloader
from core.TextSplitter import StigTextSplitter
from core.embeddingAgents import StigEmbeddingAgent
from core.vectordb import StigVectorDatabase
import traceback


# a list of Nigerian history topics to load from Wikipedia
NIGERIAN_HISTORY_TOPICS = [
    # add 5 Wikipedia topics here yourself
    # think about what STIG should know about
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
]

def run_ingestion():
    # step 1 - create your splitter, embedding agent and vector database
    splitter = StigTextSplitter(chunk_size=500, chunk_overlap=50)
    embedding_agent = StigEmbeddingAgent()
    vector_db = StigVectorDatabase(embedding_agent, persist_directory="db")

    # step 2 - loop through each topic in NIGERIAN_HISTORY_TOPICS
    for topic in NIGERIAN_HISTORY_TOPICS:
        try:
            print(f"Starting ingestion for topic: {topic}")
            # step 3 - load the documents for this topic
            loader = Documentloader("wikipedia", topic)
            documents = loader.load_documents()

            # step 4 - split the documents into chunks
            chunks = splitter.split(documents)

            # step 5 - store the chunks in the vector database
            vector_db.create_vectorstore(chunks)


            # step 6 - print that this topic is done
            print(f"Finished ingesting topic: {topic}")
        except Exception as e:
            print(f"Error ingesting topic {topic}: {e}")
            traceback.print_exc()
            continue


# this tells Python to run the function when you execute this file directly
if __name__ == "__main__":
    run_ingestion()