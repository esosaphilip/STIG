# ingestion/run_ingestion.py
import argparse
import time

from config.settings import STIGConfig, CONFIG_DIR
from core.documentIngestion import Documentloader
from core.TextSplitter import StigTextSplitter
from core.embeddingAgents import StigEmbeddingAgent
from core.vectordb import StigVectorDatabase, get_persist_directory


def run_ingestion(country: str):
    """
    Ingest one country's Wikipedia topics into that country's own vector store.

    Why this is parameterised by country: the topic list used to be a hardcoded
    NIGERIAN_HISTORY_TOPICS constant, which made every other country impossible.
    Now the topics come from the country's config file and the destination store
    is chosen per country, so Nigerian data is never re-ingested and never mixed
    with another country's chunks.
    """
    config = STIGConfig(CONFIG_DIR).load_country(country)
    topics = config["topics"]
    persist_directory = get_persist_directory(country)

    splitter = StigTextSplitter(chunk_size=500, chunk_overlap=50)
    embedding_agent = StigEmbeddingAgent()
    vector_db = StigVectorDatabase(embedding_agent, persist_directory=persist_directory)

    print(f"Ingesting {len(topics)} topics for '{country}' into '{persist_directory}'")

    for topic in topics:
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
    # CLI so ingestion can target a country, e.g.
    #   python -m ingestion.run_ingestion nigeria
    #   python -m ingestion.run_ingestion germany
    parser = argparse.ArgumentParser(
        description="Ingest a country's history into its own vector store."
    )
    parser.add_argument("country", help="Country matching a config file, e.g. nigeria")
    args = parser.parse_args()
    run_ingestion(args.country)
