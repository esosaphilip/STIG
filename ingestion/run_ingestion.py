# ingestion/run_ingestion.py
from core.document_loader import DocumentLoader
from core.TextSplitter import StigTextSplitter
from core.embeddingAgents import StigEmbeddingAgent
from core.vectordb import StigVectorDatabase

# a list of Nigerian history topics to load from Wikipedia
NIGERIAN_HISTORY_TOPICS = [
    # add 5 Wikipedia topics here yourself
    # think about what STIG should know about
    "the history of Nigeria", 
    "the Nigerian Civil War", 
    "the transatlantic slave trade in Nigeria", 
    "the colonial period in Nigeria", 
    "the independence movement in Nigeria",
    "Obasanjo's presidency",
    "Sani Abacha's regime",
    "abdulsalem Abubakar's transitional government",
    "the First Republic of Nigeria",
    "the Second Republic of Nigeria",
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
            loader = DocumentLoader("wikipedia", topic)
            documents = loader.load()

            # step 4 - split the documents into chunks
            chunks = splitter.split(documents)

            # step 5 - store the chunks in the vector database
            vector_db.create_vectorstore(chunks)


            # step 6 - print that this topic is done
            print(f"Finished ingesting topic: {topic}")
        except Exception as e:
            print(f"Error ingesting topic {topic}: {e}")
            continue


# this tells Python to run the function when you execute this file directly
if __name__ == "__main__":
    run_ingestion()