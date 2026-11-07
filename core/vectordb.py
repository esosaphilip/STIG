from langchain_chroma import Chroma


def get_persist_directory(country: str) -> str:
    """
    Resolve the on-disk directory for a country's vector store.

    Why this rule exists: each country's documents must live in a SEPARATE Chroma
    store. If German and Nigerian chunks shared one collection, a question about
    one country could retrieve the other's context. Nigeria keeps the original
    "db" path so its already-ingested data is never migrated or re-ingested;
    every other country gets a sibling "db_<country>" directory instead.
    """
    country = country.lower()
    return "db" if country == "nigeria" else f"db_{country}"


class StigVectorDatabase:
    def __init__(self, embedding_agent, persist_directory="db"):
        self.persist_directory = persist_directory
        self.embedding_agent = embedding_agent
        self.vectorstore = None

    def create_vectorstore(self, documents):
        """Create vector store from documents and persist to disk"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_agent.get_embeddings_model(),
            persist_directory=self.persist_directory
        )
        print(f"Vector store created with {len(documents)} documents")

    def load_vectorstore(self):
        """Load an existing vector store from disk"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_agent.get_embeddings_model()
        )
        print(f"Vector store loaded from {self.persist_directory}")

    def similarity_search(self, query, k=3):
        """Search for the most relevant documents"""
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore.similarity_search(query, k=k)