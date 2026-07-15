from langchain_chroma import Chroma

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