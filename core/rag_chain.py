# core/rag_chain.py
from core.vectordb import StigVectorDatabase
from core.llm_agent import StigLLMAgent
from core.embeddingAgents import StigEmbeddingAgent





# Python convention
DEFAULT_MODEL = "mistral"



class StigRAGChain:
    
    """
    Orchestrator that coordinates the embedding agent, vector database,
    and LLM agent to answer questions about Nigerian history using RAG.
    """
    
    def __init__(self, model_name=DEFAULT_MODEL, persist_directory="db"):
        # Initialize the Orchestrator with the embedding agent, vector database, and LLM agent
        self.embedding_agent = StigEmbeddingAgent()
        self.vector_db = StigVectorDatabase(self.embedding_agent, persist_directory)
        self.vector_db.load_vectorstore()
        self.llm_agent = StigLLMAgent()

    def answer_question(self, question: str) -> str:
        """Answer a question using the RAG approach"""
        # Step 1: Retrieve relevant context chunks from the vector database
        context_chunks = self.vector_db.similarity_search(question, k=3)

        # Step 2: Use the LLM agent to generate an answer based on the retrieved context
        answer = self.llm_agent.answer(question, context_chunks)
        return answer