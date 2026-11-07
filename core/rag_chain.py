# core/rag_chain.py
from core.vectordb import StigVectorDatabase, get_persist_directory
from core.llm_agent import StigLLMAgent
from core.embeddingAgents import StigEmbeddingAgent
from config.settings import STIGConfig, CONFIG_DIR


# Python convention
DEFAULT_MODEL = "mistral"


class StigRAGChain:

    """
    Orchestrator that coordinates the embedding agent, vector database,
    and LLM agent to answer questions about a given country's history using RAG.
    """

    def __init__(self, model_name=DEFAULT_MODEL, country: str = "nigeria"):
        # Bind this chain to a single country: it gets that country's own vector
        # store (chosen from the country name, not a caller-supplied path, so a
        # country can't be pointed at the wrong directory) and that country's
        # system prompt from its config file. One chain == one country.
        self.country = country.lower()
        config = STIGConfig(CONFIG_DIR).load_country(self.country)

        # Initialize the Orchestrator with the embedding agent, vector database, and LLM agent
        self.embedding_agent = StigEmbeddingAgent()
        self.vector_db = StigVectorDatabase(
            self.embedding_agent, get_persist_directory(self.country)
        )
        self.vector_db.load_vectorstore()
        self.llm_agent = StigLLMAgent(
            model_name=model_name, system_prompt=config["system_prompt"]
        )

    def answer_question(self, question: str) -> str:
        """Answer a question using the RAG approach"""
        # Step 1: Retrieve relevant context chunks from the vector database
        context_chunks = self.vector_db.similarity_search(question, k=5)

        # Step 2: Use the LLM agent to generate an answer based on the retrieved context
        answer = self.llm_agent.answer(question, context_chunks)
        return answer
