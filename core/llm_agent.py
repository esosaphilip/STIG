# core/llm_agent.py
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class StigLLMAgent:
    def __init__(self, model_name: str = "mistral"):
        self.llm = ChatOllama(
    model=model_name,
    validate_model_on_init=True,
    temperature=0.1,
    num_predict=1250
)
        self.system_prompt = """You are STIG, an AI assistant specialized in 
        Nigerian history and governance. You answer questions based strictly 
        on the context provided to you. If the context does not contain enough 
        information to answer, say so clearly. Always mention your sources."""

    def answer(self, question: str, context_chunks: list[Document]) -> str:
        """Takes a question and retrieved chunks, returns an answer"""

        context = "\n\n---\n\n".join([
            f"Source: {chunk.metadata.get('source', 'Unknown')}\n{chunk.page_content}"
            for chunk in context_chunks
        ])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Context from STIG knowledge base:
{context}

Question: {question}

Answer based only on the context above.
""")
        ]

        response = self.llm.invoke(messages)
        
        # Guard check to satisfy Pylance
        if isinstance(response.content, str):
            return response.content
        return str(response.content)