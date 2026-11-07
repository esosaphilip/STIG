# core/llm_agent.py
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Fallback persona used only when no system prompt is supplied, so calling
# StigLLMAgent() directly still behaves exactly as it did before.
DEFAULT_SYSTEM_PROMPT = """You are STIG, an AI assistant specialized in 
        Nigerian history and governance. You answer questions based strictly 
        on the context provided to you. If the context does not contain enough 
        information to answer, say so clearly. Always mention your sources."""

class StigLLMAgent:
    def __init__(self, model_name: str = "mistral", system_prompt: str | None = None):
        self.llm = ChatOllama(
    model=model_name,
    validate_model_on_init=True,
    temperature=0.1,
    num_predict=1250
)
        # Accept the persona as a parameter so each country supplies its own
        # system prompt (see config/<country>.json). Hardcoding it here is what
        # made the model always claim to be a Nigerian-history expert.
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

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