# core/llm_agent.py
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class StigLLMAgent:
    def __init__(self, model_name="claude-3-5-sonnet-20241022"):
        self.llm = ChatAnthropic(model=model_name)
        self.system_prompt = """You are STIG, an AI assistant specialized in 
        Nigerian history and governance. You answer questions based strictly 
        on the context provided to you. If the context does not contain enough 
        information to answer, say so clearly. Always mention your sources."""

    def answer(self, question: str, context_chunks: list) -> str:
        """Takes a question and retrieved chunks, returns an answer"""

        # Format the chunks into readable context
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
        return response.content