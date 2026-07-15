import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class StigEmbeddingAgent:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(model=self.model_name)

    def get_embedding(self, text):
        """Embed a single string — useful for testing"""
        return self.embeddings.embed_query(text)

    def get_embeddings_model(self):
        """Return the embeddings object — used by the vector database"""
        return self.embeddings