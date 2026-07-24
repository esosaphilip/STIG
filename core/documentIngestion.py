from langchain_community.document_loaders import (
    WikipediaLoader,
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader
)
import requests
from langchain_core.documents import Document


class Documentloader:
    def __init__(self, source_type, source_path):
        self.source_type = source_type
        self.source_path = source_path

    def load_documents(self):
        # inside load_documents(), replace the wikipedia case:
        if self.source_type == "wikipedia":
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + self.source_path.replace(" ", "_")
            headers = {"User-Agent": "STIG/1.0"}
            response = requests.get(url, headers=headers)
            data = response.json()
            text = data.get("extract", "")
            return [Document(
                page_content=text,
                metadata={"source": url, "title": self.source_path}
            )]
        elif self.source_type == "web":
            loader = WebBaseLoader(self.source_path)
        elif self.source_type == "pdf":
            loader = PyPDFLoader(self.source_path)
        elif self.source_type == "text":
            loader = TextLoader(self.source_path)
        elif self.source_type == "csv":
            loader = CSVLoader(self.source_path)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")

        return loader.load()
