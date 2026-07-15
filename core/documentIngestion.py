from langchain_community.document_loaders import (
    WikipediaLoader,
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader
)

class Documentloader:
    def __init__(self, source_type, source_path):
        self.source_type = source_type
        self.source_path = source_path

    def load_documents(self):
        if self.source_type == "wikipedia":
            loader = WikipediaLoader(self.source_path)
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
