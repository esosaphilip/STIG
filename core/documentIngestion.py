import requests
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain_core.documents import Document


class Documentloader:
    def __init__(self, source_type, source_path):
        self.source_type = source_type
        self.source_path = source_path

    def load_documents(self):
        if self.source_type == "wikipedia":
            # 1. Endpoint without the trailing slash
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": True,      # Plain text instead of HTML
                "titles": self.source_path,
                "redirects": 1,           # Automatically follow page redirects
            }
            
            # Wikipedia requires a descriptive User-Agent header
            headers = {"User-Agent": "STIG/1.0 (contact@example.com)"}
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            # 2. Extract pages from the nested response
            pages = data.get("query", {}).get("pages", {})
            
            documents = []
            for page_id, page_info in pages.items():
                # -1 indicates the page was not found
                if page_id == "-1":
                    continue
                
                text = page_info.get("extract", "")
                title = page_info.get("title", self.source_path)
                
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                "title": title,
                                "page_id": page_id
                            }
                        )
                    )
            
            return documents

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