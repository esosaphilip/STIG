# main.py
from document_loader import DocumentLoader
from text_splitter import StigTextSplitter
from embedding_agent import StigEmbeddingAgent
from vector_database import StigVectorDatabase

# Step 1 — Load
loader = DocumentLoader(source_type="wikipedia", source_path="Nigerian Civil War")
documents = loader.load()

# Step 2 — Chunk
splitter = StigTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split(documents)

# Step 3 — Embed + Store (these two happen together inside Chroma)
embedding_agent = StigEmbeddingAgent()
db = StigVectorDatabase(embedding_agent=embedding_agent)
db.create_vectorstore(chunks)

# Step 4 — Query
results = db.similarity_search("What caused the Nigerian Civil War?")
for r in results:
    print(r.page_content)
    print("---")