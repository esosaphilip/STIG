from langchain_text_splitters import RecursiveCharacterTextSplitter


class StigTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,        # max characters per chunk
            chunk_overlap=chunk_overlap,  # overlap between chunks so context is not lost
            separators=["\n\n", "\n", " ", ""]  # how to split — paragraphs first
        )

    def split(self, documents):
        """Takes raw documents, returns smaller chunks"""
        chunks = self.splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks