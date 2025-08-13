from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_FILES,
    QDRANT_URL,
    COLLECTION_NAME,
    get_embeddings,
)

docs = []
for path in PDF_FILES:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
texts = splitter.split_documents(docs)

qdrant = Qdrant.from_documents(
    texts,
    get_embeddings(),
    url=QDRANT_URL,
    prefer_grpc=False,
    collection_name=COLLECTION_NAME,
)

print("Vector DB created")
