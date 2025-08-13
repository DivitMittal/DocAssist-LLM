from langchain_community.embeddings import HuggingFaceBgeEmbeddings

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vector_db"

EMBEDDING_MODEL = "BAAI/bge-large-en"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEARCH_K = 5

PDF_FILES = [
    "./dataset/impatient-js-preview-book.pdf",
    "./dataset/react-js-book.pdf",
]


def get_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
