import streamlit as st
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

from config import QDRANT_URL, COLLECTION_NAME, SEARCH_K, get_embeddings

st.set_page_config(
    page_title="DocAssist Chat",
    page_icon=":speech_balloon:",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
        .reportview-container .main .block-container {
            max-width: 800px;
            padding: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db():
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
    return Qdrant(client=client, embeddings=get_embeddings(), collection_name=COLLECTION_NAME)


_, center, _ = st.columns(3)
with center:
    st.image("./assets/logo.png", width=150)

_, center, _ = st.columns(3)
with center:
    st.title("DocAssist")

st.write(
    "Welcome to DocAssist Chat! I'm proficient in JavaScript & React.js Framework "
    "and I'll do my best to find relevant information for you."
)

db = get_db()
query = st.text_input("You:")

if st.button("Ask") and query:
    st.write("Searching...")
    results = db.similarity_search_with_score(query=query, k=SEARCH_K)

    if not results:
        st.write("No results found.")
    else:
        doc, _ = results[0]
        st.write(doc.page_content)
        st.write(f"Source: {doc.metadata}")
