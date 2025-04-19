import streamlit as st
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient

primary_color = "#FF0000"
background_color = "#FFFFFF"
text_color = "#000000"

st.set_page_config(
    page_title="DocAssist Chat",
    page_icon=":speech_balloon:",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    f"""
    <style>
        .reportview-container .main .block-container {{
            max-width: 800px;
            padding: 2rem;
        }}
        .reportview-container .main {{
            color: {text_color};
            background-color: {background_color};
        }}
        .css-1v3fvcr {{
            background-color: {primary_color} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")


# def reload_chat(u, a, e):
#     # Unique keys for each widget
#     user_query_key = u + 1
#     ask_button_key = a + 1
#     end_button_key = e + 1
#
#     user_query = st.text_input("You:", key=user_query_key)
#
#     if st.button("Ask", key=ask_button_key):
#         if user_query:
#             docs = db.similarity_search_with_score(query=user_query, k=5)
#             if not docs:
#                 st.write("Sorry, I couldn't find any relevant information.")
#             else:
#                 st.write("Here are some results:")
#                 count = 0
#                 for i, (doc, score) in enumerate(docs, 1):
#                     if count == 2:
#                         break
#                     if count == 1:
#                         st.write(f"Also, I know that: ")
#                     st.write(f"**Result {i}:**")
#                     st.write(f"Content: {doc.page_content}")
#                     st.write(f"Source: {doc.metadata}")
#                     st.write("---")
#                     count+=1


def main():
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("./assets/logo.png", width=150)

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.title("DocAssist")

    st.write(
        "Welcome to DocAssist Chat! I'm proficient in Javascipt & React.js Framework and I'll do my best to find relevant information for you."
    )

    user_query_key = 0
    ask_button_key = 100
    end_button_key = 200

    user_query = st.text_input("You:", key=user_query_key)

    if st.button("Ask", key=ask_button_key):
        if user_query:
            st.write("DocAssist is generating a response...")
            docs = db.similarity_search_with_score(query=user_query, k=5)
            if not docs:
                st.write("Sorry, I couldn't find any relevant information.")
            else:
                st.write("Here are some results:")
                for i, (doc, score) in enumerate(docs, 1):
                    st.write(doc.page_content)
                    st.write(f"\n")
                    st.write(f"Source: {doc.metadata}")
                    st.write("---")
                    break

    # if st.button("New Chat", key=end_button_key):
    #     reload_chat(user_query_key, ask_button_key, end_button_key)


if __name__ == "__main__":
    main()
