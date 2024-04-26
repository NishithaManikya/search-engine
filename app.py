import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="my_chromadb")
sentence_transformer = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="distilbert-base-nli-mean-tokens")

collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=sentence_transformer, metadata={"hnsw:space": "cosine"})

def main():
    st.title("SEARCH ENGINE")

    # getting the user input
    user_query = st.text_input("Enter your query :")

    if st.button("Search"):
        if user_query:
            # query the collection
            results = collection.query(
                query_texts=[user_query],
                n_results=10,
                include=['documents', 'distances', 'metadatas']
            )

            # display user input
            st.write(f"Your search query: {user_query}")

            # display output documents
            st.write("Search Results:")
            for i, document in enumerate(results['documents'][0], 1):
                st.write(f"{i}. {document}")

if __name__ == "__main__":
    main()