import os
import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: FAISS Vector Database Creation Function
def create_vector_db():
    # Load the dataset
    df = pd.read_csv('c:\Users\RAKSHAN\Downloads\Larsen.csv')
    # Combine relevant columns to create a 'text' column
    df['text'] = df[['ProjectName', 'Location', 'StartDate', 'EndDate', 'Branch']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df['text'] = df['text'].str.strip()  # Clean up text data

    # Initialize the SentenceTransformer model for encoding
    encoder = SentenceTransformer("all-mpnet-base-v2")

    # Encode the text column into vectors
    vectors = encoder.encode(df['text'].tolist())  # Convert the text column to a list and encode it

    # Create FAISS index
    dim = vectors.shape[1]  # The dimension of the embeddings
    index = faiss.IndexFlatL2(dim)  # Create a FAISS index
    index.add(vectors)  # Add the vectors to the FAISS index

    # Create vector_index directory if it doesn't exist
    vector_index_path = "vector_index/index.faiss"
    os.makedirs(vector_index_path, exist_ok=True)

    # Save the FAISS index with the project path
    faiss.write_index(index, os.path.join(vector_index_path, "index.faiss"))

    # Save the dataframe so it can be loaded later for retrieval
    df.to_csv(os.path.join(vector_index_path, "index.csv"), index=False)

    print("Vector DB created and saved locally.")

# Step 2: Function to load FAISS and perform retrieval
def get_qa_chain():
    # Load the FAISS index with the project path
    index = faiss.read_index("C:/Users/RAKSHAN/Downloads/MAIN PRJ/vector_index/index.faiss")

    # Load the dataframe containing the text
    df = pd.read_csv("C:/Users/RAKSHAN/Downloads/MAIN PRJ/vector_index/index.csv")

    # Initialize the SentenceTransformer model for query encoding
    encoder = SentenceTransformer("all-mpnet-base-v2")

    # Query handling for local data only
    def retrieve_from_faiss(query, k=2):
        # Encode the query
        search_vector = encoder.encode(query).reshape(1, -1)

        # Perform the search in FAISS
        distances, indices = index.search(search_vector, k)

        # Retrieve the corresponding rows from the dataframe
        row_indices = indices.flatten().tolist()
        results = df.loc[row_indices]

        return results

    return retrieve_from_faiss


# Streamlit App
st.title("Larsen and Toubro Project Information Retrieval")
st.sidebar.header("Chatbot")

# User input
question = st.sidebar.text_input("Ask me anything about the projects:")

if question:
    # Initialize the FAISS-based retrieval
    retrieve_from_faiss = get_qa_chain()

    # Retrieve the results for the user's query
    results = retrieve_from_faiss(question)

    # Display the results
    st.header("Search Results:")
    st.write(results)

