import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the dataset
df = pd.read_csv("c:/Users/RAKSHAN/Downloads/Larsen.csv")

# Combine relevant columns to create a 'text' column
df['text'] = df[['ProjectName', 'Location', 'StartDate', 'EndDate', 'Branch']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df['text'] = df['text'].str.strip()  # Clean up text data

# Initialize the SentenceTransformer model
encoder = SentenceTransformer("all-mpnet-base-v2")

# Encode the text column into vectors
vectors = encoder.encode(df['text'].tolist())  # Convert the text column to a list and encode it

# Print the shape of the resulting vectors
print("Shape of vectors:", vectors.shape)

# Step 2: Indexing the vectors using FAISS
dim = vectors.shape[1]  # The dimension of the embeddings
index = faiss.IndexFlatL2(dim)  # Create a FAISS index
index.add(vectors)  # Add the vectors to the FAISS index

# Step 3: Create a search query and encode it
search_query = "River bridge"
search_vector = encoder.encode(search_query).reshape(1, -1)

# Normalize the vector for better search results
faiss.normalize_L2(search_vector)

# Step 4: Perform the search
distances, indices = index.search(search_vector, k=2)  # Search for the top 2 closest results

# Step 5: Retrieve the results
row_indices = indices.flatten().tolist()
results = df.loc[row_indices]

# Output the search results
print("Search Results:\n", results)
