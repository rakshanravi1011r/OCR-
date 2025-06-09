from langchain_community.document_loaders import TextLoader, UnstructuredURLLoader
file_path = 'c:/Users/RAKSHAN/Downloads/Larsen.csv'

# Create a TextLoader instance
loader = TextLoader(file_path)

# Load the text data from the file
data = loader.load()

# Print the loaded data
for doc in data:
    print(doc)

# Example of how to use UnstructuredURLLoader if needed
urls = ['https://en.wikipedia.org/wiki/Larsen_%26_Toubro']
url_loader = UnstructuredURLLoader(urls)
url_data = url_loader.load()

# Print URL data
for url_doc in url_data:
    print(url_doc)
# Import necessary libraries
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Load the CSV file
df = pd.read_csv('c:/Users/RAKSHAN/Downloads/Larsen.csv')

# Check the column names in the DataFrame
print("Column Names:", df.columns)

# Strip any leading/trailing spaces from the column names
df.columns = df.columns.str.strip()
df['text'] = df[['ProjectName', 'Location', 'StartDate', 'EndDate', 'Branch']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
# Check for 'text' column and extract text if found
if 'text' in df.columns:
    text = df['text'].iloc[0]  # Get the first row of the text column
else:
    print("Error: 'text' column not found. Please check the column names.")
    text = ""  # Set to empty if the 'text' column is not found

# Verify that we have some text to work with
if text:
    # Finding the number of lines
    words = text.split("\n")
    print(f"Number of lines: {len(words)}")

    # Manual Chunks Splitting using CharacterTextSplitter
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=130,
        chunk_overlap=0
    )

    chunks = splitter.split_text(text)
    print(f"Number of chunks (manual): {len(chunks)}")

    # Print the length of each chunk for manual chunking
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1} length (manual): {len(chunk)}")

    # Recursive Chunk Splitting using RecursiveCharacterTextSplitter
    r_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],  # Define separators
        chunk_size=130,  # size of each chunk
        chunk_overlap=0,  # overlap between chunks
        length_function=len  # Function to calculate size
    )

    r_chunks = r_splitter.split_text(text)
    print(f"Number of chunks (recursive): {len(r_chunks)}")

    # Print the length of each chunk for recursive chunking
    for idx, chunk in enumerate(r_chunks):
        print(f"Chunk {idx+1} length (recursive): {len(chunk)}")
else:
    print("No text to process. Please ensure the correct column is selected.")
