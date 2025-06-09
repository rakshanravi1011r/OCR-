import streamlit as st
from Retrieval import create_vector_db, get_qa_chain

st.title("LARSEN AND TOUBRO")
st.header("TITLE ")
st.write("Page content here")
st.header("Another Blog Post Title")
st.write("This is another example blog post. You can add more details here.")
st.sidebar.header("Chatbot")

# Input from the user
question = st.sidebar.text_input("Ask me anything:")

if question:
    # Call get_qa_chain without the unnecessary argument
    chain = get_qa_chain()  # Removed 'allow_dangerous_deserialization=True'

    # Get the response using the chain
    response = chain({"query": question})

    # Display the answer
    st.header("Answer:")
    st.write(response["result"])













