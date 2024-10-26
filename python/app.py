from dotenv import load_dotenv
load_dotenv()

from ingest_pdf import ingest_pdf   
from pdf_retriever import PDFRetriever
import streamlit as st
import tempfile
import os

@st.cache_resource
def cached_ingest_pdf(_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        result = ingest_pdf(tmp_file_path)
    finally:
        os.unlink(tmp_file_path)
    
    return result

st.title("RAG vs No RAG")
st.write("This app allows you to compare the performance of a RAG system vs a LLM that does not use RAG.")


if 'pdf_ingested' not in st.session_state:
    st.session_state.pdf_ingested = False

file = st.file_uploader("Upload a PDF file. The app will ingest the file and store it in a local vector store.", type="pdf")

# Create a button to process the uploaded file
if file is not None and st.button("Process PDF"):
    if not st.session_state.pdf_ingested:
        with st.spinner("Processing PDF..."):
            count, collection_name, chroma_persist_directory = cached_ingest_pdf(file)
        st.session_state.pdf_ingested = True
        st.session_state.rag_retriever = PDFRetriever(collection_name=collection_name, 
                                                      chroma_persist_directory=chroma_persist_directory)
        st.success(f"Successfully ingested the file into {count} chunks and stored into local vector store")
    else:
        st.warning("A PDF has already been processed. Please refresh the page to upload a new one.")
   

# ask questions about the uploaded PDF file. Each time the user asks a question, also give a option to the app whether to use the RAG or use LLM directly with no RAG     
if st.session_state.pdf_ingested:
    st.write("Ask a question about the uploaded PDF")

    query = st.text_input("Enter your question:")
    rag = st.checkbox("Use RAG")
        
    if query and st.button("Get Answer"):
        if rag:
            answer = st.session_state.rag_retriever.chat(query)
        else:
            answer = PDFRetriever.query_llm_directly_with_no_rag(query)
        st.write("Answer:", answer)


