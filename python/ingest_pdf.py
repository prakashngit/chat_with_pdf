from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import os 


def ingest_pdf(file_path, name = "default_collection"):
    """ Given a pdf file located under the file_path, 
    ingest the data into a local Chroma vector store. 
    Embeddings are generated using Ollama deployed locally.

    Pass in a name to create a new collection, or leave blank to add to the default collection.
    The vector store is persisted locally in ./chroma_db
    
    Returns the number of chunks embedded, the name of the collection, and the path to the local vector store.
    """
    
    # load the pdf file
    loader = PyPDFLoader(file_path, extract_images=False)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = text_splitter.split_documents(document)
    
    # embed the chunks
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_persist_directory = os.path.join(script_dir, "chroma_db")
    vector_store = Chroma.from_documents(texts, 
                                         embeddings, 
                                         collection_name = name, 
                                         persist_directory = chroma_persist_directory)
    chroma_client = vector_store._client
    collection = chroma_client.get_collection(name)
    return collection.count(), name, chroma_persist_directory

if __name__ == "__main__":
    print("Ingesting data...")
    load_dotenv()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "react.pdf")
    
    count, name, chroma_persist_directory = ingest_pdf(file_path, name="react")
    print(f"Ingested {count} chunks into collection {name}")
    print(f"Chroma vector store persisted in {chroma_persist_directory}")
    
    
    