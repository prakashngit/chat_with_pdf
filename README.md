# PDF Question Answering with RAG

PDF Question Answering with RAG is a Python application that allows users to ask questions about the content of PDF documents. It uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on the information contained in the ingested PDFs. This project demonstrates the power of combining document retrieval with language models to create an intelligent question-answering system. 

This project is part of the exercises for the LangChain- Develop LLM powered applications with LangChain course by Eden Marco, and is available via Udemy [here](https://www.udemy.com/course/langchain/learn/). See Section 5 in the course for Marco's version of the project. The main differences are:
-  I use Ollama instead of OpenAI for the embeddings
-  I use Chroma instead of FAISS for the vector store
-  I have added a simple Streamlit app to allow for easier testing of the RAG system.

For anyone looking to learn LangChain, I highly recommend Marco's course!

## Features

- PDF ingestion and embedding storage
- Question answering using RAG
- Comparison between RAG-based answers and direct LLM responses
- Simple Streamlit app for easy testing of the RAG system

## Technologies Used

- Python 3.x
- LangChain
- OpenAI's GPT models
- ChromaDB for vector storage
- Ollama for embeddings
- PyPDF for PDF parsing
- Streamlit for the web app

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   Create a `.env` file in the root directory and add your API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Run the Streamlit app:
   ```
   cd python
   streamlit run app.py
   ```

## License

This project is licensed under the Apache License, Version 2.0 (APL 2.0). See the [LICENSE](LICENSE) file for details.