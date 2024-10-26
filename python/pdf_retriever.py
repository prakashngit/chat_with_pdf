from dotenv import load_dotenv
import os


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


class PDFRetriever:
    def __init__(self, collection_name, chroma_persist_directory):
        
        chroma_vector_store = Chroma(embedding_function = OllamaEmbeddings(model="mxbai-embed-large"),
                            collection_name = collection_name, 
                            persist_directory = chroma_persist_directory)

        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        self.retrieval_chain_chroma = create_retrieval_chain(
            retriever=chroma_vector_store.as_retriever(), 
            combine_docs_chain=combine_documents_chain,
        )

    def chat(self, query):
        res = self.retrieval_chain_chroma.invoke(input= {"input": query})
        return res['answer']
    
    @staticmethod
    def print_answer(query, answer):
        print("Question: ", query)
        print("Answer: ", answer)
        print("-"*100, "\n")
    
    @staticmethod
    def query_llm_directly_with_no_rag(query):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        template = """Answer the question {input} to the best of your ability"""
        chain = PromptTemplate.from_template(template=template) | llm
        return chain.invoke(input={"input": query}).content

if __name__ == "__main__":
    print("Retrieving...")
    load_dotenv()
    
    print("Responses using RAG \n ", "-"*100, "\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_retriever = PDFRetriever(collection_name="react", chroma_persist_directory=os.path.join(script_dir, "./chroma_db"))

    query1 = "Explain in about 100 words what is the main idea of the paper?"
    PDFRetriever.print_answer(query1, pdf_retriever.chat(query1))

    query2 = "Does react user chain of thought prompting. Explain in about 100 words."
    PDFRetriever.print_answer(query2, pdf_retriever.chat(query2))

    query3 = "Can React be used without chain of thought prompting? Explain in about 100 words."
    PDFRetriever.print_answer(query3, pdf_retriever.chat(query3 ))

    query4 = "What problems does React solve that chain of thought prompting cannot? Explain in about 100 words."
    PDFRetriever.print_answer(query4, pdf_retriever.chat(query4))

    print("Responses using the LLM directly with no RAG \n ", "-"*100, "\n")

    PDFRetriever.print_answer(query1, PDFRetriever.query_llm_directly_with_no_rag(query1))
    PDFRetriever.print_answer(query2, PDFRetriever.query_llm_directly_with_no_rag(query2))
    PDFRetriever.print_answer(query3, PDFRetriever.query_llm_directly_with_no_rag(query3))
    PDFRetriever.print_answer(query4, PDFRetriever.query_llm_directly_with_no_rag(query4))
