import os
from document_processing import load_and_parse_pdf, documents_to_chunks
from vector_store import create_vector_store, load_vector_store
from llm_models import getOpenAI
from query_processing import multi_query, rag_chain
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import argparse

PERSIST_DIRECTORY = "./chroma_db"
PARSED_MD_PATH = "./data_md/parsed_document.md"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF documents and query them using RAG")
    parser.add_argument('-f', '--file', type=str, 
                       help='Path to the PDF file to process')
    parser.add_argument('-q', '--query', type=str, 
                       help='Query to ask the system')
    parser.add_argument('-k', type=int, default=1,
                       help='Number of documents to retrieve')
    parser.add_argument('--persist-dir', type=str, default=PERSIST_DIRECTORY,
                       help='Directory to store/load the vector store')
    args = parser.parse_args()

    load_dotenv()
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    llm = getOpenAI()

    if os.path.exists(args.persist_dir) and os.listdir(args.persist_dir):
            vector_store = load_vector_store(embeddings, args.persist_dir)
    else:
        print("Nessun database trovato, creazione nuovo database...")
        if os.path.exists(args.persist_dir):
            os.mkdir(args.persist_dir)
        documents = load_and_parse_pdf(args.file)
        langchain_documents = documents_to_chunks(documents)
        vector_store = create_vector_store(langchain_documents, embeddings, args.persist_dir)

    if not args.query:
        print("Error: No query provided. Use -q/--query to specify a question.")
        exit(1)
        
    retriever = vector_store.as_retriever(search_kwargs={"k": args.k})
    retrieval_chain = multi_query(llm, retriever)

    llm_template = """Answer the question based only on the following context:
    {context}
    Question: {question}"""

    rag_chain_instance = rag_chain(llm, retrieval_chain, llm_template)
    response = rag_chain_instance.invoke({"question": args.query})
    print(response)
