from langchain_chroma import Chroma

def create_vector_store(documents, embeddings, persist_directory):
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    return vector_store

def load_vector_store(embeddings, persist_directory):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
