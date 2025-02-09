from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads
from operator import itemgetter
from langchain.schema import StrOutputParser

def merge_retrieved_docs(documents: list[list]):
    flattened = [dumps(doc) for sublist in documents for doc in sublist]
    flattened_unique = list(set(flattened))
    return [loads(doc) for doc in flattened_unique]

def multi_query(llm, retriever):
    multy_query_template = """You are an A language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines.
    if it's not in English, translate. Original question: {question}"""
    return (
        {"question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(multy_query_template)
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
        | retriever.map()
        | merge_retrieved_docs
    )

def rag_chain(llm, retriever, template):
    return (
        {"context": retriever, "question": itemgetter("question")}
        | ChatPromptTemplate.from_template(template)
        | llm
        | StrOutputParser()
    )
