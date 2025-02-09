# RAG System for PDF Documents

This project implements a Retrieval-Augmented Generation (RAG) system to query PDF documents using advanced language models.

## Main Features

- Parsing PDF documents into markdown format  
- Splitting content into chapters and chunks  
- Creating a vector store for semantic search  
- Querying the document using natural language
    - Multi-query support to enhance search results  
- LLM models (OpenAI and DeepSeek) for output based on the content retrived

## Technologies Used

This project leverages various state-of-the-art technologies to ensure efficient document processing, semantic search, and interaction with language models.

- **LlamaParse**: For high-quality PDF parsing, we use **LlamaParse** by LlamaIndex. As stated by LlamaIndex:  
  *"At LlamaIndex we have a mission to connect your data to LLMs. A key factor in the effectiveness of presenting your data to LLMs is that it be easily understood by the model. Our experiments show that high-quality parsing makes a significant difference to the outcomes of your generative AI applications. So we compiled all of our expertise in document parsing into LlamaParse, to make it easy for you to get your data into the best possible shape for your LLMs."*  
  - [llamaa Parse](https://docs.cloud.llamaindex.ai/llamaparse/getting_started)

- **ChromaDB**: A vector database to store and retrieve document embeddings for fast and accurate semantic search.  
- **OpenAI & DeepSeek LLMs**: Used for natural language processing and generating responses based on retrieved document context.  
- **LangChain**: A framework that helps integrate LLMs with external data sources, enhancing retrieval-augmented generation (RAG) capabilities.  

## Requirements
- Python 3.8+  
- API keys for OpenAI and/or DeepSeek  
- API keys for llamaCloude, [llamaa Parse](https://docs.cloud.llamaindex.ai/llamaparse/getting_started)
 
## Run options
```bash
python main.py -f path/to/document.pdf -q "Your question here"
```
- Available parameters:
    -  -f/--file: Path to the PDF file to be processed
    -  -q/--query: Question to ask the system
    -  -k: Number of retrieved documents (default: 1)
    -  --persist-dir: Directory to save/load the vector store (default: ./chroma_db)
