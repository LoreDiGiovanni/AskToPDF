from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os

PARSED_MD_PATH = "./parsed_document.md"

def load_and_parse_pdf(file_path):
    if PARSED_MD_PATH and os.path.exists(PARSED_MD_PATH) and os.path.getsize(PARSED_MD_PATH) > 0:
        with open(PARSED_MD_PATH, "r", encoding="utf-8") as f:
            markdown_text = f.read()
        return [Document(page_content=markdown_text)]
    else:
        parser = LlamaParse(result_type="markdown")
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
        return documents

def split_by_chapters(markdown_text):
    chapter_pattern = re.compile(r'(#+\s+.+?)(?=\n#|\Z)', re.DOTALL)
    chapters = chapter_pattern.findall(markdown_text)
    chapter_docs = []
    for chapter in chapters:
        header_lines = chapter.split('\n')[0]
        chapter_docs.append(Document(
            page_content=chapter,
            metadata={'chapter_title': header_lines.strip('#').strip()}
        ))
    return chapter_docs

def documents_to_chunks(documents):
    full_text = "\n".join([doc.text for doc in documents])
    with open(PARSED_MD_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)
    chapters = split_by_chapters(full_text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    final_docs = []
    for chapter in chapters:
        chunks = text_splitter.split_text(chapter.page_content)
        for chunk in chunks:
            final_docs.append(Document(
                page_content=chunk,
                metadata=chapter.metadata
            ))
    return final_docs
