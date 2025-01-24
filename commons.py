from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

def load_markdown_file(file_path):
    loader = UnstructuredMarkdownLoader(file_path)
    data = loader.load
    
    assert len(data) == 1
    assert isinstance(data[0], Document)
    
    return data

