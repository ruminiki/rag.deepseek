# document_processor.py
from abc import ABC, abstractmethod
from typing import List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from conf.config import VectorDBConfig

class DocumentProcessor(ABC):
    @abstractmethod
    def extract_text(self) -> str:
        pass

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass

class PDFProcessor(DocumentProcessor):
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )

    def extract_text(self, pdf_path: str) -> str:
        text = ""
        reader = PdfReader(pdf_path)
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def split_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)