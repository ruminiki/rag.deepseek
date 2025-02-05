# rag_manager.py
from pathlib import Path
from langchain.docstore.document import Document
import pandas as pd
from typing import List, Optional

from config.config import RAGConfig, VectorDBConfig
from src.document_processor import PDFProcessor
from src.vector_store import VectorStore

class RAGManager:
    def __init__(
        self,
        pdf_path: str,
        vector_config: VectorDBConfig,
        rag_config: RAGConfig,
        services_csv_path: Optional[str] = None
    ):
        self.pdf_processor = PDFProcessor(vector_config)
        self.vector_store = VectorStore(vector_config, rag_config)
        self.rag_config = rag_config
        self.services_df: Optional[pd.DataFrame] = None
        
        self._initialize_system(pdf_path, services_csv_path)

    def _initialize_system(self, pdf_path: str, services_csv_path: Optional[str]) -> None:
        # Process PDF
        text = self.pdf_processor.extract_text(pdf_path)
        chunks = self.pdf_processor.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Create vector store
        self.vector_store.create_vectorstore(documents)
        
        # Load services if CSV provided
        if services_csv_path:
            self._load_services(services_csv_path)

    def _load_services(self, csv_path: str) -> None:
        self.services_df = pd.read_csv(csv_path, sep='|', encoding='utf-8')

    def get_available_services(self) -> List[str]:
        if self.services_df is None:
            return []
        return self.services_df["Serviço"].dropna().unique().tolist()