# vector_store.py
from typing import Optional, List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import torch

from config import RAGConfig, VectorDBConfig

class VectorStore:
    def __init__(self, vector_config: VectorDBConfig, rag_config: RAGConfig):
        self.vector_config = vector_config
        self.rag_config = rag_config
        self.embeddings = self._initialize_embeddings()
        self.vectorstore: Optional[FAISS] = None

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        device = self.vector_config.device if torch.cuda.is_available() else "cpu"
        return HuggingFaceEmbeddings(
            model_name=self.vector_config.embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore

    def get_retriever(self):
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.rag_config.top_k_results,
                "lambda_mult": self.rag_config.mmr_lambda
            }
        )