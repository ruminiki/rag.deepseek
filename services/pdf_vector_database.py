import os
import fitz  # PyMuPDF para extração de texto do PDF
import faiss
import pandas as pd
import torch
import numpy as np
import nltk
from nltk.corpus import stopwords
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Baixar stopwords em português
nltk.download("stopwords", quiet=True)

class PDFVectorDatabase:
    def __init__(self, pdf_path: str, embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Classe para processar um PDF e armazenar embeddings em um banco vetorial FAISS.
        
        Args:
            pdf_path (str): Caminho do arquivo PDF a ser processado.
            embedding_model (str): Modelo de embeddings para representar o texto.
        """
        self.pdf_path = pdf_path

        # Inicializar modelo de embeddings otimizado para português
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Melhor para busca semântica
        )

        # Processar e indexar o conteúdo do PDF
        self.text_chunks = self.extract_text_from_pdf()
        self.vectorstore = self.create_vectorstore()

    def extract_text_from_pdf(self) -> list:
        """Extrai o texto do PDF e divide em partes significativas."""
        text = ""
        with fitz.open(self.pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"

        # Divisão do texto em chunks de tamanho adequado para embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Tamanho do trecho para embedding
            chunk_overlap=100,  # Sobreposição para manter contexto
            length_function=len
        )
        text_chunks = text_splitter.split_text(text)
        print(f"🔹 {len(text_chunks)} trechos extraídos do PDF.")
        return text_chunks

    def create_vectorstore(self) -> FAISS:
        """Cria e armazena embeddings dos textos extraídos no FAISS."""
        docs = [Document(page_content=chunk) for chunk in self.text_chunks]
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        print("✅ Banco vetorial FAISS criado com sucesso!")

        # Configure retriever
        #self.retriever = self.vectorstore.as_retriever(
        #    search_kwargs={
        #        "k": 5,
        #        "fetch_k": 10
        #    }
        #)

        #self.retriever = self.vectorstore.as_retriever(
        #    search_type="similarity",
        #    search_kwargs={
        #        "k": 6,
        #        "distance_metric": "cosine",  # Options: cosine, euclidean, dot_product
        #        "fetch_k": 6,
        #        "maximal_distance": 0.5  # Filter out documents too far in vector space
        #    }
        #)
        self.base_retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "lambda_mult": 0.5}
        )

        return vectorstore

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> list:
        """
        Retorna os trechos mais relevantes do PDF com base na consulta.

        Args:
            query (str): Pergunta ou termo de busca do usuário.
            top_k (int): Número de trechos mais relevantes a retornar.

        Returns:
            list: Lista de trechos de texto mais relevantes.
        """
        docs = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]


    def get_retriever(self):
        return self.base_retriever
    
    def get_vector_store(self):
        return self.vectorstore

    def available_services(self):
        # Read CSV
        csv_path = "data/catalogo_de_servicos.csv"
        raw_data = pd.read_csv(csv_path, sep='|', encoding='utf-8')

        print("Serviços: ", raw_data["Serviço"])

        if raw_data is not None and "Serviço" in raw_data:
            return raw_data["Serviço"].dropna().unique().tolist()
        return []

# Exemplo de uso:
#if __name__ == "__main__":
#    pdf_path = "data/catalogo_de_servicos.pdf"
#    pdf_db = PDFVectorDatabase(pdf_path)#

    # Consulta de exemplo
#    query = "Como restaurar arquivos apagados?"
#    results = pdf_db.retrieve_relevant_context(query)

#    print("\n🔍 Trechos relevantes encontrados:")
#    for idx, res in enumerate(results, 1):
#        print(f"\n📌 Trecho {idx}:")
#        print(res)
