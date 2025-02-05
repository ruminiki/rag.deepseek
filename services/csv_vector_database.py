import pandas as pd
import numpy as np
import faiss
import torch
import re
from typing import List, Dict, Any

# Portuguese NLP Libraries
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Transformers and Gemini imports
from transformers import AutoTokenizer, AutoModel

# Langchain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

class CSVVectorDatabase:
    def __init__(self,
                 csv_path: str = "data/catalogo_de_servicos.csv",
                 embedding_model = "neuralmind/bert-base-portuguese-cased"):
        # Initialize Portuguese Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
        # Portuguese stopwords
        self.stop_words = set(stopwords.words('portuguese'))
        
        # Initialize enhanced embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load and process data
        self.load_and_process_data(csv_path)
        
    def preprocess_text(self, text: str) -> str:
        # Custom tokenization to avoid NLTK issues
        def custom_tokenize(text: str) -> List[str]:
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation and special characters
            text = re.sub(r'[^\w\s]', '', text)
            
            # Split on whitespace
            tokens = text.split()

            return tokens
        
        # Tokenize
        tokens = custom_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def create_embeddings(self, documents: List[str]) -> np.ndarray:
        embeddings = []
        for doc in documents:
            # Preprocess text
            preprocessed_doc = self.preprocess_text(doc)
            
            # Tokenize
            inputs = self.tokenizer(
                preprocessed_doc, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512
            )
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def load_and_process_data(self, csv_path: str):
        # Read CSV
        self.raw_data = pd.read_csv(csv_path, sep='|', encoding='utf-8')
        
        # Create document texts
        documents = []
        for _, row in self.raw_data.iterrows():
            doc_text = (
                f"Serviço: {row['Serviço']} "
                f"Categoria: {row['Categoria']} "
                f"Descrição: {row['Descrição da requisição']} "
                f"Prioridade: {row['Prioridade']} "
                f"Horário: {row['Horário de atendimento']} "
                f"Tempo de solução: {row['Tempo de solução']} "
                f"Público-alvo: {row['Público-alvo']} "
                f"Termo de Uso: {row['Termo de Uso']}"
            )
            documents.append(doc_text)
        
        # Create embeddings
        self.document_embeddings = self.create_embeddings(documents)
        #print("Embbedings: ", self.document_embeddings)

        # Create FAISS index
        dimension = self.document_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.document_embeddings)
        
        # Create Langchain vector store
        langchain_docs = [
            Document(
                page_content=doc, 
                metadata={'source': row['Serviço']}
            ) for doc, row in zip(documents, self.raw_data.to_dict('records'))
        ]
        
        self.vectorstore = FAISS.from_documents(
            langchain_docs, 
            self.embeddings
        )
        
    def retrieve_relevant_context(self, query: str, top_k: int = 3):
        # Preprocess query
        preprocessed_query = self.preprocess_text(query)
        
        # Retrieve documents
        docs = self.vectorstore.similarity_search(preprocessed_query, k=top_k)
        
        return docs
    

    def available_services(self):
       csv_path: str = "data/catalogo_de_servicos.csv"
       
       return self.raw_data['Serviço'].unique().tolist()
