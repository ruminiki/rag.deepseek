import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ServiceDeskRAG:
    def __init__(self, knowledge_base_url: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG system with knowledge base URL and embedding model
        """
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.deepseek_api_key:
            st.error("DeepSeek API key not found in environment variables")
            return

        # Load embedding model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
        # Scrape and process knowledge base
        self.knowledge_base = self._scrape_knowledge_base(knowledge_base_url)
        self.embeddings = self._create_embeddings()
        
        # Create FAISS index for semantic search
        self.index = self._create_faiss_index()
    
    def _scrape_knowledge_base(self, url: str):
        """Scrape knowledge base from given URL"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            documents = []
            for article in soup.find_all('div', class_='service-article'):
                documents.append({
                    'title': article.find('h2').text,
                    'content': article.find('p').text
                })
            
            return documents
        except Exception as e:
            st.error(f"Error scraping knowledge base: {e}")
            return []
    
    def _create_embeddings(self):
        """Create embeddings for scraped documents"""
        embeddings = []
        for doc in self.knowledge_base:
            text = f"{doc['title']} {doc['content']}"
            
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _create_faiss_index(self):
        """Create FAISS index for efficient semantic search"""
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings)
        return index
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3):
        """Retrieve most relevant documents for a given query"""
        query_inputs = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        D, I = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        return [self.knowledge_base[i] for i in I[0]]
    
    def generate_response(self, query: str):
        """Generate response using DeepSeek API with retrieved context"""
        relevant_docs = self.retrieve_relevant_docs(query)
        
        context = "\n\n".join([f"Title: {doc['title']}\nContent: {doc['content']}" for doc in relevant_docs])
        
        prompt = f"Context: {context}\n\nUser Query: {query}\n\nGenerate a helpful and concise response based on the context:"
        
        response = self._call_deepseek_api(prompt)
        
        return response, relevant_docs
    
    def _call_deepseek_api(self, prompt: str):
        """Call DeepSeek API to generate response"""
        try:
            headers = {
                'Authorization': f'Bearer {self.deepseek_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'deepseek-chat',
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful service desk assistant.'},
                    {'role': 'user', 'content': prompt}
                ]
            }
            
            response = requests.post('https://api.deepseek.com/v1/chat/completions', 
                                     headers=headers, 
                                     json=payload)
            
            return response.json()['choices'][0]['message']['content']
        
        except Exception as e:
            return f"Error generating response: {e}"

def main():
    # Streamlit app configuration
    st.set_page_config(page_title="Service Desk Assistant", page_icon="🤖")
    st.title("🤖 Service Desk AI Assistant")
    
    # Knowledge base URL configuration
    KNOWLEDGE_BASE_URL = st.text_input(
        "Knowledge Base URL", 
        placeholder="Enter URL of your service desk knowledge base"
    )
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for additional controls
    st.sidebar.header("Assistant Settings")
    top_k = st.sidebar.slider("Number of Retrieved Documents", 1, 5, 3)
    
    # Initialize RAG system when URL is provided
    if KNOWLEDGE_BASE_URL:
        try:
            rag_system = ServiceDeskRAG(KNOWLEDGE_BASE_URL)
            
            # Chat input
            user_query = st.chat_input("Ask about our services...")
            
            if user_query:
                # Generate response
                response, relevant_docs = rag_system.generate_response(user_query)
                
                # Update chat history
                st.session_state.chat_history.append({"user": user_query, "bot": response})
                
                # Display chat history
                for chat in st.session_state.chat_history:
                    st.chat_message("human").write(chat["user"])
                    st.chat_message("assistant").write(chat["bot"])
                
                # Display retrieved documents
                with st.expander("Retrieved Knowledge Base Documents"):
                    for doc in relevant_docs:
                        st.write(f"**Title:** {doc['title']}")
                        st.write(f"**Content:** {doc['content']}")
        
        except Exception as e:
            st.error(f"Error initializing Service Desk Assistant: {e}")
    else:
        st.info("Please enter a knowledge base URL to get started.")

if __name__ == "__main__":
    main()