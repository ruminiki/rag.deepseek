import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Portuguese NLP Libraries
import nltk
from nltk.corpus import stopwords

from services.pdf_vector_database import PDFVectorDatabase

# Ensure stopwords are downloaded
try:
    stopwords.words('portuguese')
except LookupError:
   nltk.download('stopwords', quiet=True)

# Langchain imports
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

class RAGServiceChat:
    def __init__(self, vector_database: PDFVectorDatabase):
        """
        Initialize RAG system with Portuguese language model and knowledge base
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize Gemini API
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize Gemini LLM
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_key,
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=1024
        )
        
        # Enhanced memory with service context
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=4,
            input_key='question',
            output_key='answer'
        )

        self.vector_database = vector_database
        
    def setup_rag_chain(self):
        
        self.available_services = self.vector_database.available_services
        if self.available_services is None:
            self.available_services = []
        #print("Serviços: ", self.available_services)

        # Custom prompt template
        CUSTOM_PROMPT = """Você é um assistente virtual amigável e profissional da Universidade Federal da Integração Latino-Americana (UNILA).

            **Instruções:**
            1. Mantenha um tom acolhedor e profissional
            2. Forneça respostas diretas e objetivas
            3. Inclua detalhes específicos do contexto quando relevante
            5. Sugira até 3 serviços relacionados
            6. Peça informações adicionais se necessário
            7. Confirme se a resposta resolve a necessidade
            8. Responda diretamente à pergunta com emojis relevantes

            **Serviços Disponíveis:**
            {services}

            **Contexto Relevante:**
            {context}

            **Histórico:**
            {chat_history}

            Pergunta atual: {question}

            Responda de forma estruturada:
            1. Primeiro, aborde diretamente a pergunta do usuário
            2. Em seguida, forneça detalhes relevantes do serviço (prazos, requisitos, etc.)
            3. Se aplicável, mencione o público-alvo e restrições importantes
            4. Se o contato do usuário for por problema técnico, oriente-o com procedimentos comuns para reparação do problema.
            5. Certifique que o usuário encontrou o que procurava e então oriente-o a abrir um chamado em https://servicos.unila.edu.br

            Se a informação não estiver disponível no contexto, diga: Para informações mais detalhadas sobre este serviço, por favor, visite a Central de Serviços em https://servicos.unila.edu.br/catalogo/

            Lembre-se: Seja preciso, mas mantenha um tom amigável e prestativo.""".format(
                services=", ".join(self.available_services)
            )
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=CUSTOM_PROMPT
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_database.get_retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        # Create RAG chain (perguntas e respostas)
        #chain = RetrievalQAWithSourcesChain.from_chain_type(
        #    llm=self.llm,
        #    chain_type="stuff",
        #    retriever=retriever,
        #    memory=self.memory,
        #    chain_type_kwargs={
        #        "prompt": prompt,
        #        "document_variable_name": "context"
        #    },
        #    return_source_documents=True
        #)
        
        return chain
    
    async def process_query(self, query: str):
        
        # Get relevant context
        context = self.vector_database.retrieve_relevant_context(query)

        # Setup RAG chain
        chain = self.setup_rag_chain()
        
        #print("Context: ", context)

        try:
            # Get response
            response = await chain.ainvoke({"question": query, "context": context})
            
            return {
                "answer": response.get('answer', 'Desculpe, não encontrei uma resposta precisa.'),
                "context": context,
                "sources": [doc.metadata['source'] for doc in response.get('source_documents', [])]
            }
        except Exception as e:
            return {
                "answer": f"Erro ao processar sua pergunta: {str(e)}",
                "context": None,
                "sources": []
            }
