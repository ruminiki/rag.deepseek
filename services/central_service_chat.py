import pandas as pd
import numpy as np
import faiss
import torch
import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# Portuguese NLP Libraries
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Transformers and Gemini imports
import transformers
from transformers import AutoTokenizer, AutoModel

# Langchain imports
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

class RAGServiceChat2:
    def __init__(self,
                 csv_path: str = "data/catalogo_de_servicos.csv",
                 embedding_model = "neuralmind/bert-base-portuguese-cased"):
        """
        Initialize RAG system with Portuguese language model and knowledge base
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize Gemini API
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize Portuguese Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
        # Portuguese stopwords
        self.stop_words = set(stopwords.words('portuguese'))
        
        # Initialize Gemini LLM
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_key,
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=1024
        )
        
        # Initialize enhanced embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Set up enhanced memory with window buffer
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=3
        )
        
        # Load and process data
        self.load_and_process_data(csv_path)
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for Portuguese tokenization
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            str: Preprocessed text
        """
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
        """
        Create embeddings for documents using Portuguese BERT
        
        Args:
            documents (List[str]): List of documents to embed
        
        Returns:
            np.ndarray: Embeddings for documents
        """
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
        """
        Load and process CSV data for RAG
        
        Args:
            csv_path (str): Path to CSV file
        """
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
        """
        Retrieve context for a given query
        
        Args:
            query (str): User query
            top_k (int): Number of top documents to retrieve
        
        Returns:
            List[Dict]: Retrieved context documents
        """
        # Preprocess query
        preprocessed_query = self.preprocess_text(query)
        
        # Retrieve documents
        docs = self.vectorstore.similarity_search(preprocessed_query, k=top_k)
        
        return docs
    
    def setup_rag_chain(self):
        """
        Setup Retrieval-Augmented Generation Chain
        
        Returns:
            RetrievalQAWithSourcesChain: Configured RAG chain
        """
        # Custom prompt template
        CUSTOM_PROMPT = """Você é um assistente virtual amigável e profissional da Universidade Federal da Integração Latino-Americana (UNILA).

            Instruções importantes:
            1. Mantenha um tom acolhedor e profissional
            2. Forneça respostas diretas e objetivas
            3. Inclua detalhes específicos do contexto quando relevante
            4. Se houver ambiguidade, peça esclarecimentos
            5. Priorize informações sobre prazos e requisitos importantes

            Contexto do serviço:
            {context}

            Histórico da conversa:
            {chat_history}

            Pergunta atual: {question}

            Responda de forma estruturada:
            1. Primeiro, aborde diretamente a pergunta do usuário
            2. Em seguida, forneça detalhes relevantes do serviço (prazos, requisitos, etc.)
            3. Se aplicável, mencione o público-alvo e restrições importantes
            4. Se o contato do usuário for por problema técnico, oriente-o com procedimentos comuns para reparação do problema.
            5. Certifique que o usuário encontrou o que procurava e então oriente-o a abrir um chamado em https://servicos.unila.edu.br

            Se a informação não estiver disponível no contexto, diga: "Para informações mais detalhadas sobre este serviço, por favor, visite a Central de Serviços em https://servicos.unila.edu.br/catalogo/"

            Lembre-se: Seja preciso, mas mantenha um tom amigável e prestativo."""
        
        # Create prompt template
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=CUSTOM_PROMPT
        )
        
        # Configure retriever
        #retriever = self.vectorstore.as_retriever(
        #    search_kwargs={
        #        "k": 5,
        #        "fetch_k": 10
        #    }
        #)

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6,
                "distance_metric": "cosine",  # Options: cosine, euclidean, dot_product
                "fetch_k": 6,
                "maximal_distance": 0.5  # Filter out documents too far in vector space
            }
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        # Create RAG chain
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
        """
        Process user query and generate response
        
        Args:
            query (str): User input query
        
        Returns:
            Dict: Response with answer and context
        """
        # Setup RAG chain
        chain = self.setup_rag_chain()
        
        # Get relevant context
        context = self.retrieve_relevant_context(query)

        #print("Context: ", context)

        try:
            # Get response
            response = await chain.ainvoke({"question": query})
            
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


async def run_chat():
    """Run the enhanced chat interface"""
    rag_system = RAGServiceChat("data/catalogo_de_servicos.csv")
    
    print("\n🎓 Bem-vindo ao Chat da Central de Serviços da UNILA! 🎓")
    print("\nComandos disponíveis:")
    print("- Digite 'sair' para encerrar o chat")
    print("- Digite 'contexto' para ver o contexto da última resposta")
    print("- Digite 'ajuda' para ver exemplos de perguntas")
    print("-" * 50)
    
    last_response = None
    
    while True:
        try:
            user_input = input("\n👤 Você: ").strip()
            
            if user_input.lower() == 'sair':
                print("\n👋 Obrigado por usar o Chat da Central de Serviços! Tenha um ótimo dia!")
                break
                
            if user_input.lower() == 'contexto' and last_response:
                print("\n📚 Contexto usado na última resposta:")
                print("-" * 50)
                for idx, (context, score) in enumerate(zip(last_response['context'], last_response['confidence_scores']), 1):
                    print(f"\nTrecho {idx} (Relevância: {(1-score)*100:.1f}%):")
                    print(context)
                print("-" * 50)
                continue
                
            if user_input.lower() == 'ajuda':
                print("\n💡 Exemplos de perguntas que você pode fazer:")
                print("- Como solicito abono de faltas?")
                print("- Qual o procedimento para atualização cadastral?")
                print("- Quais os horários de atendimento da biblioteca?")
                print("- Como faço para solicitar histórico escolar?")
                continue
            
            print("\n⌛ Processando sua pergunta...")

            # Process query
            response = await rag_system.process_query(user_input)
            last_response = response
            
            # Display response
            print("\n🤖 Assistente:", response['answer'])

            # Show sources if available
            if response['sources']:
                print("\nFontes consultadas:")
                for source in response['sources']:
                    print(f"- {source}")
            
        except Exception as e:
            print(f"\n❌ Ocorreu um erro: {str(e)}")
            print("Por favor, tente novamente ou contate o suporte técnico.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_chat())