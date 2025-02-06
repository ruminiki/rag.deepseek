# chat_service.py
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from typing import Dict, Any

from src.rag_manager import RAGManager

class ChatService:
    def __init__(self, rag_manager: RAGManager, api_key: str):
        self.rag_manager = rag_manager
        self.llm = self._initialize_llm(api_key)
        self.memory = self._initialize_memory()
        self.chain = self._setup_chain()

    def _initialize_llm(self, api_key: str) -> GoogleGenerativeAI:
        return GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=self.rag_manager.rag_config.temperature,
            top_p=self.rag_manager.rag_config.top_p,
            max_output_tokens=self.rag_manager.rag_config.max_output_tokens
        )

    def _initialize_memory(self) -> ConversationBufferWindowMemory:
        return ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=self.rag_manager.rag_config.memory_window,
            input_key='question',
            output_key='answer'
        )

    def _setup_chain(self) -> ConversationalRetrievalChain:
        prompt = self._create_prompt_template()
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.rag_manager.vector_store.get_retriever(),
            memory=self.memory,
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
            },
            return_source_documents=True
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        # Custom prompt template
        CUSTOM_PROMPT = """Você é um assistente virtual amigável e profissional da Universidade Federal da Integração Latino-Americana (UNILA).

        **Serviços Disponíveis:**
        {services}

        **Contexto Relevante:**
        {context}

        **Histórico:**
        {chat_history}

        Pergunta atual: {question}

        Responda de forma estruturada:
        1. Forneça detalhes relevantes do serviço (prazos, requisitos, etc.)
        2. Mencione o público-alvo e restrições importantes
        3. Peça informações adicionais, se necessário
        4. Confirme se a resposta resolve a necessidade e obtenha novas informações
        5. Certifique-se que o usuário encontrou o que procurava e então oriente-o a abrir um chamado em https://servicos.unila.edu.br
        
        Por fim:
        1. Questione o usuário se ele está enfrentando problemas técnicos. Se sim, sugira procedimentos comuns para reparação do problema apresentado
        2. Sugira perguntas relevantes que o usuário possa fazer com base no contexto e histórico da conversa

        Se a informação não estiver disponível no contexto, diga: Para informações mais detalhadas sobre este serviço, por favor, visite a Central de Serviços em https://servicos.unila.edu.br/catalogo/

        Lembre-se: Seja preciso, mas mantenha um tom amigável e prestativo."""

        return PromptTemplate(
            input_variables=["context", "chat_history", "question", "services"],
            template=CUSTOM_PROMPT
        )

    async def process_query(self, query: str) -> Dict[str, Any]:
        try:
            context = self.rag_manager.vector_store.vectorstore.similarity_search(query)
            response = await self.chain.ainvoke({
                "question": query,
                "context": context,
                "services": ", ".join(self.rag_manager.get_available_services())
            })
            
            return {
                "answer": response.get('answer', 'Desculpe, não encontrei uma resposta precisa.'),
                "context": context,
                "sources": [doc.metadata.get('source', '') for doc in response.get('source_documents', [])]
            }
        except Exception as e:
            return {
                "answer": f"Erro ao processar sua pergunta: {str(e)}",
                "context": None,
                "sources": []
            }