import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# Import your existing classes
from chat_service import ChatService
from config.config import RAGConfig, VectorDBConfig
from rag_manager import RAGManager

class StreamlitChatInterface:
    def __init__(self):
        # Set page configuration
        st.set_page_config(
            page_title="UNILA Central de Serviços",
            page_icon="🎓",
            layout="wide"
        )

    def initialize_session_state(self):
        """Initialize or reset session state variables."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'last_response' not in st.session_state:
            st.session_state.last_response = None

    def setup_rag_manager(self):
        """Set up the RAG manager for the chat service."""
        # Load environment variables
        load_dotenv()

        # Initialize configurations
        vector_config = VectorDBConfig()
        rag_config = RAGConfig()
        
        # Initialize RAG manager
        rag_manager = RAGManager(
            pdf_path="data/catalogo_de_servicos.pdf",
            vector_config=vector_config,
            rag_config=rag_config,
            services_csv_path="data/catalogo_de_servicos.csv"
        )

        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found in environment variables")
            return None

        # Initialize chat service
        return ChatService(
            rag_manager=rag_manager,
            api_key=api_key
        )

    def render_sidebar(self):
        """Create the sidebar with additional information and controls."""
        st.sidebar.title("🎓 UNILA Central de Serviços")
        
        st.sidebar.header("Exemplos de Perguntas")
        example_queries = [
            "Como solicitar um computador?",
            "Qual o procedimento para atualização de dados no SIG?",
            "Como solicitar um ramal?",
            "Como solicitar troca de tonner?"
        ]
        
        for query in example_queries:
            if st.sidebar.button(query):
                st.session_state.user_input = query

        st.sidebar.header("Informações")
        st.sidebar.info(
            "Este é um assistente virtual para ajudar "
            "estudantes e servidores da UNILA com "
            "informações sobre serviços acadêmicos."
        )

    def display_context(self):
        """Display the context of the last response."""
        if st.session_state.last_response:
            with st.expander("📚 Contexto da Última Resposta"):
                response = st.session_state.last_response
                for idx, (context, score) in enumerate(zip(response['context'], response['confidence_scores']), 1):
                    st.markdown(f"**Trecho {idx}** (Relevância: {(1-score)*100:.1f}%)")
                    st.text(context)

    def run(self):
        """Main method to run the Streamlit interface."""
        # Initialize session state
        self.initialize_session_state()

        # Set up title and description
        st.title("🤖 Assistente Virtual da Central de Serviços")
        st.write("Bem-vindo! Como posso ajudar você hoje?")

        # Render sidebar
        self.render_sidebar()

        # Set up RAG manager and chat service
        chat_service = self.setup_rag_manager()
        if not chat_service:
            return

        # Context display section
        self.display_context()

        # Chat input
        user_input = st.chat_input("Digite sua pergunta aqui...")
        
        # If there's user input from sidebar button
        if hasattr(st.session_state, 'user_input'):
            user_input = st.session_state.user_input
            del st.session_state.user_input

        # Process user input
        if user_input:
            with st.spinner('Processando sua pergunta...'):
                try:
                    # Use asyncio to run the async method
                    response = asyncio.run(chat_service.process_query(user_input))
                    
                    # Store the last response
                    st.session_state.last_response = response
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'user': user_input,
                        'bot': response['answer']
                    })
                except Exception as e:
                    st.error(f"Ocorreu um erro: {str(e)}")

        # Display chat history
        for chat in st.session_state.chat_history:
            st.chat_message("human").write(chat['user'])
            st.chat_message("ai").write(chat['bot'])

def main():
    chat_interface = StreamlitChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main()