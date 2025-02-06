import streamlit as st
import os
from dotenv import load_dotenv
from src.chat_service import ChatService
from conf.config import RAGConfig, VectorDBConfig
from src.rag_manager import RAGManager
import asyncio

# Load environment variables
load_dotenv()

# Initialize Streamlit session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'rag_manager' not in st.session_state:
    # Initialize configurations
    vector_config = VectorDBConfig()
    rag_config = RAGConfig()
    
    # Initialize RAG manager
    st.session_state.rag_manager = RAGManager(
        pdf_path="data/catalogo_de_servicos.pdf",
        vector_config=vector_config,
        rag_config=rag_config,
        services_csv_path="data/catalogo_de_servicos.csv"
    )

if 'chat_service' not in st.session_state:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    # Initialize chat service
    st.session_state.chat_service = ChatService(
        rag_manager=st.session_state.rag_manager,
        api_key=api_key
    )

# Set up the Streamlit page
st.set_page_config(page_title="Central de Serviços UNILA - Chat", page_icon="🎓")
st.title("💬 Chat da Central de Serviços UNILA")

# Add a sidebar with help information
with st.sidebar:
    st.header("ℹ️ Ajuda")
    st.write("Exemplos de perguntas que você pode fazer:")
    examples = [
        "Como solicitar um computador?",
        "Qual o procedimento para atualização de dados no SIG?",
        "Como solicitar um ramal?",
        "Como solicitar troca de tonner?"
    ]
    for example in examples:
        st.write(f"- {example}")

# Display chat history
#for message in st.session_state.chat_history:
#    with st.chat_message(message["role"]):
#        st.write(message["content"])
#        if message.get("context") and message["role"] == "assistant":
#            with st.expander("Ver contexto da resposta"):
#                for idx, (context, score) in enumerate(zip(message["context"], message["confidence_scores"]), 1):
#                    st.write(f"\nTrecho {idx} (Relevância: {(1-score)*100:.1f}%):")
#                    st.write(context)

# Chat input
if prompt := st.chat_input("Digite sua mensagem aqui..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Processando sua pergunta..."):
            # Create a new event loop for async call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                st.session_state.chat_service.process_query(prompt)
            )
            loop.close()

            st.write(response['answer'])
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response['answer'],
                "context": response['context']
                #"confidence_scores": response['confidence_scores']
            })

# Add a button to clear chat history
if st.button("Limpar Histórico"):
    st.session_state.chat_history = []
    st.rerun()

