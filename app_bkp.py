# Usage example
import os

from dotenv import load_dotenv
from chat_service import ChatService
from config.config import RAGConfig, VectorDBConfig
from rag_manager import RAGManager
import asyncio

# Load environment variables
load_dotenv()

async def run_chat():
    """Run the enhanced chat interface"""
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
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    # Initialize chat service
    chat_service = ChatService(
        rag_manager=rag_manager,
        api_key=api_key
    )
    
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
                print("- Como solicitar um computador?")
                print("- Qual o procedimento para atualização de dados no SIG?")
                print("- Como solicitar um ramal?")
                print("- Como solicitar troca de tonner?")
                continue
            
            print("\n⌛ Processando sua pergunta...")

            # Process query
            response = await chat_service.process_query(user_input)
            last_response = response
            
            # Display response
            print("\n🤖 Assistente:", response['answer'])

            # Show sources if available
            #print("\nFontes consultadas:", response['context'])
            #if response['sources']:
            #    print("\nFontes consultadas:")
            #    for source in response['sources']:
            #        print(f"- {source}")
            
        except Exception as e:
            print(f"\n❌ Ocorreu um erro: {str(e)}")
            print("Por favor, tente novamente ou contate o suporte técnico.")

if __name__ == "__main__":
    asyncio.run(run_chat())